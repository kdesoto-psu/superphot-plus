import contextlib
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from dynesty import NestedSampler
from dynesty import utils as dyfunc
from scipy.optimize import curve_fit
from scipy.stats import truncnorm

from .constants import *  # pylint: disable=wildcard-import
from .file_paths import FIT_PLOTS_FOLDER


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def import_data(fn, t0_lim=None):
    """
    Import the datafile.
    """
    npy_array = np.load(fn)
    arr = npy_array['arr_0']

    ferr = arr[2]
    t = arr[0][ferr != "nan"].astype(float)
    f = arr[1][ferr != "nan"].astype(float)
    b = arr[3][ferr != "nan"]
    ferr = ferr[ferr != "nan"].astype(float)

    if t0_lim is not None:
        f = f[t <= t0_lim]
        b = b[t <= t0_lim]
        ferr = ferr[t <= t0_lim]
        t = t[t <= t0_lim]

    max_flux_loc =  t[b == "r"][np.argmax(f[b == "r"] - np.abs(ferr[b == "r"]))]
 
    t -= max_flux_loc # make relative

    return t, f, ferr, b


def trunc_gauss(quantile, clip_a, clip_b, mean, std):
    a, b = (clip_a - mean) / std, (clip_b - mean) / std
    return truncnorm.ppf(quantile, a, b, loc=mean, scale=std)


def params_valid(A, beta, gamma, t0, tau_rise, tau_fall):
    """
    Checks if params are valid given certain model constraints.
    """
    if tau_fall > 1. / beta:
        return False

    if gamma > (1. - beta * tau_fall) / beta:
        return False

    if tau_rise * (1. + np.exp(gamma / tau_rise)) < tau_fall:
        return False

    return True

def run_mcmc(fn, t0_lim=None, plot=False, rstate=None):
    """
    Run dynesty importance nested sampling on datafile. Returns
    set of equally weighted posteriors (sets of fit parameters).

    Parameters:
    fn: file name to run MCMC on
    t0_lim: ?
    plot: if yes, draw plots for the fits
    rstate: random state that is seeded. if none, use machine entropy.
    """
    ref_band_idx = 1 # red band # pylint: disable=unused-variable

    prefix = fn.split("/")[-1][:-4]

    print(prefix)
    n_params = 14

    tdata, fdata, ferrdata, bdata = import_data(fn, t0_lim)

    if (tdata[bdata == "r"] is None) or (len(tdata[bdata == "r"]) == 0):
        return None
    if (tdata[bdata == "g"] is None) or (len(tdata[bdata == "g"]) == 0):
        return None

    max_flux = np.max(fdata[bdata == "r"] - np.abs(ferrdata[bdata == "r"]))

    def flux_model(cube, t_data, b_data):

        A, beta, gamma, t0, tau_rise, tau_fall, es = cube[:7] # pylint: disable=unused-variable

        if not params_valid(A, beta, gamma, t0, tau_rise, tau_fall):
            return 1e10 * np.ones(len(t_data))

        phase = t_data - t0
        f_model = (
            A
            / (1.0 + np.exp(-phase / tau_rise))
            * (1.0 - beta * gamma)
            * np.exp((gamma - phase) / tau_fall)
        )
        f_model[phase < gamma] = (
            A
            / (1.0 + np.exp(-phase[phase < gamma] / tau_rise))
            * (1.0 - beta * phase[phase < gamma])
        )

        # for secondary band
        start_idx = 7
        A_b = A * cube[start_idx]
        beta_b = beta * cube[start_idx + 1]
        gamma_b = gamma * cube[start_idx + 2]
        t0_b = t0 * cube[start_idx + 3]
        tau_rise_b = tau_rise * cube[start_idx + 4]
        tau_fall_b = tau_fall * cube[start_idx + 5]

        if not params_valid(A_b, beta_b, gamma_b, t0_b, tau_rise_b, tau_fall_b):
            return 1e10 * np.ones(len(t_data))

        inc_band_ix = np.array(b_data) == "g"
        phase_b = (t_data - t0_b)[inc_band_ix]
        phase_b2 = (t_data - t0_b)[inc_band_ix & (t_data - t0_b < gamma_b)]

        f_model[inc_band_ix] = (
            A_b
            / (1.0 + np.exp(-phase_b / tau_rise_b))
            * (1.0 - beta_b * gamma_b)
            * np.exp((gamma_b - phase_b) / tau_fall_b)
        )
        f_model[inc_band_ix & (t_data - t0_b < gamma_b)] = (
            A_b / (1.0 + np.exp(-phase_b2 / tau_rise_b)) * (1.0 - phase_b2 * beta_b)
        )
        return f_model


    def create_prior(cube):
        """
        Creates prior for pymultinest, where each side
        of the "cube" is a value sampled between 0 and 1
        representing each parameter.
        """

        cube[0] = max_flux * 10 ** (
            trunc_gauss(cube[0], *PRIOR_A)
        )  # log-uniform for A from 1.0x to 16x of max flux
        cube[1] = trunc_gauss(
            cube[1], *PRIOR_BETA
        )  # beta UPDATED, looks more Lorentzian so widened by 1.5x
        cube[2] = 10 ** trunc_gauss(
            cube[2], *PRIOR_GAMMA
        )  # very broad Gaussian temporary solution for gamma
        max_flux_loc = tdata[
            np.argmax(fdata[bdata == "r"] - np.abs(ferrdata[bdata == "r"]))
        ]
        cube[3] = trunc_gauss(
            cube[3], np.amin(tdata) - 50.0, np.amax(tdata) + 50.0, max_flux_loc, 20.0
        )  # t0
        cube[4] = 10 ** (trunc_gauss(cube[4], *PRIOR_TAU_RISE))  # taurise, UPDATED
        cube[5] = 10 ** (trunc_gauss(cube[5], *PRIOR_TAU_FALL))  # tau fall UPDATED
        cube[6] = 10 ** (
            trunc_gauss(cube[6], *PRIOR_EXTRA_SIGMA)
        )  # lognormal for extrasigma, UPDATED


        # green band
        cube[7] = trunc_gauss(cube[7], *PRIOR_A_g)  # A UPDATED
        cube[8] = trunc_gauss(cube[8], *PRIOR_BETA_g)  # beta UPDATED
        cube[9] = trunc_gauss(cube[9], *PRIOR_GAMMA_g)  # gamma, GAUSSIAN not Lorentzian
        cube[10] = trunc_gauss(cube[10], *PRIOR_T0_g)  # t0 UPDATED
        cube[11] = trunc_gauss(cube[11], *PRIOR_TAU_RISE_g)  # taurise UPDATED, Gaussian
        cube[12] = trunc_gauss(cube[12], *PRIOR_TAU_FALL_g)  # taufall UPDATED
        cube[13] = trunc_gauss(
            cube[13], *PRIOR_EXTRA_SIGMA_g
        )  # extra sigma UPDATED, Gaussian

        return cube

    def create_logL(cube):
        """
        Define the log-likelihood function. Is proportional to
        chi-squared of data's fit to generated flux model.
        """
        f_model = flux_model(cube, tdata, bdata)
        extra_sigma_arr = np.ones(len(tdata)) * cube[6] * max_flux
        extra_sigma_arr[bdata == "g"] *= cube[13]

        sigma_sq = ferrdata**2 + extra_sigma_arr**2
        logL = np.sum(
            np.log(1.0 / np.sqrt(2.0 * np.pi * sigma_sq))
            - 0.5 * (f_model - fdata) ** 2 / sigma_sq
        )
        return logL

    st = time.time() # pylint: disable=unused-variable

    sampler = NestedSampler(
        create_logL, create_prior, n_params, sample="rwalk", bound="single", nlive=NLIVE,
        rstate=rstate
    )
    sampler.run_nested(maxiter=MAX_ITER, dlogz=DLOGZ, print_progress=False)
    res = sampler.results

    samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])

    eq_wt_samples = dyfunc.resample_equal(samples, weights)

    if plot:
        plt.errorbar(
            tdata[bdata == "g"],
            fdata[bdata == "g"],
            yerr=ferrdata[bdata == "g"],
            c="g",
            label="g",
            fmt="o",
        )
        plt.errorbar(
            tdata[bdata == "r"],
            fdata[bdata == "r"],
            yerr=ferrdata[bdata == "r"],
            c="r",
            label="r",
            fmt="o",
        )

        trange_fine = np.linspace(np.amin(tdata), np.amax(tdata), num=500)

        for sample in eq_wt_samples[:30]:
            plt.plot(
                trange_fine,
                flux_model(sample, trange_fine, ["g"] * len(trange_fine)),
                c="g",
                lw=1,
                alpha=0.1,
            )
            plt.plot(
                trange_fine,
                flux_model(sample, trange_fine, ["r"] * len(trange_fine)),
                c="r",
                lw=1,
                alpha=0.1,
            )

        plt.xlabel("MJD")
        plt.ylabel("Flux")
        plt.title(prefix)
        if t0_lim is None:
            plt.savefig(os.path.join(FIT_PLOTS_FOLDER, prefix+".png"))
        else:
            plt.savefig(os.path.join(FIT_PLOTS_FOLDER, prefix+"_%.02f.png" % t0))
        plt.close()

    return eq_wt_samples


def run_curve_fit(fn):
    ref_band_idx = 1 # red band # pylint: disable=unused-variable

    prefix = fn.split("/")[-1][:-4]

    print(prefix)
    n_params = 14 # pylint: disable=unused-variable

    tdata, fdata, ferrdata, bdata = import_data(fn)

    if (tdata[bdata == "r"] is None) or (len(tdata[bdata == "r"]) == 0):
        return None
    if (tdata[bdata == "g"] is None) or (len(tdata[bdata == "g"]) == 0):
        return None

    max_flux = np.max(fdata[bdata == "r"] - np.abs(ferrdata[bdata == "r"]))

    max_flux_loc =  tdata[bdata == "r"][np.argmax(fdata[bdata == "r"] - np.abs(ferrdata[bdata == "r"]))]
    
    #p0 = np.array([max_flux, 0.0052, 10.**1.1391, max_flux_loc, 10**0.5990, 10**1.4296])
    bounds = (
        [max_flux * 10 ** (-0.2), 0.0, -2.0, np.amin(tdata) - 50.0, -1.0, 0.5],
        [max_flux * 10 ** (1.2), 0.02, 2.5, np.amax(tdata) + 50.0, 3.0, 4.0],
    )
    p0 = np.array(
        [
            max_flux,
            0.0052,
            1.1391,
            max_flux_loc,
            0.5990,
            1.4296,
            1.0607,
            1.0424,
            np.log10(1.0075),
            1.0006,
            np.log10(0.9663),
            np.log10(0.5488),
        ]
    )


    def flux_model_smooth(t_data, A, beta, gamma, t0, tau_rise, tau_fall):
        """
        Tests the smooth model implemented in ALERCE's
        classifier
        """
        gamma = 10.**gamma
        tau_rise = 10.**tau_rise
        tau_fall = 10.**tau_fall

        sigma_arg = (t_data - gamma - t0) / 3.
        sigma = 1. / (1. + np.exp(-sigma_arg))
        if not params_valid(A, beta, gamma, t0, tau_rise, tau_fall):
            return 1e10 * np.ones(len(t_data))

        phase = t_data - t0
        f_model = (
            A
            / (1.0 + np.exp(-phase / tau_rise))
            * (1.0 - beta * gamma)
            * np.exp((gamma - phase) / tau_fall)
            * sigma
        )
        f_model += (
            A / (1.0 + np.exp(-phase / tau_rise)) * (1.0 - beta * phase) * (1.0 - sigma)
        )

        return f_model

    popt_r, pcov = curve_fit( # pylint: disable=unused-variable
        flux_model_smooth,
        tdata[bdata == "r"],
        fdata[bdata == "r"],
        p0=p0[:6],
        sigma=ferrdata[bdata == "r"],
        bounds=bounds,
        maxfev=100000,
    )

    p0_g = popt_r * p0[6:]
    p0_g[2] = popt_r[2] + p0[8]
    p0_g[5] = popt_r[5] + p0[-1]
    p0_g[4] = popt_r[4] + p0[-2]

    popt_g, pcov = curve_fit(
        flux_model_smooth,
        tdata[bdata == "g"],
        fdata[bdata == "g"],
        p0=p0_g,
        sigma=ferrdata[bdata == "g"],
        bounds=bounds,
        maxfev=100000,
    )

    print(popt_g, popt_r)

    prefix = fn.split("/")[-1][:-4]

    plt.errorbar(
        tdata[bdata == "g"],
        fdata[bdata == "g"],
        yerr=ferrdata[bdata == "g"],
        c="g",
        label="g",
        fmt="o",
    )
    plt.errorbar(
        tdata[bdata == "r"],
        fdata[bdata == "r"],
        yerr=ferrdata[bdata == "r"],
        c="r",
        label="r",
        fmt="o",
    )

    trange_fine = np.linspace(np.amin(tdata), np.amax(tdata), num=500)

    bdata = ["g"] * len(trange_fine)
    plt.plot(trange_fine, flux_model_smooth(trange_fine, *popt_g), c="g", lw=1)
    bdata = ["r"] * len(trange_fine)
    plt.plot(trange_fine, flux_model_smooth(trange_fine, *popt_r), c="r", lw=1)

    print(flux_model_smooth(1e20, *popt_r), flux_model_smooth(1e20, *popt_g))
    plt.xlabel("MJD")
    plt.ylabel("Flux")
    plt.title(prefix)

    plt.savefig("../figs/fits_curve_fit/"+prefix+".png")

    return popt_g, popt_r


def dynesty_single_file(test_fn, output_dir, skip_if_exists=True,rstate=None):
    os.makedirs(output_dir, exist_ok=True)
    prefix = test_fn.split("/")[-1][:-4]
    if skip_if_exists and os.path.exists(
        os.path.join(output_dir, f"{prefix}_eqwt.npz")
    ):
        return None

    eq_samples = run_mcmc(test_fn, plot=False, rstate=rstate)
    if eq_samples is None:
        return None
    print(np.mean(eq_samples, axis=0))

    np.savez_compressed(
        os.path.join(output_dir, f"{prefix}_eqwt_dynesty.npz"), eq_samples
    )
    