import os

import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from jax.config import config
from numpyro.contrib.nested_sampling import NestedSampler
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.initialization import init_to_uniform

from .constants import *  # pylint: disable=wildcard-import
from .file_paths import FIT_PLOTS_FOLDER, FITS_DIR

config.update("jax_enable_x64", True)
numpyro.enable_x64()


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

    if (t[b == "r"] is None) or (len(t[b == "r"]) == 0):
        return None
    if (t[b == "g"] is None) or (len(t[b == "g"]) == 0):
        return None

    b = np.where(b == "r", 1, 0) # change to integers

    # sort
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    f = f[sort_idx]
    ferr = ferr[sort_idx]
    b = b[sort_idx]

    max_flux_loc =  t[b == 1][np.argmax(f[b == 1] - np.abs(ferr[b == 1]))]

    t -= max_flux_loc # make relative

    # separate r and g band points
    # necessary for static indexing for jax

    # pad data
    t_padded, f_padded, ferr_padded, b_padded = (
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    )

    for b_int in [0,1]:
        len_b = len(b[b == b_int])
        t_s = t[b == b_int]
        f_s = f[b == b_int]
        ferr_s = ferr[b == b_int]
        b_s = b[b == b_int]

        if len_b > PAD_SIZE:
            t_padded = np.append(t_padded, t_s[:PAD_SIZE])
            f_padded = np.append(f_padded, f_s[:PAD_SIZE])
            ferr_padded = np.append(ferr_padded, ferr_s[:PAD_SIZE])
            b_padded = np.append(b_padded, b_s[:PAD_SIZE])
        else:
            t_padded = np.append(t_padded, t_s)
            f_padded = np.append(f_padded, f_s)
            ferr_padded = np.append(ferr_padded, ferr_s)
            b_padded = np.append(b_padded, b_s)

            t_padded = np.append(t_padded, [5000] * (PAD_SIZE - len_b))
            f_padded = np.append(f_padded, [0.] * (PAD_SIZE - len_b))
            ferr_padded = np.append(ferr_padded, [1e10] * (PAD_SIZE - len_b))
            b_padded = np.append(b_padded, [b_int] * (PAD_SIZE - len_b) )

    return t_padded, f_padded, ferr_padded, b_padded


def trunc_norm(low, high, loc, scale):
    """
    Helper function for dist.TruncatedNormal()
    """
    return dist.TruncatedNormal(loc=loc, scale=scale, low=low, high=high)


def run_mcmc(fn, sampler="NUTS", t0_lim=None, plot=False):
    """
    Run dynesty importance nested sampling on datafile. Returns
    set of equally weighted posteriors (sets of fit parameters).
    """
    rng_key = random.PRNGKey(4)
    rng_key, rng_key_ = random.split(rng_key) # pylint: disable=unused-variable

    ref_band_idx = 1 # red band # pylint: disable=unused-variable

    #prefix = fn.split("/")[-1][:-4]

    #print(prefix)
    n_params = 14 # pylint: disable=unused-variable

    prefix = fn.split("/")[-1][:-4]
    tdata, fdata, ferrdata, bdata = import_data(fn, t0_lim)
    if tdata is None:
        return

    max_flux = np.max( fdata[PAD_SIZE:] - np.abs(ferrdata[PAD_SIZE:]) )
    inc_band_ix = np.arange( 0, PAD_SIZE )


    def jax_model(
        t=None, obsflux=None, uncertainties=None, max_flux=None, inc_band_ix=None
    ):
        A = max_flux * 10**numpyro.sample("logA", trunc_norm(*PRIOR_A))
        beta = numpyro.sample("beta", trunc_norm(*PRIOR_BETA))
        gamma = 10**numpyro.sample("log_gamma", trunc_norm(*PRIOR_GAMMA))
        t0 = numpyro.sample("t0", trunc_norm(*PRIOR_T0))
        tau_rise = 10**numpyro.sample("log_tau_rise", trunc_norm(*PRIOR_TAU_RISE))
        tau_fall = 10**numpyro.sample("log_tau_fall", trunc_norm(*PRIOR_TAU_FALL))
        extra_sigma = 10**numpyro.sample("log_extra_sigma", trunc_norm(*PRIOR_EXTRA_SIGMA))

        A_g = numpyro.sample("A_g", trunc_norm(*PRIOR_A_g))
        beta_g = numpyro.sample("beta_g", trunc_norm(*PRIOR_BETA_g))
        gamma_g = numpyro.sample("gamma_g", trunc_norm(*PRIOR_GAMMA_g))
        t0_g = numpyro.sample("t0_g", trunc_norm(*PRIOR_T0_g))
        tau_rise_g = numpyro.sample("tau_rise_g", trunc_norm(*PRIOR_TAU_RISE_g))
        tau_fall_g = numpyro.sample("tau_fall_g", trunc_norm(*PRIOR_TAU_FALL_g))
        extra_sigma_g = numpyro.sample("extra_sigma_g", trunc_norm(*PRIOR_EXTRA_SIGMA_g))
        """
        A_g = numpyro.param("A_g", 1.)
        beta_g = numpyro.param("beta_g", 1.)
        gamma_g = numpyro.param("gamma_g", 1.)
        t0_g = numpyro.param("t0_g", 1.)
        tau_rise_g = numpyro.param("tau_rise_g", 1.)
        tau_fall_g = numpyro.param("tau_fall_g", 1.)
        extra_sigma_g = numpyro.param("extra_sigma_g", 1.)
        """
        A_b = A * A_g # pylint: disable=unused-variable
        beta_b = beta * beta_g
        gamma_b = gamma * gamma_g
        t0_b = t0 * t0_g
        tau_rise_b = tau_rise * tau_rise_g
        tau_fall_b = tau_fall * tau_fall_g

        phase = t - t0
        flux_const = A / (1. + jnp.exp(-phase / tau_rise))
        sigmoid = 1 / (1 + jnp.exp(10.*(gamma - phase)))

        flux = flux_const * (
            (1 - sigmoid) * (1 - beta * phase)
            + sigmoid * (1 - beta * gamma) * jnp.exp(-(phase - gamma) / tau_fall)
        )

        # g band
        phase_b = (t - t0_b)[inc_band_ix]
        flux_const_b = A / (1. + jnp.exp(-phase_b / tau_rise_b))
        sigmoid_b = 1 / (1 + jnp.exp(10.*(gamma_b - phase_b)))

        flux = flux.at[inc_band_ix].set(
            flux_const_b
            * (
                (1 - sigmoid_b) * (1 - beta_b * phase_b)
                + sigmoid_b
                * (1 - beta_b * gamma_b)
                * jnp.exp(-(phase_b - gamma_b) / tau_fall_b)
            )
        )

        sigma_tot = jnp.sqrt(uncertainties**2 + extra_sigma**2)
        sigma_tot = sigma_tot.at[inc_band_ix].set(
            jnp.sqrt(
                uncertainties[inc_band_ix] ** 2 + extra_sigma_g**2 * extra_sigma**2
            )
        )

        obs = numpyro.sample("obs",dist.Normal(flux, sigma_tot),obs=obsflux) # pylint: disable=unused-variable


    def jax_guide(
        t=None, obsflux=None, uncertainties=None, max_flux=None, inc_band_ix=None # pylint: disable=unused-variable
    ):
        logA_mu = numpyro.param(
            "logA_mu",
            PRIOR_A[2],
            constraint=constraints.interval(PRIOR_A[0], PRIOR_A[1]),
        )
        logA_sigma = numpyro.param("logA_sigma", 1e-3, constraint=constraints.positive)
        numpyro.sample("logA", dist.Normal(logA_mu, logA_sigma))

        beta_mu = numpyro.param(
            "beta_mu",
            PRIOR_BETA[2],
            constraint=constraints.interval(PRIOR_BETA[0], PRIOR_BETA[1]),
        )
        beta_sigma = numpyro.param("beta_sigma", 1e-5, constraint=constraints.positive)
        numpyro.sample("beta", dist.Normal(beta_mu, beta_sigma))

        log_gamma_mu = numpyro.param(
            "log_gamma_mu",
            PRIOR_GAMMA[2],
            constraint=constraints.interval(PRIOR_GAMMA[0], PRIOR_GAMMA[1]),
        )
        log_gamma_sigma = numpyro.param(
            "log_gamma_sigma", 1e-3, constraint=constraints.positive
        )
        numpyro.sample("log_gamma", dist.Normal(log_gamma_mu, log_gamma_sigma))

        t0_mu = numpyro.param(
            "t0_mu",
            PRIOR_T0[2],
            constraint=constraints.interval(PRIOR_T0[0], PRIOR_T0[1]),
        )
        t0_sigma = numpyro.param("t0_sigma", 1e-3, constraint=constraints.positive)
        numpyro.sample("t0", dist.Normal(t0_mu, t0_sigma))

        log_tau_rise_mu = numpyro.param(
            "log_tau_rise_mu",
            PRIOR_TAU_RISE[2],
            constraint=constraints.interval(PRIOR_TAU_RISE[0], PRIOR_TAU_RISE[1]),
        )
        log_tau_rise_sigma = numpyro.param(
            "log_tau_rise_sigma", 1e-3, constraint=constraints.positive
        )
        numpyro.sample("log_tau_rise", dist.Normal(log_tau_rise_mu, log_tau_rise_sigma))

        log_tau_fall_mu = numpyro.param(
            "log_tau_fall_mu",
            PRIOR_TAU_FALL[2],
            constraint=constraints.interval(PRIOR_TAU_FALL[0], PRIOR_TAU_FALL[1]),
        )
        log_tau_fall_sigma = numpyro.param(
            "log_tau_fall_sigma", 1e-3, constraint=constraints.positive
        )
        numpyro.sample("log_tau_fall", dist.Normal(log_tau_fall_mu, log_tau_fall_sigma))

        log_extra_sigma_mu = numpyro.param(
            "log_extra_sigma_mu",
            PRIOR_EXTRA_SIGMA[2],
            constraint=constraints.interval(PRIOR_EXTRA_SIGMA[0], PRIOR_EXTRA_SIGMA[1]),
        )
        log_extra_sigma_sigma = numpyro.param(
            "log_extra_sigma_sigma", 1e-3, constraint=constraints.positive
        )
        numpyro.sample(
            "log_extra_sigma", dist.Normal(log_extra_sigma_mu, log_extra_sigma_sigma)
        )

        # aux bands

        Ag_mu = numpyro.param(
            "A_g_mu",
            PRIOR_A_g[2],
            constraint=constraints.interval(PRIOR_A_g[0], PRIOR_A_g[1]),
        )
        Ag_sigma = numpyro.param("A_g_sigma", 1e-3, constraint=constraints.positive)
        numpyro.sample("A_g", dist.Normal(Ag_mu, Ag_sigma))

        beta_g_mu = numpyro.param(
            "beta_g_mu",
            PRIOR_BETA_g[2],
            constraint=constraints.interval(PRIOR_BETA_g[0], PRIOR_BETA_g[1]),
        )
        beta_g_sigma = numpyro.param(
            "beta_g_sigma", 1e-3, constraint=constraints.positive
        )
        numpyro.sample("beta_g", dist.Normal(beta_g_mu, beta_g_sigma))

        gamma_g_mu = numpyro.param(
            "gamma_g_mu",
            PRIOR_GAMMA_g[2],
            constraint=constraints.interval(PRIOR_GAMMA_g[0], PRIOR_GAMMA_g[1]),
        )
        gamma_g_sigma = numpyro.param(
            "gamma_g_sigma", 1e-3, constraint=constraints.positive
        )
        numpyro.sample("gamma_g", dist.Normal(gamma_g_mu, gamma_g_sigma))

        t0_g_mu = numpyro.param(
            "t0_g_mu",
            PRIOR_T0_g[2],
            constraint=constraints.interval(PRIOR_T0_g[0], PRIOR_T0_g[1]),
        )
        t0_g_sigma = numpyro.param("t0_g_sigma", 1e-3, constraint=constraints.positive)
        numpyro.sample("t0_g", dist.Normal(t0_g_mu, t0_g_sigma))

        tau_rise_g_mu = numpyro.param(
            "tau_rise_g_mu",
            PRIOR_TAU_RISE_g[2],
            constraint=constraints.interval(PRIOR_TAU_RISE_g[0], PRIOR_TAU_RISE_g[1]),
        )
        tau_rise_g_sigma = numpyro.param(
            "tau_rise_g_sigma", 1e-3, constraint=constraints.positive
        )
        numpyro.sample("tau_rise_g", dist.Normal(tau_rise_g_mu, tau_rise_g_sigma))

        tau_fall_g_mu = numpyro.param(
            "tau_fall_g_mu",
            PRIOR_TAU_FALL_g[2],
            constraint=constraints.interval(PRIOR_TAU_FALL_g[0], PRIOR_TAU_FALL_g[1]),
        )
        tau_fall_g_sigma = numpyro.param(
            "tau_fall_g_sigma", 1e-3, constraint=constraints.positive
        )
        numpyro.sample("tau_fall_g", dist.Normal(tau_fall_g_mu, tau_fall_g_sigma))

        extra_sigma_g_mu = numpyro.param(
            "extra_sigma_g_mu",
            PRIOR_EXTRA_SIGMA_g[2],
            constraint=constraints.interval(
                PRIOR_EXTRA_SIGMA_g[0], PRIOR_EXTRA_SIGMA_g[1]
            ),
        )
        extra_sigma_g_sigma = numpyro.param(
            "extra_sigma_g_sigma", 1e-3, constraint=constraints.positive
        )
        numpyro.sample(
            "extra_sigma_g", dist.Normal(extra_sigma_g_mu, extra_sigma_g_sigma)
        )



    if sampler == "NUTS":
        kernel = NUTS(jax_model, init_strategy=init_to_uniform)

        num_samples = 300
        mcmc = MCMC(
            kernel,
            num_warmup=1000,
            num_samples=num_samples,
            num_chains=1,
            chain_method="parallel",
            jit_model_args=True,
        )

        #with numpyro.validation_enabled():
        res = mcmc.run( # pylint: disable=unused-variable
            rng_key,
            obsflux=fdata,
            t=tdata,
            uncertainties=ferrdata,
            max_flux=max_flux,
            inc_band_ix=inc_band_ix,
        )

        #mcmc.print_summary()
        posterior_samples = mcmc.get_samples()

    elif sampler == "nested":
        ns = NestedSampler(jax_model, constructor_kwargs=None)
        ns.run(
            random.PRNGKey(1),
            obsflux=fdata,
            t=tdata,
            uncertainties=ferrdata,
            max_flux=max_flux,
            inc_band_ix=inc_band_ix,
        )
        posterior_samples = ns.get_samples(random.PRNGKey(3), num_samples=num_samples)

    elif sampler == "svi":
        optimizer = numpyro.optim.Adam(step_size=0.001)
        svi = SVI(jax_model, jax_guide, optimizer, loss=Trace_ELBO())
        num_iter = 10000
        with numpyro.validation_enabled():
            svi_result = svi.run(
                random.PRNGKey(1),
                num_iter,
                stable_update=True,
                obsflux=fdata,
                t=tdata,
                uncertainties=ferrdata,
                max_flux=max_flux,
                inc_band_ix=inc_band_ix,
            )
        params = svi_result.params
        posterior_samples = {}
        for p in params:
            if p[-2:] == "mu":
                posterior_samples[p[:-3]] = np.random.normal(
                    loc=params[p], scale=params[p[:-2] + "sigma"], size=100
                )

    else:
        return None


    """
    predictive = Predictive(jax_model, posterior_samples, infer_discrete=False)
    
    discrete_samples = predictive(random.PRNGKey(1), 
                       t=tdata_stacked, 
                       uncertainties=ferrdata_stacked, 
                       max_flux=max_flux, 
                       inc_band_ix=inc_band_ix)
    
    print(discrete_samples.keys())
    """
    if plot:
        plt.hist(posterior_samples['log_tau_fall'].flatten(),bins=10)
        plt.savefig('test_hist.png')
        plt.close()

        post_reformatted = {}
        for p in posterior_samples:
            post_reformatted[p] = np.array([posterior_samples[p],])

        az.plot_trace(post_reformatted, compact=True)
        plt.savefig('test_trace.png')
        plt.close()

        ignore_idx = (ferrdata == 1e10) # pylint: disable=superfluous-parens
        tdata = tdata[~ignore_idx]
        fdata = fdata[~ignore_idx]
        ferrdata = ferrdata[~ignore_idx]
        bdata = bdata[~ignore_idx]

        model_i = np.array(
            [
                {k: posterior_samples[k][j] for k in posterior_samples.keys()}
                for j in range(len(posterior_samples["log_tau_fall"]))
            ]
        )

        plt.errorbar(
            tdata[bdata == 0],
            fdata[bdata == 0],
            yerr=ferrdata[bdata == 0],
            c="g",
            label="g",
            fmt="o",
        )
        plt.errorbar(
            tdata[bdata == 1],
            fdata[bdata == 1],
            yerr=ferrdata[bdata == 1],
            c="r",
            label="r",
            fmt="o",
        )

        trange_fine = np.linspace(np.amin(tdata), np.amax(tdata), num=500)

        for sample in model_i[:30]:
            plt.plot(
                trange_fine,
                flux_from_posteriors(trange_fine, sample, max_flux)[0],
                c="g",
                lw=1,
                alpha=0.1,
            )
            plt.plot(
                trange_fine,
                flux_from_posteriors(trange_fine, sample, max_flux)[1],
                c="r",
                lw=1,
                alpha=0.1,
            )

        plt.xlabel("MJD")
        plt.ylabel("Flux")
        plt.title(prefix)

        if t0_lim is None:
            plt.savefig(os.path.join(FIT_PLOTS_FOLDER, "%s.pdf" % prefix))
        else:
            plt.savefig(os.path.join(FIT_PLOTS_FOLDER, "%s_%.02f.pdf" % (prefix,t0)))
        plt.close()

    param_list = [
        "logA",
        "beta",
        "log_gamma",
        "t0",
        "log_tau_rise",
        "log_tau_fall",
        "log_extra_sigma",
        "A_g",
        "beta_g",
        "gamma_g",
        "t0_g",
        "tau_rise_g",
        "tau_fall_g",
        "extra_sigma_g",
    ]

    post_reformatted_for_save = []
    for p in param_list:
        if p == "logA":
            post_reformatted_for_save.append(max_flux * 10**posterior_samples[p])
        elif p[:3] == "log":
            post_reformatted_for_save.append(10**posterior_samples[p])
        else:
            post_reformatted_for_save.append(posterior_samples[p])

    return np.array(post_reformatted_for_save).T


def run_mcmc_batch(fns, t0_lim=None, plot=False):
    """
    Run dynesty importance nested sampling on datafile. Returns
    set of equally weighted posteriors (sets of fit parameters).
    """
    rng_key = random.PRNGKey(4)
    rng_key, rng_key_ = random.split(rng_key) # pylint: disable=unused-variable

    ref_band_idx = 1 # red band # pylint: disable=unused-variable

    #prefix = fn.split("/")[-1][:-4]

    #print(prefix)
    n_params = 14 # pylint: disable=unused-variable

    tdata_stacked = []
    fdata_stacked = []
    ferrdata_stacked = []
    bdata_stacked = []
    prefixes = []

    for fn in fns:
        prefixes.append(fn.split("/")[-1][:-4])
        tdata, fdata, ferrdata, bdata = import_data(fn, t0_lim)
        if tdata is None:
            continue
        tdata_stacked.append(tdata)
        fdata_stacked.append(fdata)
        ferrdata_stacked.append(ferrdata)
        bdata_stacked.append(bdata)

    tdata_stacked = np.array(tdata_stacked)
    fdata_stacked = np.array(fdata_stacked)
    ferrdata_stacked = np.array(ferrdata_stacked)
    bdata_stacked = np.array(bdata_stacked)

    max_flux = np.max(fdata_stacked[:,PAD_SIZE:] - np.abs(ferrdata_stacked[:,PAD_SIZE:]), axis=1)

    inc_band_ix = np.arange(0,PAD_SIZE)

    N = len(tdata_stacked)
    print(N)

    def jax_model(
        t=None, obsflux=None, uncertainties=None, max_flux=None, inc_band_ix=None
    ):
        with numpyro.plate('components', N) as sn_index: # pylint: disable=unused-variable
            A = max_flux * 10**numpyro.sample("logA", trunc_norm(*PRIOR_A))
            beta = numpyro.sample("beta", trunc_norm(*PRIOR_BETA))
            gamma = 10**numpyro.sample("log_gamma", trunc_norm(*PRIOR_GAMMA))
            t0 = numpyro.sample("t0", trunc_norm(-100., 200., 0., 20.))
            tau_rise = 10**numpyro.sample("log_tau_rise", trunc_norm(*PRIOR_TAU_RISE))
            tau_fall = 10**numpyro.sample("log_tau_fall", trunc_norm(*PRIOR_TAU_FALL))
            extra_sigma = 10**numpyro.sample("log_extra_sigma", trunc_norm(*PRIOR_EXTRA_SIGMA))

            A_g = numpyro.sample("A_g", trunc_norm(*PRIOR_A_g))
            beta_g = numpyro.sample("beta_g", trunc_norm(*PRIOR_BETA_g))
            gamma_g = numpyro.sample("gamma_g", trunc_norm(*PRIOR_GAMMA_g))
            t0_g = numpyro.sample("t0_g", trunc_norm(*PRIOR_T0_g))
            tau_rise_g = numpyro.sample("tau_rise_g", trunc_norm(*PRIOR_TAU_RISE_g))
            tau_fall_g = numpyro.sample("tau_fall_g", trunc_norm(*PRIOR_TAU_FALL_g))
            extra_sigma_g = numpyro.sample("extra_sigma_g", trunc_norm(*PRIOR_EXTRA_SIGMA_g))

        A_b = A * A_g # pylint: disable=unused-variable
        beta_b = beta * beta_g
        gamma_b = gamma * gamma_g
        t0_b = t0 * t0_g
        tau_rise_b = tau_rise * tau_rise_g
        tau_fall_b = tau_fall * tau_fall_g

        phase = t - t0[:, np.newaxis]
        flux_const = A[:, np.newaxis] / (
            1.0 + jnp.exp(-phase / tau_rise[:, np.newaxis])
        )
        sigmoid = 1 / (1 + jnp.exp(10.0 * (gamma[:, np.newaxis] - phase)))

        flux = flux_const * (
            (1 - sigmoid) * (1 - beta[:, np.newaxis] * phase)
            + sigmoid
            * (1 - beta[:, np.newaxis] * gamma[:, np.newaxis])
            * jnp.exp(-(phase - gamma[:, np.newaxis]) / tau_fall[:, np.newaxis])
        )

        # g band
        phase_b = (t - t0_b[:, np.newaxis])[:, inc_band_ix]
        flux_const_b = A[:, np.newaxis] / (
            1.0 + jnp.exp(-phase_b / tau_rise_b[:, np.newaxis])
        )
        sigmoid_b = 1 / (1 + jnp.exp(10.0 * (gamma_b[:, np.newaxis] - phase_b)))

        flux = flux.at[:, inc_band_ix].set(
            flux_const_b
            * (
                (1 - sigmoid_b) * (1 - beta_b[:, np.newaxis] * phase_b)
                + sigmoid_b
                * (1 - beta_b[:, np.newaxis] * gamma_b[:, np.newaxis])
                * jnp.exp(
                    -(phase_b - gamma_b[:, np.newaxis]) / tau_fall_b[:, np.newaxis]
                )
            )
        )

        sigma_tot = jnp.sqrt(uncertainties**2 + extra_sigma[:, np.newaxis] ** 2)
        sigma_tot = sigma_tot.at[:, inc_band_ix].set(
            jnp.sqrt(
                uncertainties[:, inc_band_ix] ** 2
                + extra_sigma_g[:, np.newaxis] ** 2 * extra_sigma[:, np.newaxis] ** 2
            )
        )

        obs = numpyro.sample("obs", dist.Normal(flux, sigma_tot), obs=obsflux) # pylint: disable=unused-variable

    kernel = NUTS(jax_model, init_strategy=init_to_sample)
    num_samples = 100
    mcmc = MCMC(
        kernel,
        num_warmup=100,
        num_samples=num_samples,
        num_chains=1,
        chain_method="parallel",
    )  # jit_model_args=True)

    #with numpyro.validation_enabled():
    res = mcmc.run( # pylint: disable=unused-variable
        rng_key,
        obsflux=fdata_stacked,
        t=tdata_stacked,
        uncertainties=ferrdata_stacked,
        max_flux=max_flux,
        inc_band_ix=inc_band_ix,
    )

    #mcmc.print_summary()
    posterior_samples = mcmc.get_samples()
    """
    predictive = Predictive(jax_model, posterior_samples, infer_discrete=False)
    
    discrete_samples = predictive(
        random.PRNGKey(1),
        t=tdata_stacked,
        uncertainties=ferrdata_stacked,
        max_flux=max_flux,
        inc_band_ix=inc_band_ix,
    )
    
    print(discrete_samples.keys())
    """
    plt.hist(posterior_samples['log_tau_fall'][:,0],bins=10)
    plt.savefig('test_hist.png')
    plt.close()

    post_reformatted = {}
    for p in posterior_samples:
        post_reformatted[p] = np.array([posterior_samples[p],])

    az.plot_trace(post_reformatted, compact=True)
    plt.savefig('test_trace.png')
    plt.close()

    if plot:
        for i in range(len(tdata_stacked)):

            ignore_idx = (ferrdata_stacked[i] == 1e10) # pylint: ignore-superfluous parens
            tdata = tdata_stacked[i][~ignore_idx]
            fdata = fdata_stacked[i][~ignore_idx]
            ferrdata = ferrdata_stacked[i][~ignore_idx]
            bdata = bdata_stacked[i][~ignore_idx]

            model_i = np.array(
                [
                    {k: posterior_samples[k][j, i] for k in posterior_samples.keys()}
                    for j in range(len(posterior_samples["log_tau_fall"]))
                ]
            )

            plt.errorbar(
                tdata[bdata == 0],
                fdata[bdata == 0],
                yerr=ferrdata[bdata == 0],
                c="g",
                label="g",
                fmt="o",
            )
            plt.errorbar(
                tdata[bdata == 1],
                fdata[bdata == 1],
                yerr=ferrdata[bdata == 1],
                c="r",
                label="r",
                fmt="o",
            )

            trange_fine = np.linspace(np.amin(tdata), np.amax(tdata), num=500)

            for sample in model_i[:30]:
                plt.plot(
                    trange_fine,
                    flux_from_posteriors(trange_fine, sample, max_flux[i])[0],
                    c="g",
                    lw=1,
                    alpha=0.1,
                )
                plt.plot(
                    trange_fine,
                    flux_from_posteriors(trange_fine, sample, max_flux[i])[1],
                    c="r",
                    lw=1,
                    alpha=0.1,
                )

            plt.xlabel("MJD")
            plt.ylabel("Flux")
            plt.title(prefixes[i])
            if t0_lim is None:
                plt.savefig(os.path.join(FIT_PLOTS_FOLDER, "%s.pdf" % prefixes[i]))
            else:
                plt.savefig(os.path.join(FIT_PLOTS_FOLDER, "%s_%.02f.pdf" % (prefixes[i],t0)))
            plt.close()

    return posterior_samples


def flux_from_posteriors(t, params, max_flux):
    logA, beta, log_gamma = params['logA'], params['beta'], params['log_gamma']
    t0, log_tau_rise, log_tau_fall, log_extra_sigma = (
        params["t0"],
        params["log_tau_rise"],
        params["log_tau_fall"],
        params["log_extra_sigma"],
    )

    A = max_flux * 10**logA
    gamma = 10**log_gamma
    tau_rise = 10**log_tau_rise
    tau_fall = 10**log_tau_fall
    extra_sigma = 10**log_extra_sigma # pylint: disable=unused-variable

    A_g, beta_g, gamma_g = params['A_g'], params['beta_g'], params['gamma_g']
    t0_g, tau_rise_g, tau_fall_g, extra_sigma_g = ( # pylint: disable=unused-variable
        params["t0_g"],
        params["tau_rise_g"],
        params["tau_fall_g"],
        params["extra_sigma_g"],
    )

    A_b = A * A_g # pylint: disable=unused-variable
    beta_b = beta * beta_g
    gamma_b = gamma * gamma_g
    t0_b = t0 * t0_g
    tau_rise_b = tau_rise * tau_rise_g
    tau_fall_b = tau_fall * tau_fall_g

    phase = t - t0
    flux_const = A / (1. + jnp.exp(-phase / tau_rise))
    sigmoid = 1 / (1 + jnp.exp(10.*(gamma - phase)))

    flux_r = flux_const * (
        (1 - sigmoid) * (1 - beta * phase)
        + sigmoid * (1 - beta * gamma) * jnp.exp(-(phase - gamma) / tau_fall)
    )

    # g band
    phase_b = t - t0_b
    flux_const_b = A / (1. + jnp.exp(-phase_b / tau_rise_b))
    sigmoid_b = 1 / (1 + jnp.exp(10.*(gamma_b-phase_b)))

    flux_g = flux_const_b * (
        (1 - sigmoid_b) * (1 - beta_b * phase_b)
        + sigmoid_b
        * (1 - beta_b * gamma_b)
        * jnp.exp(-(phase_b - gamma_b) / tau_fall_b)
    )

    return flux_g, flux_r


def main_loop_directory(test_fns, output_dir=FITS_DIR):
    #try:
    os.makedirs(output_dir, exist_ok=True)
    #prefix = test_fn.split("/")[-1][:-4]
    #if os.path.exists(output_dir + str(prefix) + '_eqwt.npz'):
    #    return None

    eq_samples = run_mcmc_batch(test_fns, plot=True)

    if eq_samples is None:
        return None
    print(np.mean(eq_samples['log_tau_fall']))

    return None

    np.savez_compressed(output_dir + str(prefix) + '_eqwt.npz', eq_samples)
    #except:
    #    print("skipped")
    #    return None


def numpyro_single_file(test_fn, output_dir=FITS_DIR, sampler="svi"):
    os.makedirs(output_dir, exist_ok=True)

    eq_samples = run_mcmc(test_fn, sampler=sampler, plot=False)

    if eq_samples is None:
        return None

    print(np.mean(eq_samples, axis=0))
    prefix = test_fn.split("/")[-1][:-4]

    np.savez_compressed(os.path.join(output_dir , f"{prefix}_eqwt_{sampler}.npz" ), eq_samples)
    
    return None
