import os

import extinction
import numpy as np
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery

sfd = SFDQuery()

def get_band_extinctions(ra, dec):
    """
    Get green and red band extinctions in magnitudes for
    a single supernova LC based on RA and DEC.
    """
    #First look up the amount of mw dust at this location
    coords = SkyCoord(ra,dec, frame='icrs', unit="deg")
    Av_sfd = 2.742 * sfd(coords) # from https://dustmaps.readthedocs.io/en/latest/examples.html

    # for gr, the was are:
    band_wvs = 1./ (0.0001 * np.asarray([4741.64, 6173.23])) # in inverse microns

    #Now figure out how much the magnitude is affected by this dust
    ext_list = extinction.fm07(band_wvs, Av_sfd, unit='invum') # in magnitudes

    return ext_list

def calc_accuracy(pred_classes, test_labels):
    """
    Calculates the accuracy of the random forest after predicting
    all classes.
    
    Parameters
    ----------
    pred_classes : numpy array (int)
        classes predicted by MLP
    test_labels : numpy array (int)
        true spectroscopic classes
    """
    num_total = len(pred_classes)
    num_correct = np.sum(np.where(pred_classes == test_labels, 1, 0))
    return num_correct/num_total


def f1_score(pred_classes, true_classes, class_average=False):
    """
    Calculates the F1 score for the classifier. If class_average=True, then
    the macro-F1 is used. Else, uses the weighted-F1 score.
    
    Parameters
    ----------
    pred_classes : numpy array (int)
        classes predicted by MLP
    true_classes : numpy array (int)
        true spectroscopic classes
    class_average : bool
        Determines whether F1 score is weighted equally for each class,
        or by number of samples per class. Defaults to False.
    """
    samples_per_class = {}
    for c in true_classes:
        if c in samples_per_class:
            samples_per_class[c] += 1
        else:
            samples_per_class[c] = 1

    f1_sum = 0.
    for c in samples_per_class:
        tp = len(pred_classes[(pred_classes == c) & (true_classes == c)])
        purity = tp / len(pred_classes[pred_classes == c])
        completeness = tp / len(true_classes[true_classes == c])
        f1 = 2. * purity * completeness / (purity + completeness)
        if class_average:
            f1_sum += f1
        else:
            f1_sum += samples_per_class[c] * f1
    if class_average:
        return f1_sum / len(samples_per_class.keys())

    total_samples = np.sum(samples_per_class.values())
    return f1_sum / total_samples


def convert_mags_to_flux(m, merr, zp):
    fluxes = 10. ** (-1. * ( m - zp ) / 2.5)
    flux_unc = np.log(10.)/2.5 * fluxes * merr
    return fluxes, flux_unc


def flux_model(cube, t_data, b_data):
    """
    Given "cube" of fit parameters, returns the flux measurements for
    a given set of time and band data.
    """
    A, beta, gamma, t0, tau_rise, tau_fall, es = cube[:7] # pylint: disable=unused-variable

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

    inc_band_ix = (np.array(b_data) == "g") # pylint: disable=superfluous-parens
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


def calculate_chi_squareds(names, fit_dir, data_dirs):
    """
    Gets the chi-squared of posterior fits from
    the model parameters and original datafiles.
    """
    log_likelihoods = []
    for _, name in enumerate(names):
        data_fn = None
        for d in data_dirs:
            data_fn = os.path.join(d, name + ".npz")
            if os.path.exists(data_fn):
                break

        npy_array = np.load(data_fn)
        mjd, flux, flux_err, bands = npy_array['arr_0']

        flux_err = flux_err.astype(float)
        mjd = mjd.astype(float)[~np.isnan(flux_err)]
        flux = flux.astype(float)[~np.isnan(flux_err)]
        bands = bands[~np.isnan(flux_err)]
        flux_err = flux_err[~np.isnan(flux_err)]

        fit_fn = os.path.join(fit_dir, name +"_eqwt.npz")
        npy_array_fit = np.load(fit_fn)
        post_arr = npy_array_fit['arr_0']

        post_med = np.median(post_arr, axis=0)
        #print(post_med)

        model_f = flux_model(post_med, mjd, bands)
        extra_sigma_arr = np.ones(len(mjd)) * np.max(flux[bands == "r"]) * post_med[6]
        extra_sigma_arr[bands == "g"] *= post_med[-1]
        sigma_sq = extra_sigma_arr**2 + flux_err**2

        logL = np.sum(
            np.log(1.0 / np.sqrt(2.0 * np.pi * sigma_sq))
            - 0.5 * (flux - model_f) ** 2 / sigma_sq
        ) / len(mjd)
        log_likelihoods.append(logL)

    return np.array(log_likelihoods)
