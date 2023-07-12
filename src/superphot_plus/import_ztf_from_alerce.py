import csv
import glob
import os
import zipfile

import extinction
import numpy as np
from alerce.core import Alerce
from antares_client.search import get_by_ztf_object_id
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery

from .utils import convert_mags_to_flux

alerce = Alerce()
MIN_PER_FILTER = 5


def get_band_extinctions(ra, dec):
    """
    Get green and red band extinctions in magnitudes for
    a single supernova LC based on RA and DEC.
    """
    sfd = SFDQuery()
    #First look up the amount of mw dust at this location
    coords = SkyCoord(ra,dec, frame='icrs', unit='deg')
    Av_sfd = 2.742 * sfd(coords) # from https://dustmaps.readthedocs.io/en/latest/examples.html

    # for gr, the was are:
    band_wvs = 1./ (0.0001 * np.asarray([4741.64, 6173.23])) # in inverse microns

    #Now figure out how much the magnitude is affected by this dust
    ext_list = extinction.fm07(band_wvs, Av_sfd, unit='invum') # in magnitudes

    return ext_list


def import_lc(fn):
    """
    Import single file, but only the points from a single telescope,
    in only g and r bands. 
    """
    ra = None
    dec = None
    if not os.path.exists(fn):
        return [None,] * 6
    with open(fn, 'r') as csv_f:
        csvreader = csv.reader(csv_f, delimiter=',')
        row_intro = next(csvreader)

        ra_idx = row_intro.index("ra")
        dec_idx = row_intro.index("dec")
        b_idx = row_intro.index("fid")
        f_idx = row_intro.index("magpsf")
        ferr_idx = row_intro.index("sigmapsf")

        flux = []
        flux_err = []
        mjd = []
        bands = []

        for row in csvreader:

            if ra is None:
                ra = float(row[ra_idx])
                dec = float(row[dec_idx])
                #ra = float(row[19])
                #dec = float(row[20])
                try:
                    g_ext, r_ext = get_band_extinctions(ra, dec)
                except:
                    return [None,] * 6
            if int(row[b_idx]) == 2:
                flux.append(float(row[f_idx]) - r_ext)
                bands.append("r")
            elif int(row[b_idx]) == 1:
                flux.append(float(row[f_idx]) - g_ext)
                bands.append("g")
            else:
                continue
            mjd.append(float(row[1]))
            flux_err.append(float(row[ferr_idx]))


    sort_idx = np.argsort(np.array(mjd))
    t = np.array(mjd)[sort_idx].astype(float)
    m = np.array(flux)[sort_idx].astype(float)
    merr = np.array(flux_err)[sort_idx].astype(float)
    b = np.array(bands)[sort_idx]

    t = t[merr != np.nan]
    m = m[merr != np.nan]
    b = b[merr != np.nan]
    merr = merr[merr != np.nan]

    f, ferr = convert_mags_to_flux(m, merr, 26.3)
    t, f, ferr, b = clip_lightcurve_end(t, f, ferr, b)
    snr = np.abs(f / ferr)

    for band in ["g", "r"]:
        #print(len(snr[(snr > 3.) & (b == band)]))
        if len(snr[(snr > 3.) & (b == band)]) < 5: # not enough good datapoints
            #print("snr too low")
            return [None,] * 6
        if (np.max(f[b == band]) - np.min(f[b == band])) < 3. * np.mean(ferr[b == band]):
            return [None,] * 6
    return t, f, ferr, b, ra, dec


def clip_lightcurve_end(times, fluxes, fluxerrs, bands):
    """
    Clip end of lightcurve with approx. 0 slope.
    Checks from back to max of lightcurve.
    """
    def line_fit(x, a, b): # pylint: disable=unused-variable
        return a*x + b

    t_clip, flux_clip, ferr_clip, b_clip = [], [], [], []
    for b in ["g", "r"]:
        idx_b = bands == b
        t_b, f_b, ferr_b = times[idx_b], fluxes[idx_b], fluxerrs[idx_b]
        if len(f_b) == 0:
            continue
        end_i = len(t_b) - np.argmax(f_b)
        num_to_cut = 0

        m_cutoff = 0.2 * np.abs((f_b[-1] - np.amax(f_b)) / (t_b[-1] - t_b[np.argmax(f_b)]))

        for i in range(2, end_i):
            cut_idx = -1*i
            m = (f_b[cut_idx] - f_b[-1]) / (t_b[cut_idx] - t_b[-1])

            if np.abs(m) < m_cutoff:
                num_to_cut = i

        if num_to_cut > 0:
            print("LC SNIPPED")
            t_clip.extend(t_b[:-num_to_cut])
            flux_clip.extend(f_b[:-num_to_cut])
            ferr_clip.extend(ferr_b[:-num_to_cut])
            b_clip.extend([b] * len(f_b[:-num_to_cut]))
        else:
            t_clip.extend(t_b)
            flux_clip.extend(f_b)
            ferr_clip.extend(ferr_b)
            b_clip.extend([b] * len(f_b))

    return np.array(t_clip), np.array(flux_clip), np.array(ferr_clip), np.array(b_clip)


def save_datafile(name, t, f, ferr, b, save_dir):
    """
    Save reformatted version of datafile to
    output folder.
    """
    arr = np.array([t, f, ferr, b])
    print(arr[:,0])
    np.savez_compressed(save_dir + str(name) + '.npz', arr)


def add_to_new_csv(name, label, redshift):
    """
    Add row to CSV of included files for
    training.
    """
    print(name, label, redshift)
    with open(OUTPUT_CSV, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([name, label, redshift])

def generate_single_flux_file(ztf_name, save_folder):
    """
    Uses ALeRCE's API to generate flux files for all ZTF
    samples in master_csv.
    """
    global alerce
    os.makedirs(save_folder, exist_ok=True)

    # Getting detections for an object
    detections = alerce.query_detections(ztf_name, format="pandas")
    print(os.path.join(save_folder, ztf_name+".csv"))
    detections.to_csv(os.path.join(save_folder, ztf_name+".csv"), index=False)
