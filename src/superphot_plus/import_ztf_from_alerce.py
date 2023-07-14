"""This script provides functions for importing and manipulating ZTF 
data from the Alerce API."""

import csv
import glob
import os
import zipfile

import extinction
import numpy as np
from alerce.core import Alerce
from antares_client.search import get_by_ztf_object_id
from astropy.coordinates import SkyCoord
from dustmaps.config import config
from dustmaps.sfd import SFDQuery

from superphot_plus.sfd import dust_filepath

from .utils import convert_mags_to_flux

alerce = Alerce()
MIN_PER_FILTER = 5

def add_stamp_column(input_filename, output_filename):
    """Checks whether stamp classifier categorizes each lightcurve in
    spreadsheet as a supernova-like transient, and adds as additional
    column.

    Parameters
    ----------
    input_filename : str
        Path to the input CSV file.
    output_filename : str
        Path to the output CSV file.
    """
    csv_rows = []
    with open(input_filename, "r") as fn_csv:
        csv_reader = csv.reader(fn_csv, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            csv_rows.append(row)

    print("done reading in rows")
    with open(output_filename, "w+") as new_csv:
        csv_writer = csv.writer(new_csv, delimiter=",")
        csv_writer.writerow(["NAME", "PROB", "CLASS", "STAMP"])

    with open(output_filename, "a+") as new_csv:
        csv_writer = csv.writer(new_csv, delimiter=",")
        for row in csv_rows:
            try:
                name = row[0]
                print(name)

                p = alerce.query_probabilities(oid=name, format="pandas")

                p_class = p[p["classifier_name"] == "stamp_classifier"]
                prob = p_class[p_class["ranking"] == 1]["probability"].iat[0]
                best_label = p_class[p_class["ranking"] == 1]["class_name"].iat[0]

                stamp = ((best_label == "SN") and (prob >= 0.5))
                csv_writer.writerow([*row, stamp])

            except:
                csv_writer.writerow([*row, "None"])


def get_spreadsheet_diff(s1, s2, new_fn):
    """Determines which elements are in s2 but not s1, and vice versa.

    Saves overlap as separate CSV.

    Parameters
    ----------
    s1 : str
        Path to the first input CSV file.
    s2 : str
        Path to the second input CSV file.
    new_fn : str
        Path to the output CSV file.
    """
    rows_reduced = []
    names = set()
    with open(s1, "r") as s1_csv:
        csvreader = csv.reader(s1_csv)
        for row in csvreader:
            name = row[0]
            if name in names:
                print("REPEAT IN CSV", name)
            else:
                #rows_reduced.append(row)
                names.add(name)

    name_diff = set()
    with open(s2, "r") as s2_csv:
        csvreader = csv.reader(s2_csv)
        for row in csvreader:
            name2 = row[0]
            if (name2 not in names) and (name2 != "Name"):
                rows_reduced.append(row)
                name_diff.add(name2)

    with open(new_fn, "w") as nf:
        csvwriter = csv.writer(nf)
        #csvwriter.writerow(["NAME", "RA", "DEC", "CLASS", "Z"])
        csvwriter.writerow(["NAME", "PROB", "CLASS"])

    with open(new_fn, "a+") as nf:
        csvwriter = csv.writer(nf)
        for row_reduced in rows_reduced:
            csvwriter.writerow(row_reduced)

    print(name_diff)


def filter_tns_all_transients(tns_csv, reduced_fn):
    """Filter TNS CSV by type and ZTF name.
    
    With 0 indexing:
    row 3 = ra
    row 4 = dec
    row 5 = redshift
    row 7 = obj type
    row 18 = name.

    Parameters
    ----------
    tns_csv : str
        Path to the TNS CSV file.
    reduced_fn : str
        Path to the output filtered CSV file.
    """

    ras = []
    decs = []
    zs = []
    ztf_names = []
    obj_types = []
    ct = 0

    with open(tns_csv, "r") as tc:
        csvreader = csv.reader(tc)
        next(csvreader)
        for row in csvreader:
            ct += 1

            """
            obj_type = row[7]
            redshift = row[5]
            ra = row[3]
            dec = row[4]
            names = row[18].split(",")
            """

            obj_type = row[11]
            redshift = row[12]
            ra = row[2]
            dec = row[3]
            names = row[0].split(",")

            if obj_type == "":
                continue

            ztf_name = None
            for name in names:
                name = name.strip("'").strip()
                if len(names) > 2:
                    print(name)
                if name[:3] == "ZTF":
                    ztf_name = name
                    break

            if ztf_name is None:
                continue

            if ra == "":
                continue
            if dec == "":
                continue
            if redshift == "":
                redshift = "-1"

            z = float(redshift)
            #ra = float(ra)
            #dec = float(dec)

            ras.append(ra)
            decs.append(dec)
            zs.append(z)
            ztf_names.append(ztf_name)
            obj_types.append(obj_type)

    with open(reduced_fn, "w") as rf:
        csvwriter = csv.writer(rf)
        csvwriter.writerow(["NAME", "RA", "DEC", "CLASS", "Z"])

    with open(reduced_fn, "a+") as rf:
        csvwriter = csv.writer(rf)
        for i in range(len(ras)):
            csvwriter.writerow([ztf_names[i], ras[i], decs[i], obj_types[i], zs[i]])


def extract_all_zip_files(zip_folder):
    """Extract all ZIP files in the specified folder.

    Parameters
    ----------
    zip_folder : str
        Path to the folder containing the ZIP files.
    """
    all_zip_files = glob.glob(zip_folder+"*.zip")
    for zf in all_zip_files:
        name = zf.split("_")[-2].split("/")[-1]
        with zipfile.ZipFile(zf, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
            #os.remove(EXTRACT_DIR+"non_detections.csv")
            os.rename(EXTRACT_DIR+"non_detections.csv", EXTRACT_DIR+name+"_nd.csv")
            os.rename(EXTRACT_DIR+"detections.csv", EXTRACT_DIR+name+"_d.csv")


def get_all_unclassified_samples(save_csv):
    """Get all unclassified samples and save them to a CSV file.

    Parameters
    ----------
    save_csv : str
        Path to the output CSV file.
    """
    global alerce

    classifiers = alerce.query_classifiers()
    print(classifiers)
    i = 40
    repeat_names = set()
    """
    with open(save_csv, "w+") as sc:
        sc.write("")
        csv_writer = csv.writer(sc, delimiter=",")
        csv_writer.writerow(["Name","Probability","Label"])
    """

    with open(save_csv, "r") as sc:
        csv_reader = csv.reader(sc, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            repeat_names.add(row[0])

    while True:
        print(i)

        while True:
            try:

                objs = alerce.query_objects(classifier="stamp_classifier",
                                            classifier_version="stamp_classifier_1.0.4",
                                            class_name="SN",
                                            format="pandas",
                                            page_size=500,
                                            probability=0.5,
                                            page=i)
                """
                objs = alerce.query_objects(classifier="lc_classifier_top",
                                            class_name="Transient",
                                            probability=0.5,
                                            format="pandas",
                                            page_size=100,
                                            page=i)
                """
                break
            except:
                pass

        if len(objs) == 0: # finished
            return None

        with open(save_csv, "a+") as sc:
            csv_writer = csv.writer(sc, delimiter=",")

            for row_idx in range(len(objs)):
                #if row_idx % 100 == 0:
                try:
                    row = objs.iloc[row_idx]
                    name = row.iat[0]
                    #if os.path.exists(OUTPUT_FOLDER+name+".npz"):
                        #print("true class known")
                    #    continue
                    if name in repeat_names:
                        print("REPEAT")
                        continue
                    p = alerce.query_probabilities(oid=name, format="pandas")

                    p_class = p[p["classifier_name"] == "stamp_classifier"]
                    prob = p_class[p_class["ranking"] == 1]["probability"].iat[0]
                    best_label = p_class[p_class["ranking"] == 1]["class_name"].iat[0]
                    if best_label != "SN":
                        print("wrong type")
                        continue
                    """
                    p_top = p[p["classifier_name"] == "lc_classifier_top"]

                    prob = p_top[p_top["ranking"] == 1]["probability"].iat[0]
                    p_transient = p[p["classifier_name"] == "lc_classifier_transient"]
                    best_label = p_transient[p_transient["ranking"] == 1]["class_name"].iat[0]

                    csv_writer.writerow([name, prob, best_label])
                    """
                    csv_writer.writerow([name, prob, best_label])
                    repeat_names.add(name)

                except:
                    print("skipped")
                    continue
        i += 1


def get_band_extinctions(ra, dec):
    """Gets g- and r-band extinctions in magnitudes for a single
    supernova lightcurve based on right ascension and declination
    coordinates.
    
    Parameters
    ----------
    ra : float
        Right ascension coordinate.
    dec : float
        Declination coordinate.

    Returns
    -------
    list (of float?) # TODOLIV
        List containing the g- and r-band extinctions.
    """
    config["data_dir"] = dust_filepath
    sfd = SFDQuery()
    #First look up the amount of mw dust at this location
    coords = SkyCoord(ra,dec, frame='icrs', unit='deg')
    Av_sfd = 2.742 * sfd(coords) # from https://dustmaps.readthedocs.io/en/latest/examples.html

    # for gr, the was are:
    band_wvs = 1./ (0.0001 * np.asarray([4741.64, 6173.23])) # in inverse microns

    #Now figure out how much the magnitude is affected by this dust
    ext_list = extinction.fm07(band_wvs, Av_sfd, unit='invum') # in magnitudes

    return ext_list


def import_lc(filename):
    """Imports a single file, but only the points from a single
    telescope, in only g and r bands.

    Parameters
    ----------
    filename : str
        Path to the input CSV file.

    Returns
    -------
    tuple
        Tuple containing the imported light curve data.
    """
    ra = None
    dec = None
    if not os.path.exists(filename):
        return [None,] * 6
    with open(filename, 'r') as csv_f:
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
    """Clips end of lightcurve with approximately 0 slope. Checks from
    back to max of lightcurve.

    Parameters
    ----------
    times : np.ndarray
        Time values of the light curve.
    fluxes : np.ndarray
        Flux values of the light curve.
    fluxerrs : np.ndarray
        Flux error values of the light curve.
    bands : np.ndarray
        Band information of the light curve.

    Returns
    -------
    tuple
        Tuple containing the clipped light curve data.
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


def save_new_datafiles(): # TODOLIV missed one
    """Save new data files based on the provided CSV file."""
    with open(OUTPUT_CSV, "w+") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["Name", "Label", "Redshift"])

    label_dict = {}
    with open(CSV_FILE, 'r') as csv_f:
        csvreader = csv.reader(csv_f, delimiter=",", skipinitialspace=True)
        next(csvreader)
        for row in csvreader:

            if len(row) == 0:
                continue
            #name = row[1].strip().split()[-1]
            #label = row[4].strip()
            #ztf_name = row[12]
            name = row[0]
            label = row[2]
            #label = row[11]
            #label = row[3]
            if row[3].strip() != "True":
                continue
            try:
                #redshift = float(row[12].strip())
                #redshift = float(row[4].strip())
                redshift = -1 # pylint: disable=unused-variable
            except:
                redshift = -1
            print(name)
            data_fn = DATA_FOLDER + name + ".csv"
            t, f, ferr, b, ra, dec = import_lc(data_fn) # pylint: disable=unused-variable
            print(t)
            if t is not None:
                save_datafile(name, t, f, ferr, b, OUTPUT_FOLDER)
                add_to_new_csv(name, label, float(row[1].strip()))
                if label in label_dict:
                    label_dict[label] += 1
                else:
                    label_dict[label] = 1

    for l in label_dict:
        print(l, label_dict[l])


def save_datafile(name, times, fluxes, fluxerrs, bands, save_dir):
    """Saves a reformatted version of data file to the output folder.

    Parameters
    ----------
    name : str
        Name of the data file.
    times : np.ndarray
        Time values of the light curve.
    fluxes : np.ndarray
        Flux values of the light curve.
    fluxerrs : np.ndarray
        Flux error values of the light curve.
    bands : np.ndarray
        Band information of the light curve.
    save_dir : str
        Path to the output folder.
    """
    arr = np.array([times, fluxes, fluxerrs, bands])
    print(arr[:,0])
    np.savez_compressed(save_dir + str(name) + '.npz', arr)


def add_to_new_csv(name, label, redshift):
    """Add row to CSV of included files for training.
    
    Parameters
    ----------
    name : str
        Name in the new row.
    label : str
        Label in the new row.
    redshift : float
        Redshift value  in the new row.
    """
    print(name, label, redshift)
    with open(OUTPUT_CSV, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([name, label, redshift])


def make_master_csv(CSV_DIR, master_csv_name):
    """Checks through entire TNS SN-classified database and tallies
    count of each category.

    Parameters
    ----------
    CSV_DIR : str
        Path to the directory containing the CSV files.
    master_csv_name : str
        Name of the output master CSV file.
    """
    cat_dict = {}
    csv_arr = []
    for csv_file in glob.glob(CSV_DIR+"/*.csv"):
        print(csv_file)
        with open(csv_file, "r") as csv_f:
            csvreader = csv.reader(csv_f, delimiter=",", skipinitialspace=True)
            for row in csvreader:
                # get ZTF name, check if starts with ZTF
                ztf_name = row[12]
                if ztf_name[:3] != "ZTF":
                    continue
                # else, get SN type
                sn_type = row[4]
                # add tally to dictionary
                if sn_type not in cat_dict:
                    cat_dict[sn_type] = 1
                else:
                    cat_dict[sn_type] += 1

                ra = row[2]
                dec = row[3]
                z = row[5]

                csv_arr.append([ztf_name, ra, dec, sn_type, z])

    print(cat_dict)
    csv_arr = np.array(csv_arr)
    with open(master_csv_name, "w+") as mc:
        csvwriter = csv.writer(mc, delimiter=",", skipinitialspace=True)
        csvwriter.writerows(csv_arr)


def generate_flux_files(master_csv, save_folder):
    """Generates flux files for all ZTF samples in the master CSV file,
    using ALeRCE's API.

    Parameters
    ----------
    master_csv : str
        Path to the master CSV file.
    save_folder : str
        Path to the folder where the flux files will be saved.
    """
    global alerce
    os.makedirs(save_folder, exist_ok=True)
    with open(master_csv, "r") as mc:
        csvreader = csv.reader(mc, delimiter=",", skipinitialspace=True)
        for row in csvreader:
            try:
                ztf_name = row[0]
                if os.path.exists(os.path.join(save_folder, ztf_name+".csv")):
                    continue
                print(ztf_name)
                # Getting detections for an object
                detections = alerce.query_detections(ztf_name, format="pandas")
                detections.to_csv(os.path.join(save_folder, ztf_name+".csv"), index=False)
            except:
                continue


def generate_files_from_antares():
    """Generates flux files for all ZTF samples in the master CSV file,
    using ANTARES' API.
     
    Includes correct zeropoints.
    """
    with open(OUTPUT_CSV, "w+") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["Name", "Label", "Redshift"])

    label_dict = {}
    #os.makedirs(save_folder, exist_ok=True)
    with open(CSV_FILE, "r") as mc:
        csvreader = csv.reader(mc, delimiter=",", skipinitialspace=True)
        next(csvreader)
        for row in csvreader:
            try:
                ztf_name = row[0]
                if os.path.exists(OUTPUT_FOLDER + str(ztf_name) + '.npz'):
                    continue
                print(ztf_name)
                # Getting detections for an object
                locus = get_by_ztf_object_id(ztf_name)
                ts = locus.timeseries[
                    [
                        "ant_mjd",
                        "ztf_magpsf",
                        "ztf_sigmapsf",
                        "ztf_fid",
                        "ant_ra",
                        "ant_dec",
                        "ztf_magzpsci",
                    ]
                ]

            except:
                continue

            label = row[3]
            print(label)
            try:
                #redshift = float(row[12].strip())
                redshift = float(row[4].strip())
            except:
                redshift = -1

            t, m, merr, b_int, ra, dec, zp = ts.to_pandas().to_numpy().T
            b = np.where(b_int.astype(int) == 1, "g", "r")
            try:
                ra = np.mean(ra[~np.isnan(ra)])
                dec = np.mean(dec[~np.isnan(dec)])
                g_ext, r_ext = get_band_extinctions(ra, dec)
            except:
                continue
            m[b == "r"] -= r_ext
            m[b == "g"] -= g_ext

            valid_idx = ~np.isnan(merr) & ~np.isnan(zp)
            t = t[valid_idx]
            m = m[valid_idx]
            b = b[valid_idx]
            zp = zp[valid_idx]
            merr = merr[valid_idx]

            f, ferr = convert_mags_to_flux(m, merr, zp)
            #print(f, ferr)
            t, f, ferr, b = clip_lightcurve_end(t, f, ferr, b)
            #print(f, ferr)
            snr = np.abs(f / ferr)

            if len(snr[(snr > 3.) & (b == "g")]) < 5: # not enough good datapoints
                print("snr too low")
                continue
            if (np.max(f[b == "g"]) - np.min(f[b == "g"])) < 3. * np.mean(ferr[b == "g"]):
                continue

            if len(snr[(snr > 3.) & (b == "r")]) < 5: # not enough good datapoints
                print("snr too low")
                continue
            if (np.max(f[b == "r"]) - np.min(f[b == "r"])) < 3. * np.mean(ferr[b == "r"]):
                continue

            save_datafile(ztf_name, t, f, ferr, b)
            add_to_new_csv(ztf_name, label, redshift)
            if label in label_dict:
                label_dict[label] += 1
            else:
                label_dict[label] = 1

    for l in label_dict:
        print(l, label_dict[l])


def generate_single_flux_file(ztf_name, save_folder):
    """Generates a flux file for a single ZTF sample in the master CSV
    file, using ALeRCE's API.
    
    Parameters
    ----------
    ztf_name : str
        Name of the ZTF sample.
    save_folder : str
        Path to the folder where the flux file will be saved.
    """
    global alerce
    os.makedirs(save_folder, exist_ok=True)

    # Getting detections for an object
    detections = alerce.query_detections(ztf_name, format="pandas")
    print(os.path.join(save_folder, ztf_name+".csv"))
    detections.to_csv(os.path.join(save_folder, ztf_name+".csv"), index=False)
