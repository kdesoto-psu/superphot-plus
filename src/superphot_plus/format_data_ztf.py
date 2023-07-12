import csv
import glob
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from .file_paths import FITS_DIR, DATA_DIRS
from .utils import calculate_chi_squareds


def import_labels_only(input_csvs, allowed_types, fits_dir=None, redshift=False):
    """
    Import all features and labels, convert to label and features
    numpy arrays.
    """

    if fits_dir is None:
        fits_dir = FITS_DIR
    labels = []
    labels_orig = []
    repeat_ct = 0
    names = []
    redshifts = []
    sn1bc_alts = [
        "SN Ic",
        "SN Ib",
        "SN Ic-BL",
        "SN Ib-Ca-rich",
        "SN Ib/c",
        "SNIb",
        "SNIc",
        "SNIc-BL",
        "21",
        "20",
        "27",
        "26",
        "25",
    ]
    snIIn_alts = ["SNIIn", "35", "SLSN-II"]
    snIa_alts = [
        "SN Ia-91T-like",
        "SN Ia-CSM",
        "SN Ia-91bg-like",
        "SNIa",
        "SN Ia-91T",
        "SN Ia-91bg",
        "10",
        "11",
        "12",
    ]
    snII_alts = ["SN IIP", "SN IIL", "SNII", "SNIIP", "32", "30", "31"]
    slsnI_alts = [
        "40",
        "SLSN",
    ]
    tde_alts = [
        "42",
    ]

    # TODO: make more compact
    for input_csv in input_csvs:
        with open(input_csv, newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                name = row[0]
                if not os.path.isfile(fits_dir+name+"_eqwt.npz"):
                    continue
                label_orig = row[1]
                l = row[1]
                z = float(row[2])
                if redshift and z <= 0.:
                    print(name, l)
                    continue
                if l in sn1bc_alts:
                    l = "SN Ibc"
                elif l in snIIn_alts:
                    l = "SN IIn"
                elif l in snIa_alts:
                    l = "SN Ia"
                elif l in snII_alts:
                    l = "SN II"
                elif l in slsnI_alts:
                    l = "SLSN-I"
                elif l in tde_alts:
                    l = "TDE"
                if l not in allowed_types:
                    #print(l)
                    continue
                if name not in names:
                    names.append(name)
                    labels.append(l)
                    labels_orig.append(label_orig)
                    if redshift:
                        redshifts.append(z)
                else:
                    repeat_ct += 1

    tally_each_class(labels_orig)
    print(repeat_ct)
    if redshift:
        return np.array(names), np.array(labels), np.array(redshifts)
    return np.array(names), np.array(labels)


def generate_K_fold(features, classes, num_folds):
    """
    Generates set of K test sets and corresponding training sets
    """
    if num_folds == -1:
        kf = StratifiedKFold(n_splits=len(features), shuffle=True) # cross-one out validation
    else:
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True)
    return kf.split(features, classes)


def tally_each_class(labels):
    """
    Print number of samples with each class label.
    """
    tally_dict = {}
    for label in labels:
        if label not in tally_dict:
            tally_dict[label] = 1
        else:
            tally_dict[label] += 1
    for tally_label in tally_dict:
        print(tally_label,": ", str(tally_dict[tally_label]))
    print()

def get_posterior_samples(ztf_name, output_dir=None):
    """
    Get all EQUAL WEIGHT posterior samples from
    a ZTF lightcurve fit.
    """
    if output_dir is None:
        output_dir = FITS_DIR
    post_fn = os.path.join(output_dir, ztf_name + "_eqwt.npz")
    #output_dir = "../outputs/"
    #post_fn = output_dir + ztf_name +"/" + ztf_name + "post_equal_weights.dat"
    """
    with open(post_fn, "r") as post_ew:
        post_rows = post_ew.read().split("\n")
        post_arr = []
        for row in post_rows[:-1]:
            post_arr.append([float(x) for x in row.split()])
        post_arr = np.array(post_arr)[:,:-1] # exclude the loglikelihoods
    """
    npy_array = np.load(post_fn)
    post_arr = npy_array['arr_0']
    return post_arr


def oversample_using_posteriors(ztf_names, labels, chis, goal_per_class):
    """
    Draws from posteriors of a certain fit.
    """
    oversampled_labels = []
    oversampled_chis = []
    oversampled_features = []
    labels_unique = np.unique(labels)
    for l in labels_unique:
        idxs_in_class = np.asarray(labels == l).nonzero()[0]
        num_in_class = len(idxs_in_class)
        samples_per_fit = max(1, np.round(goal_per_class / num_in_class).astype(int))
        for i in idxs_in_class:
            ztf_name = ztf_names[i]
            all_posts = get_posterior_samples(ztf_name)
            sampled_idx = np.random.choice(np.arange(len(all_posts)), samples_per_fit)
            sampled_features = all_posts[sampled_idx]
            oversampled_features.extend(list(sampled_features))
            oversampled_labels.extend([l] * samples_per_fit)
            oversampled_chis.extend([chis[i]] * samples_per_fit)
    return np.array(oversampled_features), np.array(oversampled_labels), np.array(oversampled_chis)


def normalize_features(features, mean=None, std=None):
    """
    Normalize the features for feeding into the neural network.
    """
    if mean is None:
        mean = features.mean(axis=-2)
    if std is None:
        std = features.std(axis=-2)

    print(mean, std)
    return (features - mean) / std, mean, std
