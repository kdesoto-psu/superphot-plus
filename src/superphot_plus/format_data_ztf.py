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


def import_features_and_labels(input_csv, allowed_types):
    """
    Import all features and labels, convert to label and features
    numpy arrays.
    """
    feature_means = []
    feature_stddevs = []
    labels = []
    names = []
    sn1bc_alts = ["SN Ic", "SN Ib", "SN Ic-BL", "SN Ib-Ca-rich", "SN Ib/c"]
    snIIn_alts = ["SLSN-II"]
    snIa_alts = ["SN Ia-91 T-like", "SN Ia-CSM", "SN Ia-91bg-like"]
    snII_alts = ["SN IIP", "SN IIL"]
    with open(input_csv, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            l = row[1]
            if l in sn1bc_alts:
                l = "SN Ibc"
            elif l in snIIn_alts:
                l = "SN IIn"
            elif l in snIa_alts:
                l = "SN Ia"
            elif l in snII_alts:
                l = "SN II"
            if l not in allowed_types:
                continue
            names.append(row[0])
            feature_means.append(row[2:16])
            feature_stddevs.append(row[16:])
            labels.append(l)

    return (
        np.array(names),
        np.array(feature_x).astype(float),
        np.array(feature_stddevs).astype(float),
        np.array(labels),
    )


def return_names_from_med_arrays(input_csv, med_arr):

    names = [""] * len(med_arr) # pylint: disable=unused-variable

    t_0_expected = med_arr[3]
    best_diff = np.inf
    best_features = None
    best_match = None
    ct = 0
    for fn in glob.glob(FITS_DIR+"/*.npz"):
        try:
            name = fn.split("/")[-1].split("_")[0]
            #print(name)
            features = get_posterior_samples(name, output_dir=None)
            med_features = np.median(features, axis=0)
            t_0 = med_features[3]
            diff = np.abs(t_0_expected - t_0)
            if diff < best_diff:
                best_diff = diff
                best_match = name
                best_features = med_features
            ct += 1
        except:
            pass
    print(ct)
    print(best_match,best_features)


def divide_into_training_test_set(features, labels, test_fraction):
    """
    Divides dataset into set fraction of test samples and remaining as
    training data.
    """
    return train_test_split(features, labels, test_size=test_fraction, random_state=42)


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


def generate_two_class_labels(labels):
    """
    For the binary classification problem.
    """
    labels_copy = np.copy(labels)
    labels_copy[labels_copy != "SN Ia"] = "other"
    return labels_copy


def oversample_minority_classes(features, labels):
    """
    Uses SMOTE to oversample data from rarer classes so
    classifiers are not biased toward SN-1a or SN-II
    """
    oversample = SMOTE()
    features_smote, labels_smote = oversample.fit_resample(features, labels)
    return features_smote, labels_smote


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


# TODO: find and remove obsolete functions
def summarize_misc_classification(misc_csv):
    """
    Summarize how miscellaneous types of transients are classified.
    """
    misc_dict = {}
    with open(misc_csv, "r") as mc:
        csv_reader = csv.reader(mc, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            obj_type = row[1]
            probs = np.array(row[2:]).astype(float)
            best_type = np.argmax(probs)
            if obj_type not in misc_dict:
                misc_dict[obj_type] = [0,0,0,0,0]
            misc_dict[obj_type][best_type] += 1

    print(misc_dict)


def generate_csv_subset(orig_sn_name, new_sn_name, sn_idx, p_cutoff):
    """
    Generate smaller subset with only SNe of one type,
    with confidence above certain threshhold.
    """
    sn_names = []
    with open(orig_sn_name, "r") as orig:
        csv_reader = csv.reader(orig, delimiter=",")
        for row in csv_reader:
            if float(row[sn_idx]) > p_cutoff:
                sn_names.append(row[0])

    with open(new_sn_name, "a+") as new:
        csv_writer = csv.writer(new, delimiter=",")
        for sn in sn_names:
            csv_writer.writerow([sn, -1])


def generate_csv_subset2(orig_sn_names, new_sn_name, sn_type):
    """
    Generate smaller subset with only SNe of one type,
    from spectroscopic set.
    """
    with open(new_sn_name, "w+") as new:
        new.write("")

    alts = {"SN II": ["SN II", "SNII", "SN IIP", "SN IIL", "SNII", "SNIIP", "32", "30", "31"]}
    sn_names = []
    zs = []
    for orig_sn_name in orig_sn_names:
        with open(orig_sn_name, "r") as orig:
            csv_reader = csv.reader(orig, delimiter=",")
            for row in csv_reader:
                if row[1] in alts["SN II"]:
                    if row[0] in sn_names:
                        continue
                    if not os.path.exists(FITS_DIR + row[0] + "_eqwt.npz"):
                        continue
                    sn_names.append(row[0])
                    zs.append(float(row[2]))


    train_chis = calculate_chi_squareds(sn_names, FITS_DIR, DATA_DIRS)
    print(len(train_chis), len(sn_names))
    for _, sn_name in enumerate(sn_names):
        train_features, train_classes, train_chis_os = oversample_using_posteriors( # pylint: disable=unused-variable
            [sn_name,], [2,], [train_chis[e],], 100
        )
        med_features = np.median(train_features, axis=0)
        med_features = np.append(med_features, np.median(train_chis_os))

        with open(new_sn_name, "a") as new:
            csv_writer = csv.writer(new, delimiter=",")
            #csv_writer.writerow(["Name","Redshift"])
            csv_writer.writerow([sn_name, zs[e], *med_features])


def add_snr_to_prob_csv(probs_csv, new_csv):
    """
    Adds 10% SNR and num of SNR > 5 points columns
    to probability CSV. Useful for plots.
    """
    all_rows = []
    with open(probs_csv, "r") as csvfile:
        with open(new_csv, 'w+') as csvoutput:
            csvreader = csv.reader(csvfile)
            csvwriter = csv.writer(csvoutput)
            for row in csvreader:
                name = row[0]
                for data_dir in DATA_DIRS:
                    try:
                        #data_fn = glob.glob(data_dir + "/*/" + name + ".npz")[0]
                        data_fn = data_dir + "/" + name + ".npz"
                        npy_array = np.load(data_fn)
                        #print(npy_array)
                    except:
                        pass

                arr = npy_array['arr_0']

                ferr = arr[2]
                f = arr[1][ferr != "nan"].astype(float)
                b = arr[3][ferr != "nan"]
                ferr = ferr[ferr != "nan"].astype(float)
                snr = np.abs(f / ferr)

                n_snr_3 = len(snr[(snr > 3.)])
                n_snr_5 = len(snr[(snr > 5.)])
                n_snr_10 = len(snr[(snr > 10.)])
                snr_ten_percent = np.quantile(snr, 0.9)
                max_r_flux = np.max(f[b == "r"])
                row.append(max_r_flux)
                row.append(snr_ten_percent)
                row.append(n_snr_3)
                row.append(n_snr_5)
                row.append(n_snr_10)
                all_rows.append(row)
            csvwriter.writerows(all_rows)
