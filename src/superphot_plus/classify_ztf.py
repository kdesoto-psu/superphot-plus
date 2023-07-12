import csv
import os
import shutil

import numpy as np
from joblib import Parallel, delayed
import torch

from .constants import NUM_FOLDS, MEANS_TRAINED_MODEL, STDDEVS_TRAINED_MODEL
from .file_paths import * # pylint: disable=wildcard-import
from .format_data_ztf import (
    import_labels_only,
    tally_each_class,
    train_test_split,
    oversample_using_posteriors,
    get_posterior_samples,
    normalize_features,
    generate_K_fold,
)
from .mlp import (
    run_mlp,
    create_dataset,
    MLP,
    save_test_probabilities
)
from .plotting import plot_confusion_matrix
from .utils import calculate_chi_squareds, f1_score, calc_accuracy
from .ztf_transient_fit import import_data, run_mcmc


def adjust_log_dists(features):
    """
    Takes log of fit parameters with log-Gaussian priors before
    feeding into classifier. Also removes apparent amplitude and t0.
    
    Parameters
    ----------
    features : numpy array
        Array of fit features of all samples.
    """
    features[:, 4:7] = np.log10(features[:, 4:7])
    features[:, 2] = np.log10(features[:, 2])
    return np.delete(features, [0,3], 1)


def classify(goal_per_class, num_epochs, neurons_per_layer, num_layers, fits_plotted=False):
    """
    Train MLP to classify between supernovae of 'allowed_types'.
    
    Parameters
    ----------
    goal_per_class : int
        Oversampling such that there are this many fits per supernova type.
    num_epochs : int
        Number of training epochs.
    neurons_per_layer : int
        Number of neurons per hidden layer of MLP.
    num_layers : int
        Number of hidden layers in MLP.
    fits_plotted : bool
        If true, assumes all sample fit plots are saved in FIT_PLOTS_FOLDER. Copies
        plots of wrongly classified samples to separate folder for manual followup.
        Defaults to False.
    """

    #for file in os.scandir('models'):
    #    os.remove(file.path)
    allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc", "SLSN-II"]
    output_dim = len(allowed_types) # number of classes

    labels_to_classes = {allowed_types[i]: i for i in range(len(allowed_types))}
    classes_to_labels = {i: allowed_types[i] for i in range(len(allowed_types))}

    fn_prefix = "cm_%d_%d_%d_%d" % (goal_per_class, num_epochs, neurons_per_layer, num_layers)
    fn_purity = os.path.join(CM_FOLDER, fn_prefix + "_p.pdf")
    fn_completeness = os.path.join(CM_FOLDER, fn_prefix + "_c.pdf")
    fn_purity_07 = os.path.join(CM_FOLDER, fn_prefix + "_p_p07.pdf")
    fn_completeness_07 = os.path.join(CM_FOLDER, fn_prefix + "_c_p07.pdf")

    names, labels = import_labels_only(input_csvs, allowed_types)

    tally_each_class(labels) # original tallies

    kfold = generate_K_fold(np.zeros(len(labels)), labels, NUM_FOLDS)

    true_classes_mlp = np.array([])
    predicted_classes_mlp = np.array([])
    prob_above_07_mlp = np.array([], dtype=bool)

    def run_single_fold(x):
        train_index, test_index = x
        train_labels = labels[train_index]
        test_labels = labels[test_index]
        test_names = np.array(names[test_index])

        train_index, val_index = train_test_split(train_index, stratify=train_labels, test_size=0.1)

        train_names = names[train_index]
        val_names = names[val_index]

        train_labels = labels[train_index]
        val_labels = labels[val_index]

        train_classes = np.array([labels_to_classes[l] for l in train_labels]).astype(int)
        val_classes = np.array([labels_to_classes[l] for l in val_labels]).astype(int)
        test_classes = np.array([labels_to_classes[l] for l in test_labels]).astype(int)

        train_chis = calculate_chi_squareds(train_names, FITS_DIR, DATA_DIRS)
        train_features, train_classes, train_chis = oversample_using_posteriors(
            train_names, train_classes, train_chis, goal_per_class
        )
        val_chis = calculate_chi_squareds(val_names, FITS_DIR, DATA_DIRS)
        val_features, val_classes, val_chis = oversample_using_posteriors(
            val_names, val_classes, val_chis, round(0.1 * goal_per_class)
        )

        train_features = np.append(train_features, np.array([train_chis,]).T, 1)
        val_features = np.append(val_features, np.array([val_chis,]).T, 1)

        test_features = []
        test_classes_os = []
        test_group_idxs = []
        test_names_os = []
        test_chis_os = []
        test_chis = calculate_chi_squareds(test_names, FITS_DIR, DATA_DIRS)

        for i in range(len(test_names)):
            test_name = test_names[i]
            test_posts = get_posterior_samples(test_name)
            test_features.extend(test_posts)
            test_classes_os.extend([test_classes[i]] * len(test_posts))
            test_names_os.extend([test_names[i]] * len(test_posts))
            test_chis_os.extend([test_chis[i]] * len(test_posts))
            if len(test_group_idxs) == 0:
                start_idx = 0
            else:
                start_idx = test_group_idxs[-1][-1]+1
            test_group_idxs.append(np.arange(start_idx, start_idx+len(test_posts)))

        test_features = np.array(test_features)
        test_chis = np.array([test_chis_os,])

        test_features = np.append(test_features, test_chis.T, 1)

        # normalize the log distributions
        test_features = adjust_log_dists(test_features)
        test_classes = np.array(test_classes_os)
        test_names = np.array(test_names_os)

        #print(test_names[0])
        train_features = adjust_log_dists(train_features)
        val_features = adjust_log_dists(val_features)
        train_features, mean, std = normalize_features(train_features)
        val_features, mean, std = normalize_features(val_features, mean, std)
        test_features, mean, std = normalize_features(test_features, mean, std)

        #Convert to Torch DataSet objects
        train_data = create_dataset(train_features, train_classes)
        val_data = create_dataset(val_features, val_classes)
        #test_data = create_dataset(test_features, test_classes)

        # Train and evaluate multi-layer perceptron
        test_classes, test_names, pred_classes, pred_probs, valid_loss = run_mlp(
            train_data,
            val_data,
            test_features,
            test_classes,
            test_names,
            test_group_idxs,
            input_dim,
            output_dim,
            neurons_per_layer,
            num_layers,
            num_epochs,
        )

        return pred_classes, pred_probs > 0.7, test_classes, test_names, valid_loss

    r = Parallel(n_jobs=-1)(delayed(run_single_fold)(x) for x in kfold)
    (
        predicted_classes_mlp,
        prob_above_07_mlp,
        true_classes_mlp,
        ztf_test_names,
        valid_loss_mlp,
    ) = zip(*r)
    predicted_classes_mlp = np.hstack(tuple(predicted_classes_mlp))
    prob_above_07_mlp = np.hstack(tuple(prob_above_07_mlp))
    true_classes_mlp = np.hstack(tuple(true_classes_mlp))
    ztf_test_names = np.hstack(tuple(ztf_test_names))
    valid_loss_avg = np.mean(valid_loss_mlp)

    true_classes_mlp = np.array([classes_to_labels[l] for l in true_classes_mlp])
    predicted_classes_mlp = np.array([classes_to_labels[l] for l in predicted_classes_mlp])

    if fits_plotted:
        wrongly_classified = np.where(true_classes_mlp != predicted_classes_mlp)[0]
        for wc_idx in wrongly_classified:
            wc = ztf_test_names[wc_idx]
            wc_type = true_classes_mlp[wc_idx]
            wrong_type = predicted_classes_mlp[wc_idx]
            fn = wc+".png"
            fn_new = wc+"_"+wc_type+"_"+wrong_type+".png"
            shutil.copy(
                os.path.join(FIT_PLOTS_FOLDER, fn),
                os.path.join(WRONGLY_CLASSIFIED_FOLDER, wc_type + "/" + fn_new),
            )

    with open(CLASSIFY_LOG_FILE, "a+") as the_file:
        the_file.write(str(goal_per_class) + " samples per class\n")
        the_file.write(
            str(neurons_per_layer)
            + " neurons per each of "
            + str(num_layers)
            + " layers\n"
        )
        the_file.write(str(num_epochs) + " epochs\n")
        the_file.write(
            "HOW MANY CERTAIN "
            + str(len(true_classes_mlp))
            + " "
            + str(len(true_classes_mlp[prob_above_07_mlp]))
            + "\n"
        )
        the_file.write(
            "MLP class-averaged F1-score: %.04f\n"
            % f1_score(predicted_classes_mlp, true_classes_mlp, class_average=True)
        )
        the_file.write(
            "Accuracy: %.04f\n" % calc_accuracy(predicted_classes_mlp, true_classes_mlp)
        )
        the_file.write("Validation Loss: %.04f\n\n" % valid_loss_avg)


    # Plot full and p > 0.7 confusion matrices
    plot_confusion_matrix(true_classes_mlp, predicted_classes_mlp, fn_purity, True)
    plot_confusion_matrix(true_classes_mlp, predicted_classes_mlp, fn_completeness, False)

    plot_confusion_matrix(
        true_classes_mlp[prob_above_07_mlp],
        predicted_classes_mlp[prob_above_07_mlp],
        fn_purity_07,
        True,
    )
    plot_confusion_matrix(
        true_classes_mlp[prob_above_07_mlp],
        predicted_classes_mlp[prob_above_07_mlp],
        fn_completeness_07,
        False,
    )
