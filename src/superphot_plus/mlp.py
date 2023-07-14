"""This module implements the Multi-Layer Perceptron (MLP) model for
classification."""

import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import TensorDataset

from .constants import * # star import used due to large quantity of
items imported
from .file_paths import PROBS_FILE, PROBS_FILE2, MODEL_DIR, METRICS_DIR


def save_test_probabilities(output_filename, true_label, pred_probabilities):
    """Saves probabilities to a separate file for ROC curve generation.

    Parameters
    ----------
    output_filename : str
        The file name to save to.
    true_label : str or int
        The true label.
    pred_probabilities : array-like of float # TODOLIV look into array type
        The prediction probabilities.
    """
    with open(PROBS_FILE, "a+") as pf:
        pf.write("%s,%s" % (output_filename, str(true_label)))
        for p in pred_probabilities:
            pf.write(",%.04f" % p)
        pf.write("\n")


def save_unclassified_test_probabilities(output_filename, pred_probabilities):
    """Saves probabilities to a separate file for ROC curve generation.

    Parameters
    ----------
    output_filename : str
        The file name to save to.
    pred_probabilities : array-like # TODOLIV array type
        The prediction probabilities.
    """
    with open(PROBS_FILE2, "a+") as pf:
        pf.write("%s" % output_filename)
        for p in pred_probabilities:
            pf.write(",%.04f" % p)
        pf.write("\n")


def get_predictions(model, iterator, device):
    """Given a trained model, returns the test images, test labels, and 
    prediction probabilities across all the test labels.

    Parameters
    ----------
    model : mlp.MLP
        The trained model.
    iterator : torch.utils.data.DataLoader
        The data iterator.
    device : torch.device or str
        The device to use.

    Returns
    -------
    tuple
        A tuple containing the test images, test labels, sample indices,
        and prediction probabilities.
    """
    model.eval()

    images = []
    labels = []
    probs = []
    sample_idxs = []

    with torch.no_grad():

        for (x, y, z) in iterator:

            x = x.to(device)

            y_pred, _ = model(x)

            y_prob = F.softmax(y_pred, dim=-1)

            images.append(x.cpu())
            labels.append(y.cpu())
            sample_idxs.append(z.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)
    sample_idxs = torch.cat(sample_idxs, dim=0)

    return images, labels, sample_idxs, probs


def get_predictions_new(model, iterator, device):
    """Given a trained model, returns the test images, test labels, and
    prediction probabilities across all the test labels.

    Parameters
    ----------
    model : mlp.MLP
        The trained model.
    iterator : torch.utils.data.DataLoader
        The data iterator.
    device : torch.device or str
        The device to use.

    Returns
    -------
    tuple
        A tuple containing the test images and prediction probabilities.
    """
    model.eval()

    images = []
    probs = []

    with torch.no_grad():

        for x in iterator:

            x = x[0].to(device)

            y_pred, _ = model(x)

            y_prob = F.softmax(y_pred, dim=-1)

            images.append(x.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, probs


def calculate_accuracy(y_pred, y):
    """Calculate the prediction accuracy.

    Parameters
    ----------
    y_pred : torch.Tensor
        The predicted tensor.
    y : torch.Tensor
        The true tensor.

    Returns
    -------
    torch.Tensor
        The calculated accuracy.
    """
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


class MLP(nn.Module):
    """The Multi-Layer Perceptron. Sets the number of layers and nodes
    per layer.

    Parameters
    ----------
    input_dim : int
        The input dimension.
    output_dim : int
        The output dimension.
    neurons_per_layer : int
        The number of neurons per layer.
    num_hidden_layers : int
        The number of hidden layers. Must be >= 1.
    """
    def __init__(self, input_dim, output_dim, neurons_per_layer, num_hidden_layers):
        super().__init__()

        n_neurons = neurons_per_layer
        self.input_fc = nn.Linear(input_dim, n_neurons)

        assert num_hidden_layers >= 1

        self.hidden_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.dropouts.append(nn.Dropout(INPUT_DROPOUT_FRAC))

        for i in range(num_hidden_layers - 1): # pylint: disable=unused-variable
            self.hidden_layers.append(nn.Linear(n_neurons, n_neurons))
        for i in range(num_hidden_layers):
            self.dropouts.append(nn.Dropout(HIDDEN_DROPOUT_FRAC))

        self.output_fc = nn.Linear(n_neurons, output_dim)

    def forward(self, x):
        """Forward pass of the Multi-Layer Perceptron model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        tuple
            A tuple containing the predicted output tensor and the
            hidden tensor.
        """
        batch_size = x.shape[0]

        x = x.view(batch_size, -1)

        h_1 = self.dropouts[0](x)
        h_1 = F.relu(self.input_fc(h_1))

        h_hidden = h_1
        for i in range(len(self.hidden_layers)):
            h_hidden = self.dropouts[i+1](h_hidden)
            h_hidden = F.relu(self.hidden_layers[i](h_hidden))

        h_hidden = self.dropouts[-1](h_hidden)
        y_pred = self.output_fc(h_hidden)

        return y_pred, h_hidden


def create_dataset(features, labels, idxs=None):
    """Creates a PyTorch dataset object from numpy arrays.

    Parameters
    ----------
    features : np.ndarray
        The features array.
    labels : np.ndarray
        The labels array.
    idxs : np.ndarray, optional
        The indices array. Defaults to None.

    Returns
    -------
    torch.utils.data.TensorDataset
        The created dataset.
    """
    tensor_x = torch.Tensor(features) # transform to torch tensor
    tensor_y = torch.Tensor(labels).type(torch.LongTensor)

    if idxs is None:
        return TensorDataset(tensor_x,tensor_y) # create your datset

    tensor_z = torch.Tensor(idxs)

    return TensorDataset(tensor_x,tensor_y,tensor_z) # create your datset


def train(model, iterator, optimizer, criterion, device):
    """Does one epoch of training for a given torch model.

    Parameters
    ----------
    model : mlp.MLP
        The torch model.
    iterator : torch.utils.data.DataLoader
        The data iterator.
    optimizer : torch.optim.Optimizer
        The optimizer.
    criterion : torch.nn.Module
        The loss criterion.
    device : torch.device
        The device to use.

    Returns
    -------
    tuple
        A tuple containing the epoch loss and epoch accuracy.
    """
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred, _ = model(x)

        loss = criterion(y_pred, y)
        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    """Evaluates the model for the validation set.

    Parameters
    ----------
    model : mlp.MLP
        The torch model.
    iterator : torch.utils.data.DataLoader
        The data iterator.
    criterion : torch.nn.Module
        The loss criterion.
    device : torch.device
        The device to use.

    Returns
    -------
    tuple
        A tuple containing the epoch loss and epoch accuracy.
    """
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    """Sets the time it takes for each epoch to train.

    Parameters
    ----------
    start_time : float
        The start time.
    end_time : float
        The end time.

    Returns
    -------
    tuple
        A tuple containing the elapsed minutes and elapsed seconds.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def run_mlp(
    train_data,
    valid_data,
    test_sample_features,
    test_sample_classes,
    test_sample_names,
    test_group_idxs,
    input_dim,
    output_dim,
    neurons_per_layer,
    num_layers,
    num_epochs=EPOCHS,
    plot_metrics=False,
):
    """
    Run the MLP initialization and training. 
    
    Closely follows the demo
    https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb

    Parameters
    ----------
    train_data : TensorDataset
        The training data.
    valid_data : TensorDataset
        The validation data.
    test_sample_features : np.ndarray
        The test sample features.
    test_sample_classes : np.ndarray
        The test sample classes.
    test_sample_names : np.ndarray
        The test sample names.
    test_group_idxs : list of int
        The list of test group indices.
    input_dim : int
        The input dimension.
    output_dim : int
        The output dimension.
    neurons_per_layer : int
        The number of neurons per layer.
    num_layers : int
        The number of layers.
    num_epochs : int, optional
        The number of epochs. Defaults to EPOCHS.
    plot_metrics : bool, optional
        Whether to plot metrics. Defaults to False.

    Returns
    -------
    tuple
        A tuple containing the labels, names, predicted labels, maximum
        probabilities, and best validation loss.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    input_dim = train_data.shape[1]

    train_iterator = data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)

    valid_iterator = data.DataLoader(valid_data, batch_size=BATCH_SIZE)

    #Create model
    model = MLP(input_dim, output_dim, neurons_per_layer, num_layers)
    lr=LEARNING_RATE
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    best_valid_loss = float('inf')

    # for plotting
    train_acc_arr = []
    train_loss_arr = []
    val_acc_arr = []
    val_loss_arr = []
    for epoch in np.arange(0, num_epochs):

        start_time = time.monotonic()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)

        #print(model.input_fc.weight)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(
                model.state_dict(),
                os.path.join(MODEL_DIR, "superphot-model-%s.pt" % test_sample_names[0]),
            )

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if epoch % 5 == 0:
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        #plotting
        train_loss_arr.append(train_loss)
        train_acc_arr.append(train_acc)
        val_loss_arr.append(valid_loss)
        val_acc_arr.append(valid_acc)

    model.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "superphot-model-%s.pt" % test_sample_names[0])
        )
    )

    labels, pred_labels, max_probs, names = [], [], [], []

    for group_idx_set in test_group_idxs:

        test_data = create_dataset(
            test_sample_features[group_idx_set],
            test_sample_classes[group_idx_set],
            group_idx_set,
        )
        test_iterator = data.DataLoader(test_data, batch_size=BATCH_SIZE)

        images, labels_indiv, indx_indiv, probs = get_predictions(model, test_iterator, device) # pylint: disable=unused-variable
        probs_avg = np.mean(probs.numpy(), axis=0)
        save_test_probabilities(
            test_sample_names[indx_indiv.numpy().astype(int)[0]], labels_indiv[0], probs_avg
        )
        pred_labels.append(np.argmax(probs_avg))
        max_probs.append(np.amax(probs_avg))
        labels.append(labels_indiv[0])

        names.append(test_sample_names[indx_indiv.numpy().astype(int)[0]])

    if plot_metrics:
        #plotting of accuracy and loss for one epoch
        plt.plot(np.arange(0, EPOCHS), train_acc_arr, label="Training")
        plt.plot(np.arange(0, EPOCHS), val_acc_arr, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(METRICS_DIR, "accuracy_%s.png" % test_sample_names[0]))
        plt.close()

        plt.plot(np.arange(0, EPOCHS), train_loss_arr, label="Training")
        plt.plot(np.arange(0, EPOCHS), val_loss_arr, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        plt.savefig(os.path.join(METRICS_DIR, "loss_%s.png" % test_sample_names[0]))
        plt.close()

    return (
        np.array(labels).astype(int),
        np.array(names),
        np.array(pred_labels).astype(int),
        np.array(max_probs).astype(float),
        best_valid_loss,
    )
