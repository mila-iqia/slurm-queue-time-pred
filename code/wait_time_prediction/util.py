
import numpy as np
import matplotlib.pyplot as plt
import os
import json


def normalize(A, mus=None, sigmas=None):
    """
	This doesn't actually produce a Normal distribution,
	but it returns an equivalent to A whereby each column
	individually has a mean of 0.0 and a variance of 1.0.

    In a situation in which we have defined normalization
    constants (mu, sigma) based on data from the training set,
    we will want to apply the same values to the test set
    instead of computing the (mu, sigma) for the test set itself.
    This is done using the optional arguments `mus` and `sigmas`
    whose values are obtained previously through a call to
    this `normalize` function.
	"""
    
    assert isinstance(A, np.ndarray)
    (N, D) = A.shape
    # assert D < N  # just a sanity check
    
    if mus is None or sigmas is None:
	    # assert mus is None and sigmas is None
        mus = A.mean(axis=0, keepdims=True)
        sigmas = A.std(axis=0, keepdims=True) + 1e-4
    
    Z = (A - mus) / sigmas
    assert Z.shape == (N, D)
    return mus, sigmas, Z


def unnormalize(Z, mus, sigmas):
    return mus + (Z * sigmas)


def check_distribution(dist, dist_norm, feature_name, nb_bins):  
    """Plots distribution of a feature (non normalized and normalized)."""  
    # Creating histogram
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize =(7, 4))
    ax1.hist(dist, bins=nb_bins)
    ax2.hist(dist_norm, bins=nb_bins)
    # ax2.set_ylabel('Normalized')
    
    fig.suptitle(f'Distribution of {feature_name}')
    fig.patch.set_alpha(0)

    # Show plot
    plt.show()

  
def save_results_to_json(D_results, val_loss : float, train_loss : float, test_loss : float, 
                         y_true : list, y_pred : list, output_file : str, path_to_results : str):
    """
    Saves results (MSE, predictions, targets and metrics) of each experiment (one day, one cluster) 
    in its own json file.
    """
    D_results['results']['val_loss'] = float("{:.4f}".format(val_loss))
    D_results['results']['train_loss'] = float("{:.4f}".format(train_loss))
    D_results['results']['test_loss'] = float("{:.4f}".format(test_loss))
    D_results['y_true'] = y_true
    D_results['y_pred'] = y_pred

    with open(f'{path_to_results}/{output_file}.json', 'w') as f:
        json.dump(D_results, f, sort_keys=True, indent=4)
 
        
def get_hparams_dict(learning_rate, batch_size, optimizer, model, hidden_size, nbr_layers, l1, l2, cluster, mix_train_valid):
    return {"hparams" : {   "cluster": cluster,
                            "learning_rate": learning_rate, 
                            "batch_size": batch_size, 
                            "optimizer": optimizer, 
                            "model": model,
                            "hidden_size": hidden_size,
                            "nbr_layers": nbr_layers,
                            "l1": l1,
                            "l2": l2,
                            "mix_train_valid": mix_train_valid
                        }, 
            "results": {} }
    

def evaluate_with_best_constant(train_loader, valid_loader, test_loader):
    """Computes MSE for constant prediction with mean of labels (y) in train, valid and test sets."""

    for (segment_name, segment_loader) in [
        ("train", train_loader), ("valid", valid_loader), ("test", test_loader)]:

        best_constant_for_y = get_best_constant_for_y(segment_loader)

        squared_errors = [np.square(y - best_constant_for_y) 
                          for (_, labels) in segment_loader for y in labels.data.cpu().numpy()]
        mse_with_constant_prediction = np.mean(squared_errors)
        
        print(f"MSE for constant prediction on {segment_name} : {mse_with_constant_prediction}.")
        

def get_best_constant_for_y(loader):
    """
    Returns mean constant for y (labels) in a given dataset.
    """
    L = []
    for (_, labels) in loader:
        L.extend(labels.data.cpu().numpy())
        
    best_constant_for_y = np.mean(L)
    
    return best_constant_for_y


def count_files_in_dir(path_to_dir):
    """Returns number of files in a directory."""
    return len([name for name in os.listdir(path_to_dir) if os.path.isfile(os.path.join(path_to_dir, name))])