
import torch
import numpy as np
import time
import os

from code.wait_time_prediction.model import linearRegression, NN
from code.wait_time_prediction.datasets import create_datasets, load_data
from code.wait_time_prediction.plotting import plot_error_distribution, plot_predictions
from code.wait_time_prediction.util import save_results_to_json, get_hparams_dict, evaluate_with_best_constant

"""
W&B setup
"""

import wandb

wandb.init(project="mila-cc", entity="biancapopa")

"""
Configure script arguments
"""

from code.wait_time_prediction import configlib
from code.wait_time_prediction.configlib import config as C

parser = configlib.add_parser("Run experiment config")
parser.add_argument('--features', default="",
                    help='Comma-separated list of features')
parser.add_argument('--model', default='NN',
                    help='Model to use')
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float,
                    help='Learning rate')
parser.add_argument('--batch_size', default=128, type=int,
                    help='Batch size')
parser.add_argument('-o', '--optimizer', default='adam',
                    help='Optimizer')
parser.add_argument('--hidden_size', default=128, type=int,
                    help='Number of neurons in hidden layers')
parser.add_argument('--nbr_layers', default=6, type=int,
                    help='Number of hidden layers')
parser.add_argument('--l1', default=0.0, type=float,
                    help='L1 penalty for weights and biases (l1_lambda)')
parser.add_argument('--l2', default=0.0, type=float,
                    help='L2 penalty for weights and biases (l2_lambda)')
parser.add_argument('--mix_train_valid', default=False, action='store_true',
                    help='If arg is present, train and valid samples are taken from same pool')
parser.add_argument('-c', '--cluster', default='cedar',
                    help='Cluster from which to take data')

path_to_results = os.path.join(os.path.dirname(__file__), '../../results/reports')
path_to_models = os.path.join(os.path.dirname(__file__), '../../models')


def run(features, lr : float, batch_size : int, optimizer : str, model_str : str, hidden_size : int, nbr_layers : int, 
        l1_lambda : float, l2_lambda : float, cluster : str, mix_train_valid : bool, test_day=''):
    """Builds model and optimizer, creates dataloaders and calls training and testing loops.
    
    Returns:
        Validation, training and test losses
        Predictions and targets
    """
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(test_day, cluster)

    # If no features are specified, we take all predicting features
    if features == '' or features[0] == '':
        features = X_train.columns.tolist()
    
    print('Features: ', features)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input_dim, output_dim = len(features), 1
    model = build_model(model_str, input_dim, output_dim, hidden_size, nbr_layers)
    model = model.to(device)

    # We choose mean squared error as loss function
    criterion = torch.nn.MSELoss() 
    optimizer = build_optimizer(model, optimizer, lr)

    train_loader, valid_loader, test_loader = create_datasets(X_train, y_train, 
                                                              X_valid, y_valid, 
                                                              X_test, y_test, 
                                                              features, batch_size, 
                                                              mix_train_valid)
    evaluate_with_best_constant(train_loader, valid_loader, test_loader)

    print('Training model')
    start = time.time()

    val_loss, train_loss, model_state = train_model(train_loader=train_loader,
                                                    valid_loader=valid_loader,
                                                    test_loader=test_loader,
                                                    model=model, 
                                                    device=device,
                                                    criterion=criterion, 
                                                    optimizer=optimizer,
                                                    l1_lambda=l1_lambda,
                                                    l2_lambda=l2_lambda)
    
    end = time.time()
    training_time = end - start
    print('Training time: ', training_time, ' s')

    # PATH = 'mila_cc/saved_models/last_model.pth'
    # model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

    model.load_state_dict(model_state)
    model.eval()
    y_true, y_pred, test_loss = test_model(test_loader, model, criterion, device)
    D_hparams = get_hparams_dict(lr, batch_size, C['optimizer'], model_str, hidden_size, nbr_layers, l1_lambda, 
                                 l2_lambda, cluster, mix_train_valid)
    save_results_to_json(D_hparams, val_loss, train_loss, test_loss, y_true, y_pred, 'last_run', path_to_results)
    plot_predictions(y_true, y_pred, model_str, len(y_true))
    plot_predictions(y_true, y_pred, model_str)
    plot_error_distribution(y_true, y_pred, model_str)

    return val_loss, train_loss, test_loss, y_true, y_pred
    # return test_loss, y_true, y_pred


def build_model(model : str, input_dim : int, output_dim : int, hidden_size : int, nbr_layers : int):
    if model == "linear":
        model = linearRegression(input_dim, output_dim)
    elif model == "NN":
        model = NN(input_dim, output_dim, hidden_size, nbr_layers)
    return model


def build_optimizer(model, optimizer : str, learning_rate : float):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=learning_rate)
    return optimizer


def train_model(train_loader, valid_loader, test_loader, model, device, criterion, optimizer, 
                l1_lambda, l2_lambda, max_epochs=1000):
    """Goes through many training epochs and does validation each time.
    
    Returns:
        Validation loss at checkpoint
        Training loss at checkpoint
        Model state at checkpoint
    """
    
    epoch_checkpoint = np.inf
    valid_loss_checkpoint = np.inf
    train_loss_checkpoint = np.inf
    model_state_checkpoint = model.state_dict()
    wandb.watch(model, criterion, log="all")

    for epoch in range(max_epochs):

        # One pass across the training set.
        train_epoch_accum_loss = []
        model.train()
        for (inputs, labels) in train_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)

            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, 
            # dont want to cumulate gradients
            optimizer.zero_grad()

            # get output from the model, given the inputs
            outputs = model(inputs)

            # get loss for the predicted output
            loss = criterion(outputs, labels)
            
             # Add L1 and L2 regularizations to model weights and biases
            l1 = l1_lambda * torch.norm(torch.cat([parameter.view(-1) for parameter in model.parameters()]), 1)
            l2 = l2_lambda * torch.norm(torch.cat([parameter.view(-1) for parameter in model.parameters()]), 2)
      
            loss += l1
            loss += l2

            # get gradients w.r.t to parameters
            loss.backward()

            # update parameters
            optimizer.step()

            train_epoch_accum_loss.append(loss.item())
        train_epoch_loss = np.array(train_epoch_accum_loss).mean()
        # wandb.log({"train": {"loss": train_epoch_loss}}, step=epoch)


        # One pass across the validation set.
        model.eval()
        valid_epoch_accum_loss = []
        for i, (inputs, labels) in enumerate(valid_loader):
            # no training here, no updates to the model
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_epoch_accum_loss.append(loss.item())

        valid_epoch_loss = np.array(valid_epoch_accum_loss).mean()
        # wandb.log({"val": {"loss": valid_epoch_loss}}, step=epoch)
        
        # One pass across the test set.
        test_epoch_accum_loss = []
        for i, (inputs, labels) in enumerate(test_loader):
            # no training here, no updates to the model
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_epoch_accum_loss.append(loss.item())

        test_epoch_loss = np.array(test_epoch_accum_loss).mean()

        print(f'epoch {epoch}, train_epoch_loss {train_epoch_loss:0.4f}, valid_epoch_loss {valid_epoch_loss:0.4f}, test_epoch_loss {test_epoch_loss:0.4f}')
        
        # Early stopping
        if ((valid_epoch_loss >= 1.01 * valid_loss_checkpoint) and (epoch - epoch_checkpoint >= 50)):
            print('Early stopping!\nStart to test.')
            break   
        
        if np.isnan(valid_epoch_loss):
            print('Training crashed.')
            break
        
        if valid_epoch_loss < valid_loss_checkpoint:
            valid_loss_checkpoint = valid_epoch_loss
            train_loss_checkpoint = train_epoch_loss
            epoch_checkpoint = epoch
            model_state_checkpoint = model.state_dict()
            print('Checkpoint !')
            
        # Save model and parameters
        wandb.log({"val_loss_at_each_step": valid_epoch_loss}, step=epoch)
        wandb.log({"train_loss_at_each_step": train_epoch_loss}, step=epoch)
        wandb.log({"test_loss_at_each_step": test_epoch_loss}, step=epoch)
            
    PATH = f'{path_to_models}/last_model.pth'
    torch.save(model_state_checkpoint, PATH)
    wandb.run.summary["epoch_checkpoint"] = epoch_checkpoint
    wandb.run.summary["val_loss_checkpoint"] = valid_loss_checkpoint
    wandb.run.summary["train_loss_checkpoint"] = train_loss_checkpoint
    
    return  valid_loss_checkpoint, train_loss_checkpoint, model_state_checkpoint

def log_weights(weights, features):
    """Logs weights of trained model to W&B (only for linear model)."""
    
    weights = np.array(weights).reshape(len(features),-1)
    wandb.log({"weight" : wandb.plot.line_series(
                       xs=[ i for i in range(wandb.config["epochs"]) ], 
                       ys=weights,
                       keys=features,
                       title="Weight curve for training",
                       xname="Epoch")})


def test_model(test_loader, model, criterion, device):
    """Predicts values for y in testing set.
    
    Returns:
        Predictions and targets
        Test loss
    """
    
    x_test = []
    y_pred = []
    y_true = []

    test_loss_accum = 0.0
    steps = 0

    # iterate over test data
    for i, (inputs, labels) in enumerate(test_loader):
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) # Feed Network
            
            
            inputs = inputs.data.cpu().numpy()
            x_test.extend(inputs)

            loss = criterion(outputs, labels)

            test_loss_accum += loss.item() # Save Loss
            steps += 1
            
            outputs = outputs.data.cpu().numpy()

            y_pred.extend(outputs) # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

    test_loss = test_loss_accum / steps
    print('Test loss: ', test_loss)
    y_true = list(map(float, y_true))
    y_pred = list(map(float, y_pred))
    wandb.run.summary["test_loss"] = test_loss
    return y_true, y_pred, test_loss


if __name__ == "__main__":
    configlib.parse(save_fname="last_arguments.txt")
    print("Running with configuration:")
    configlib.print_config()

    wandb.config = {
        "learning_rate": C['learning_rate'],
        "batch_size": C['batch_size'],
        "optimizer": C['optimizer'],
        "model": C['model'],
        "hidden_size": C['hidden_size'],
        "nbr_layers": C['nbr_layers'],
        "l1_lambda": C['l1'],
        "l2_lambda": C['l2'],
        "cluster": C['cluster'],
        "mix_train_valid": C['mix_train_valid'],
    }
        
    features = C['features'].split(",")
    assert isinstance(features, list)

    run(features, wandb.config['learning_rate'], wandb.config['batch_size'], wandb.config['optimizer'], 
        wandb.config['model'], wandb.config['hidden_size'], wandb.config['nbr_layers'], wandb.config['l1_lambda'], 
        wandb.config['l2_lambda'], wandb.config['cluster'], wandb.config['mix_train_valid'])


""""
python3 run_experiment.py --features=eligible_time --learning_rate=0.001

"""
