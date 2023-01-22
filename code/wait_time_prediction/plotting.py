
import matplotlib.pyplot as plt
import random, math
import os
import numpy as np

path_to_results = os.path.join(os.path.dirname(__file__), '../../results/plots')


def plot_predictions(y_true : list, y_pred : list, model : str, sample_size: int = 1000):
    """
    Plots targets against predictions for a certain number of samples with benchmarks at one hour, day and week.
    """
    
    # Sample from all pairs of predictions and targets
    samples = random.sample(list(zip(y_true, y_pred)), sample_size)

    y_true_tmp = []
    y_pred_tmp = []
    for i in range(sample_size):
        y_true_tmp.append(samples[i][0])
        y_pred_tmp.append(samples[i][1])

    if sample_size == len(y_true):
        sample_size = 'all'
    
    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
        
    plt.title(f'Predicting time in queue (axes in seconds)\n{model} model, {sample_size} samples')
    plt.plot(y_true_tmp, y_pred_tmp, 'go', label='Predictions', alpha=0.2)

    # Draw ideal curve
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, label='Ideal', linestyle='-', color='black', alpha=0.5)

    plt.xticks(rotation=45)
    plt.xlabel(f'log10(targets)')
    plt.ylabel('log10(predictions)')
    plt.xlim([1, 8])
    plt.ylim([1, 8])

    # Add benchmarks
    plt.vlines([3.6, 4.9, 5.8], ymin=1, ymax=8, label='~ 1 hour, day, week', color='red', linestyle='--', alpha=0.5)

    ax.set_aspect('equal', adjustable='box')

    plt.legend(loc='best')
    
    # Save plot
    plt.savefig(f'{path_to_results}/{model}_{sample_size}_predictions.png')
    plt.close()


def plot_error_distribution(y_true : list, y_pred : list, model : str):
    """
    Plot the mean squared error distribution between y_true and y_pred as histogram.
    """
    # Calculate error
    error = [y_true[i] - y_pred[i] for i in range(len(y_true))]
        
    # Count number of errors below factor of 2 and 3
    num_errors = (sum(-np.log10(2) <= e <= np.log10(2) for e in error)/len(error) * 100, 
                  sum(-np.log10(3) <= e <= np.log10(3) for e in error)/len(error) * 100)    
    print('Percent of errors below factor of 2 and 3: ', num_errors)
    
    # Plot error distribution
    plt.hist(error, bins=100, density=True)
    plt.title('Distribution of differences between log10(targets) and log10(predictions)')
    plt.xlabel('Differences')
    plt.ylabel('Frequency')
    
    plt.xticks(np.arange(math.floor(min(error)), math.ceil(max(error)), 0.5))
    
    plt.savefig(f'{path_to_results}/error_distribution_{model}.png')
    plt.close()
    