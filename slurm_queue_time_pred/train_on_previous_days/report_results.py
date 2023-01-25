import os, json
import numpy as np
import matplotlib.pyplot as plt
from operator import add, sub
from slurm_queue_time_pred.wait_time_prediction.datasets import get_total_days, FIRST_TEST_DAY
import slurm_queue_time_pred.train_on_previous_days.run_experiment as run # leave here

from slurm_queue_time_pred.wait_time_prediction import configlib
from slurm_queue_time_pred.wait_time_prediction.configlib import config as C


path_to_results = os.path.join(os.path.dirname(__file__), '../../results/previous_days_results')


def report_results(cluster : str, D_hparams : dict):       
    """
    Gets results from all experiment files that match cluster and hyperparameters as passed in arguments.
    """
    
    DD_results_across_all_days = {}
    
    # 1. Read files in results/all_features directory that start with specified cluster
    for filename in [file for file in os.listdir(f'{path_to_results}') if file.startswith(f'{cluster}')
                                                                        and file.endswith('.json')]:
        with open(os.path.join(path_to_results, filename), "r") as f:
            data = json.load(f)
        
        # 2. Verify if hyperparameters match, then retrieve train and test MSE and add them to correct day in dict
        # Do the same for average GPU usage and time elapsed for experiment
        if data['hparams'] == D_hparams:
            day = int(filename.split('_')[1][1:])

            if day not in DD_results_across_all_days:
                DD_results_across_all_days[day] = {}
                DD_results_across_all_days[day]['train_MSE'] = []
                DD_results_across_all_days[day]['test_MSE'] = []
                DD_results_across_all_days[day]['gpu_usage'] = []
                DD_results_across_all_days[day]['time_elapsed'] = []
        
            DD_results_across_all_days[day]['train_MSE'].append(data['results']['train_loss'])
            DD_results_across_all_days[day]['test_MSE'].append(data['results']['test_loss'])
            DD_results_across_all_days[day]['gpu_usage'].append(data['results']['average_gpu_usage_percent'])
            DD_results_across_all_days[day]['time_elapsed'].append(data['results']['time_elapsed_in_s'])
            
            # Example
            # { 1: {'train_MSE': [1,2,3,4,5], 'test_MSE':  [1,2,3,4,5], 'gpu_usage': [80,70,60,90,40],
            # 'time_elapsed':[4500,1200,3600,500,2400] }, 2: ... }
    
    # Sort dict by key
    DD_results_across_all_days = {k: DD_results_across_all_days[k] for k in sorted(DD_results_across_all_days)}
    
    L_test_days = list(DD_results_across_all_days.keys())
    
    L_mean_train_MSE, L_mean_test_MSE, L_mean_gpu_usage, L_mean_time_elapsed, L_std_train_MSE, L_std_test_MSE, \
    L_median_train_MSE, L_median_test_MSE, L_min_train_MSE, L_max_train_MSE, L_min_test_MSE, L_max_test_MSE, \
    L_above_mid_train_MSE, L_mid_train_MSE, L_below_mid_train_MSE, L_above_mid_test_MSE, L_mid_test_MSE, \
    L_below_mid_test_MSE = process_results(DD_results_across_all_days)
    
    plot_average_train_and_test_MSE(L_test_days, L_mean_train_MSE, L_mean_test_MSE, cluster, D_hparams['model'])
    
    plot_average_time_and_gpu_usage(L_test_days, L_mean_gpu_usage, L_mean_time_elapsed, cluster, D_hparams['model'])
    
    plot_average_train_and_test_MSE_with_std(L_test_days, 
                                             L_mean_train_MSE, L_mean_test_MSE, 
                                             L_std_train_MSE, L_std_test_MSE, 
                                             cluster, D_hparams['model'])
    
    plot_median_train_and_test_MSE_with_min_max(L_test_days, 
                                                L_median_train_MSE, L_median_test_MSE, 
                                                L_min_train_MSE, L_max_train_MSE, L_min_test_MSE, L_max_test_MSE, 
                                                cluster, D_hparams['model'])
    
    plot_mid_range_train_and_test_MSE(L_test_days, 
                                      L_above_mid_train_MSE, L_mid_train_MSE, L_below_mid_train_MSE, 
                                      L_above_mid_test_MSE, L_mid_test_MSE, L_below_mid_test_MSE, 
                                      cluster, D_hparams['model'])
    
    print(f'New files at {path_to_results}')
    

def process_results(DD_results_across_all_days : dict):
    """
    Computes relevant metrics to display in plots, such as mean, standard deviation, median, min, max and range.
    These metrics are applied to train and test MSE, as well as average GPU usage and time elapsed for group of
    experiments corresponding to each test day.
    
    Returns:
        Relevant metrics computed for each test day, if data available.
    """
    
    L_mean_train_MSE = []
    L_mean_test_MSE = []
    L_mean_time_elapsed = []
    L_mean_gpu_usage = []
    L_std_train_MSE = []
    L_std_test_MSE = []
    L_median_train_MSE = []
    L_median_test_MSE = []
    L_min_train_MSE = []
    L_max_train_MSE = []
    L_min_test_MSE = []
    L_max_test_MSE = []
    L_above_mid_train_MSE = []
    L_mid_train_MSE = []
    L_below_mid_train_MSE = []
    L_above_mid_test_MSE = []
    L_mid_test_MSE = []
    L_below_mid_test_MSE = []
    
    for test_day in DD_results_across_all_days:  
        print(f"Processing test day {test_day}")
        train_MSE = DD_results_across_all_days[test_day]['train_MSE']
        test_MSE = DD_results_across_all_days[test_day]['test_MSE']
        gpu_usage = DD_results_across_all_days[test_day]['gpu_usage']
        time_elapsed = DD_results_across_all_days[test_day]['time_elapsed']
        
        L_mean_train_MSE.append(np.mean(train_MSE))
        L_mean_test_MSE.append(np.mean(test_MSE))
        L_mean_gpu_usage.append(np.mean(gpu_usage))
        L_mean_time_elapsed.append(np.mean(time_elapsed))   
        L_std_train_MSE.append(np.std(train_MSE))
        L_std_test_MSE.append(np.std(test_MSE))
        L_median_train_MSE.append(np.median(train_MSE))
        L_median_test_MSE.append(np.median(test_MSE))
        L_min_train_MSE.append(min(train_MSE))
        L_max_train_MSE.append(max(train_MSE))
        L_min_test_MSE.append(min(test_MSE))
        L_max_test_MSE.append(max(test_MSE))
        below_mid_train_MSE, mid_train_MSE, above_mid_train_MSE = get_mid_range(sorted(train_MSE))
        below_mid_test_MSE, mid_test_MSE, above_mid_test_MSE = get_mid_range(sorted(test_MSE))
        L_above_mid_train_MSE.append(above_mid_train_MSE)
        L_mid_train_MSE.append(mid_train_MSE)
        L_below_mid_train_MSE.append(below_mid_train_MSE)
        L_above_mid_test_MSE.append(above_mid_test_MSE)
        L_mid_test_MSE.append(mid_test_MSE)
        L_below_mid_test_MSE.append(below_mid_test_MSE)
    
    return  L_mean_train_MSE, L_mean_test_MSE, L_mean_gpu_usage, L_mean_time_elapsed, \
            L_std_train_MSE, L_std_test_MSE, \
            L_median_train_MSE, L_median_test_MSE, \
            L_min_train_MSE, L_max_train_MSE, L_min_test_MSE, L_max_test_MSE, \
            L_above_mid_train_MSE, L_mid_train_MSE, L_below_mid_train_MSE, \
            L_above_mid_test_MSE, L_mid_test_MSE, L_below_mid_test_MSE


def get_mid_range(L_range : list):
    """
    Returns the middle value of a list and values around it, if list has at least 3 elements.
    """
    if len(L_range) % 2 == 0:
            mid_index = int((len(L_range) + 1) / 2)    
    else:
        mid_index = int(len(L_range) / 2)
            
    if len(L_range) >= 3:
        return L_range[mid_index - 1], L_range[mid_index], L_range[mid_index + 1]
    elif len(L_range) == 2:
        return L_range[mid_index - 1], L_range[mid_index], np.nan
    elif len(L_range) == 1:
        return np.nan, L_range[mid_index], np.nan
      

def plot_average_time_and_gpu_usage(L_test_days, L_mean_gpu_usage, L_mean_time_elapsed, cluster : str, model : str):
    """
    Plots average GPU usage and time elapsed versus test, if data available, in separate charts. 
    Saves it in results/previous_days_results/plots folder.
    """
    
    fig = plt.figure()
    fig.patch.set_alpha(0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    
    ax1.plot(L_test_days, L_mean_gpu_usage, color='red')
    ax2.plot(L_test_days, L_mean_time_elapsed, color='blue')
    fig.suptitle(   f'Average GPU usage and time elapsed for {model} model trained ' +
                    f'\nwith days preceding test day using data from {cluster[0].upper() + cluster[1:]}')
    ax1.set_xlim([FIRST_TEST_DAY, get_total_days(cluster)])
    ax1.set_ylabel('Average GPU usage in percent')
    ax1.set_ylim([0,100])
    ax2.set_xlim([FIRST_TEST_DAY, get_total_days(cluster)])
    ax2.set_ylabel('Time elapsed in seconds')
    ax2.set_ylim(0.0)
    
    plt.xlabel('Test day')
    plt.tight_layout()
    
    fig.savefig(f"{path_to_results}/plots/{cluster}_average_GPU_usage_and_time")
            
            
def plot_average_train_and_test_MSE(L_test_days, L_mean_train_MSE, L_mean_test_MSE, 
                                    cluster : str, model : str):  
    """
    Plots average train and test MSE versus test, if data available, in same chart. 
    Saves it in results/previous_days_results/plots folder.
    """
     
    fig = plt.figure()
    fig.patch.set_alpha(0)
    
    plt.plot(L_test_days, L_mean_train_MSE, color='green', label='Train MSE')
    plt.plot(L_test_days, L_mean_test_MSE, color='orange', label='Test MSE')
    
    plt.title(f'Average train and test MSE for {model} model trained\nwith days preceding test day using data from {cluster[0].upper() + cluster[1:]}')
    plt.xlabel('Test day')
    plt.xlim([FIRST_TEST_DAY, get_total_days(cluster)])
    plt.ylabel('Average MSE')
    plt.ylim(0.0)
    plt.tight_layout()
    plt.legend()

    fig.savefig(f"{path_to_results}/plots/{cluster}_average_MSE")


def plot_average_train_and_test_MSE_with_std(L_test_days, L_mean_train_MSE, L_mean_test_MSE, 
                                             L_std_train_MSE, L_std_test_MSE, cluster : str, model : str): 
    """
    Plots average train and test MSE with bands corresponding to standard deviation versus test, 
    if data available, in same chart. Saves it in results/previous_days_results/plots folder.
    """
       
    fig = plt.figure()
    fig.patch.set_alpha(0)  
    
    plt.plot(L_test_days, L_mean_train_MSE, label='Train MSE', color='green')
    plt.fill_between(L_test_days, 
                     L_mean_train_MSE, 
                     list(map(add, L_mean_train_MSE, L_std_train_MSE)), color='green', alpha=0.2)
    plt.fill_between(L_test_days, 
                     L_mean_train_MSE, 
                     list(map(sub, L_mean_train_MSE, L_std_train_MSE)), color='green', alpha=0.2)
   
    plt.plot(L_test_days, L_mean_test_MSE, label='Test MSE', color='orange')
    plt.fill_between(L_test_days, 
                     L_mean_test_MSE, 
                     list(map(add, L_mean_test_MSE, L_std_test_MSE)), color='orange', alpha=0.2)
    plt.fill_between(L_test_days, 
                     L_mean_test_MSE, 
                     list(map(sub, L_mean_test_MSE, L_std_test_MSE)), color='orange', alpha=0.2)
    
    plt.title(  f'Average train and test MSE with Standard Deviation for {model} model trained ' +
                f'\nwith days preceding test day using data from {cluster[0].upper() + cluster[1:]}')
    plt.xlabel('Test day')
    plt.xlim([FIRST_TEST_DAY, get_total_days(cluster)])
    plt.ylabel('Average MSE +/- std')
    plt.ylim(0.0)
    plt.tight_layout()
    plt.legend()
    
    fig.savefig(f"{path_to_results}/plots/{cluster}_average_MSE_with_std")
    
    
def plot_median_train_and_test_MSE_with_min_max(L_test_days, L_median_train_MSE, L_median_test_MSE, 
                                                L_min_train_MSE, L_max_train_MSE, L_min_test_MSE, L_max_test_MSE,
                                                cluster : str, model : str):
    """
    Plots median train and test MSE with bands corresponding to min and max values versus test, 
    if data available, in same chart. Saves it in results/previous_days_results/plots folder.
    """
    
    fig = plt.figure()
    fig.patch.set_alpha(0)
    
    plt.plot(L_test_days, L_median_train_MSE, label='Train MSE', color='green')
    plt.fill_between(L_test_days, 
                     L_median_train_MSE, 
                     L_min_train_MSE, color='green', alpha=0.2)
    plt.fill_between(L_test_days, 
                     L_median_train_MSE, 
                     L_max_train_MSE, color='green', alpha=0.2)
   
    plt.plot(L_test_days, L_median_test_MSE, label='Test MSE', color='orange')
    plt.fill_between(L_test_days, 
                     L_median_test_MSE, 
                     L_min_test_MSE, color='orange', alpha=0.2)
    plt.fill_between(L_test_days, 
                     L_median_test_MSE, 
                     L_max_test_MSE, color='orange', alpha=0.2)
    
    plt.title(  f'Median train and test MSE with Min-Max for {model} model trained ' +
                f'\nwith days preceding test day using data from {cluster[0].upper() + cluster[1:]}')
    plt.xlabel('Test day')
    plt.xlim([FIRST_TEST_DAY, get_total_days(cluster)])
    plt.ylabel('Median MSE with min-max')
    plt.ylim(0.0)
    plt.tight_layout()
    plt.legend()
    
    fig.savefig(f"{path_to_results}/plots/{cluster}_median_MSE_min_max")
    

def plot_mid_range_train_and_test_MSE(L_test_days,
                                      L_above_mid_train_MSE, L_mid_train_MSE, L_below_mid_train_MSE, 
                                      L_above_mid_test_MSE, L_mid_test_MSE, L_below_mid_test_MSE, 
                                      cluster : str, model : str):
    """
    Plots mid train and test MSE with bands corresponding to values around it, 
    if data available, in same chart. Saves it in results/previous_days_results folder.
    """
                
    fig = plt.figure()
    fig.patch.set_alpha(0) 
    
    plt.plot(L_test_days, L_mid_train_MSE, label='Train MSE', color='green')
    plt.fill_between(L_test_days, 
                        L_mid_train_MSE,
                        L_below_mid_train_MSE, color='green', alpha=0.2)
    plt.fill_between(L_test_days, 
                        L_mid_train_MSE, 
                        L_above_mid_train_MSE, color='green', alpha=0.2)

    plt.plot(L_test_days, L_mid_test_MSE, label='Test MSE', color='orange')
    plt.fill_between(L_test_days, 
                        L_mid_test_MSE, 
                        L_above_mid_test_MSE, color='orange', alpha=0.2)
    plt.fill_between(L_test_days, 
                        L_mid_test_MSE, 
                        L_below_mid_test_MSE, color='orange', alpha=0.2)
    
    plt.title(  f'Mid-range train and test MSE for {model} model trained ' +
                f'\nwith days preceding test day using data from {cluster[0].upper() + cluster[1:]}')
    plt.xlabel('Test day')
    plt.xlim([FIRST_TEST_DAY, get_total_days(cluster)])
    plt.ylabel('Mid-range MSE')
    plt.ylim(0.0)
    plt.tight_layout()
    plt.legend()
    
    fig.savefig(f"{path_to_results}/plots/{cluster}_mid_range_MSE")
    

if __name__ == "__main__":
    configlib.parse(save_fname="last_arguments.txt")
    print("Running with configuration:")
    configlib.print_config()
    
    D_hparams = {}
    D_hparams["cluster"] = C['cluster']
    D_hparams["learning_rate"] = C['learning_rate']
    D_hparams["batch_size"] = C['batch_size']
    D_hparams["optimizer"] = C['optimizer'] 
    D_hparams["model"] = C['model']
    D_hparams["hidden_size"] = C['hidden_size']
    D_hparams["nbr_layers"] = C['nbr_layers']
    D_hparams["mix_train_valid"] = C['mix_train_valid']
    D_hparams["l1"] = C['l1']
    D_hparams["l2"] = C['l2']
    
    report_results(C['cluster'], D_hparams)
    
"""
python3 report_results.py --cluster=cedar --learning_rate=0.001 --batch_size=128 --optimizer=adam --model=NN 
--hidden_size=128 --nbr_layers=6 --mix_train_valid=True

See run_experiment.py for default values

"""