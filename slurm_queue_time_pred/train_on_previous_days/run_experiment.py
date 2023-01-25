import os, json
import random
from slurm_queue_time_pred.wait_time_prediction.datasets import get_total_days, FIRST_TEST_DAY
from slurm_queue_time_pred.wait_time_prediction.run_experiment import run
from slurm_queue_time_pred.wait_time_prediction.monitor import Monitor
from slurm_queue_time_pred.wait_time_prediction.util import get_hparams_dict, save_results_to_json

from slurm_queue_time_pred.wait_time_prediction import configlib
from slurm_queue_time_pred.wait_time_prediction.configlib import config as C

parser = configlib.add_parser("Train on previous days config")
parser.add_argument('-T', '--test_day', default=3, type=int,
                    help='Day used for testing the model, model is trained on all days before this one')
parser.add_argument('--single_run', default=False, action='store_true', 
                    help='Stop experiment after only one run')

path_to_results = os.path.join(os.path.dirname(__file__), '../../results/previous_days_results')

# Maximum number of trials for each test day
MAX_TRIALS = 5


def orchestrate(cluster : str, T : int, D_hparams):
    """
    Orchestrates parallel runs so that each day is tested for max number of trials.
    
    Returns:
        This test day if trials not done, None if all trials for all days done, else selects test day at random
    """
    
    if not is_single_experiment_done(cluster, T, D_hparams):
        return T
    elif is_big_experiment_done(cluster, D_hparams):
        return None
    else:
        orchestrate(cluster, random.randint(FIRST_TEST_DAY, get_total_days(cluster), D_hparams))
        
        
def is_big_experiment_done(cluster : str, D_hparams):
    """
    Checks if all days between first and last day were tested for max number of trials.
    
    Returns:
        True if max trials done for all possible test days, else False
    """
    
    experiment_done = True
    
    for N in range(FIRST_TEST_DAY, get_total_days(cluster)):      
        experiment_done = is_single_experiment_done(cluster, N, D_hparams)    
        if not experiment_done: break
                
    return experiment_done


def is_single_experiment_done(cluster : str, T : int, D_hparams):
    """
    Checks if specified day was tested of max number of trials.
    
    Returns:
        True if max trials done for one paticular test day, else False
    """
    
    single_experiment_done = True   
    LD_all_hparams = []
        
    for filename in [file for file in os.listdir(path_to_results) if file.startswith(f'{cluster}_T{T}') 
                    and file.endswith('.json')]:
        with open(os.path.join(path_to_results, filename), "r") as f:
            data = json.load(f)
            
        LD_all_hparams.append(data['hparams'])
    
    count = LD_all_hparams.count(D_hparams)
    if count < MAX_TRIALS:       
        single_experiment_done = False 
    
    return single_experiment_done


def run():
    """
    As long as all trials for all test days are not done, this will keep running experiments 
    with random test day on specified cluster, using parameters passed as arguments to the script.
    """
    
    # Check if data is already loaded
    cluster = C['cluster']
    T = C['test_day']
    
    while T:        
        # Instantiate monitor with a 10-second delay between updates
        monitor = Monitor(10)
        val_loss, train_loss, test_loss, y_true, y_pred = run(C['features'], C['learning_rate'], 
                                                            C['batch_size'], C['optimizer'], 
                                                            C['model'], C['hidden_size'], C['nbr_layers'], 
                                                            C['l1'], C['l2'], cluster, C['mix_train_valid'], T)
        monitor.stop()
        average_gpu_usage, time_elapsed = monitor.get_average_gpu_usage_and_time_elapsed()
        
        D_results = get_hparams_dict(C['learning_rate'], C['batch_size'], C['optimizer'], C['model'], 
                                    C['hidden_size'], C['nbr_layers'], C['l1'], C['l2'], cluster, C['mix_train_valid'])
        
        D_results['results']['time_elapsed_in_s'] = float("{:.2f}".format(time_elapsed))
        D_results['results']['average_gpu_usage_percent'] = average_gpu_usage
        
        output_file = get_output_filename(cluster, T)
        save_results_to_json(D_results, val_loss, train_loss, test_loss, y_true, y_pred, output_file, path_to_results)
        
        if C['single_run']: break
        T = orchestrate(cluster, random.randint(FIRST_TEST_DAY, get_total_days(cluster), D_results))
 
    
def get_output_filename(cluster : str, T : int):
    """
    Returns:
        File's name string
    """
    
    last_count = len([file for file in os.listdir(f'{path_to_results}') if file.startswith(f'{cluster}_T{T}') 
                                                                        and file.endswith('.json')])
    current_count = str(last_count).zfill(2)
    
    return f'{cluster}_T{T}_rep{current_count}'
        

if __name__ == "__main__":   
    configlib.parse(save_fname="last_arguments.txt")
    print("Running with configuration:")
    configlib.print_config()
    run()

"""
python3 parallel_run.py -T 37 --lr=0.001 --nbr_layers=3 --batch_size=64

"""
