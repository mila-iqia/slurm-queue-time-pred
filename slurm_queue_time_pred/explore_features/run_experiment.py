
import random
from slurm_queue_time_pred.explore_features.explore import explore_features
from slurm_queue_time_pred.explore_features.explore_util import get_min_loss, get_current_dir_path, scan_files
from slurm_queue_time_pred.wait_time_prediction.features import predicting_features

from slurm_queue_time_pred.wait_time_prediction import configlib
from slurm_queue_time_pred.wait_time_prediction.configlib import config as C

parser = configlib.add_parser("Explore in parallel config")
parser.add_argument('-n', '--nbr_top_features', default=10, type=int,
                    help='Number of best features to accumulate or stop parallel run after.')
parser.add_argument('--output_dir', default=None,
                    help='Directory name for this parallel run.')


def run_experiment():
    """
    Scans current exploration directory to determine next task
    """

    best_features = []

    while len(best_features) < C['nbr_top_features']:

        # STEP 1 : find out what you have
        # Get best features at current time, previous val loss obtained from best features, and features that already 
        # have been tested
        # in combination with best features
        tested_features, best_features, previous_val_loss, previous_train_loss, previous_test_loss = \
        scan_files(C['output_dir'])
        
        # Get features that have not been tested in combination with best features
        remaining_features = [i for i in predicting_features if i not in [tf[0] for tf in tested_features] 
                              and i not in best_features]

        # if no remaining features to test left, choose next best feature based on min validation loss
        if len(remaining_features) == 0:
            min_loss_feature, previous_val_loss, previous_train_loss, previous_test_loss = get_min_loss(tested_features)
            best_features.append(min_loss_feature)

            # breaks if all features were used
            if len(best_features) == len(predicting_features):
                break
            
            # choose first feature to be tested if not in best_features
            next_feature = random.choice([i for i in predicting_features if i not in best_features])
        else:
            # STEP 2 : run on something that you pick
            # we randomly choose one feature among remaining features
            next_feature = random.choice(remaining_features)
        
        # writes results to hard drive, explore one feature at a time
        explore_features(next_feature, best_features, previous_val_loss, previous_train_loss, previous_test_loss, 
                        C['learning_rate'], C['batch_size'], C['optimizer'], C['model'], C['hidden_size'], 
                        C['nbr_layers'], C['l1'], C['l2'], C['cluster'], C['mix_train_valid'], 
                        output_dir=get_current_dir_path(C['output_dir']))


if __name__ == "__main__":
    configlib.parse(save_fname="last_arguments.txt")
    print("Running with configuration:")
    configlib.print_config()
    
    run_experiment()


""""
python3 run_experiment.py -n 15 --output_dir=example

"""