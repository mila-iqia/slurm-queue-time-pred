
from code.wait_time_prediction.run_experiment import run
import json
from code.explore_features.explore_util import get_current_dir_path, get_hparams_dict
import os,json
import uuid


def explore_features(next_feature : str, L_best_features, previous_val_loss, previous_train_loss, previous_test_loss, 
                    learning_rate, batch_size, optimizer : str, model : str, hidden_size, nbr_layers, l1, l2, cluster,
                    mix_train_valid, output_dir : str):
    """
    Runs ML pipeline with combination of best features to date and new feature to test and saves results in json file.
    """

    DD_results = get_hparams_dict(cluster, learning_rate, batch_size, optimizer, model, hidden_size, nbr_layers, l1, l2)

    L_current_round_features = L_best_features + [next_feature]
    val_loss, train_loss, test_loss, _, _ = run(L_current_round_features, learning_rate, batch_size, optimizer, model, 
                                          hidden_size, nbr_layers, l1, l2, cluster, mix_train_valid)
    
    DD_results['results']['previous_rounds_best_features'] = L_best_features
    DD_results['results']['previous_rounds_val_loss'] = previous_val_loss
    DD_results['results']['previous_rounds_train_loss'] = previous_train_loss
    DD_results['results']['previous_rounds_test_loss'] = previous_test_loss
    DD_results['results']['current_round_feature_add'] = next_feature
    DD_results['results']['current_round_val_loss'] = float("{:.4f}".format(val_loss))
    DD_results['results']['current_round_train_loss'] = float("{:.4f}".format(train_loss))
    DD_results['results']['current_round_test_loss'] = float("{:.4f}".format(test_loss))
    
    unique_id = uuid.uuid4()
    with open(f'{output_dir}/round_{len(L_best_features)}__{next_feature}__{unique_id}.json', 'w') as f:
        json.dump(DD_results, f, sort_keys=True, indent=4)
        print(f'New file at {output_dir}')


def get_deterioration_metrics(nbr_best_features : int, output_dir : str):
    """
    Scans files in specified directory and computes frequency at which validation, training and test loss deteriorate
    from previous step. Also, computes the mean of this deterioration.
    
    Returns:
        Dictionary with deterioration metrics at each step
    """
    
    DDD_deter_metrics = {}
    DDD_deter_metrics['valid_loss_deter'] = {}
    DDD_deter_metrics['train_loss_deter'] = {}
    DDD_deter_metrics['test_loss_deter'] = {}

    for i in range(1, nbr_best_features + 1):
        total = 0
        worse_valid = 0
        worse_train = 0
        worse_test = 0
        L_diff_valid = []
        L_diff_train = []
        L_diff_test = []

        for file_name in [file for file in os.listdir(get_current_dir_path(output_dir)) 
        if file.endswith('.json') and file.startswith(f'round_{i}')]:
            
            with open(f'{get_current_dir_path(output_dir)}/{file_name}') as json_file:
                data = json.load(json_file)
                total += 1

            # For validation loss
            if data['results']['current_round_val_loss'] > data['results']['previous_rounds_val_loss']:
                # not good
                worse_valid += 1
                L_diff_valid.append(data['results']['current_round_val_loss'] - data['results']['previous_rounds_val_loss'])

            # For train loss
            if data['results']['current_round_train_loss'] > data['results']['previous_rounds_train_loss']:
                # not good
                worse_train += 1
                L_diff_train.append(data['results']['current_round_train_loss'] - data['results']['previous_rounds_train_loss'])

            # For test loss
            if data['results']['current_round_test_loss'] > data['results']['previous_rounds_test_loss']:
                # not good
                worse_test += 1
                L_diff_test.append(data['results']['current_round_test_loss'] - data['results']['previous_rounds_test_loss'])
        
        if total != 0: 
            deter_freq_valid = worse_valid / total
            deter_freq_train = worse_train / total
            deter_freq_test = worse_test / total
        else: 
            deter_freq_valid = 0
            deter_freq_train = 0
            deter_freq_test = 0

        if len(L_diff_valid) != 0: mean_diff_valid = sum(L_diff_valid) / len(L_diff_valid)
        else: mean_diff_valid = 0

        if len(L_diff_train) != 0: mean_diff_train = sum(L_diff_train) / len(L_diff_train)
        else: mean_diff_train = 0

        if len(L_diff_test) != 0: mean_diff_test = sum(L_diff_test) / len(L_diff_test)
        else: mean_diff_test = 0

        DDD_deter_metrics['valid_loss_deter'][f'Round {i}'] = { "Frequency": float("{:.4f}".format(deter_freq_valid)), 
                                                         "Mean": float("{:.4f}".format(mean_diff_valid)) }
        DDD_deter_metrics['train_loss_deter'][f'Round {i}'] = { "Frequency": float("{:.4f}".format(deter_freq_train)), 
                                                         "Mean": float("{:.4f}".format(mean_diff_train)) }
        DDD_deter_metrics['test_loss_deter'][f'Round {i}'] = { "Frequency": float("{:.4f}".format(deter_freq_test)), 
                                                        "Mean": float("{:.4f}".format(mean_diff_test)) }
    
    return DDD_deter_metrics


def get_results(nbr_best_features : int, output_dir : str):
    """
    Returns:
        List of best cumulated features at each step with corresponding validation, training and test loss
        Hyperparameters of current exploration runs
    """
    
    L_results_at_each_step = []
    D_hparams = {}
    
    for i in range(1, nbr_best_features + 1):

        for file_name in [file for file in os.listdir(get_current_dir_path(output_dir)) 
        if file.endswith('.json') and file.startswith(f'round_{i}__')]:
                        
            with open(f'{get_current_dir_path(output_dir)}/{file_name}') as json_file:
                data = json.load(json_file)

            D_hparams = data['hyperparams']
            L_results_at_each_step.append((data['results']['previous_rounds_best_features'], 
                                        float("{:.4f}".format(data['results']['previous_rounds_val_loss'])),
                                        float("{:.4f}".format(data['results']['previous_rounds_train_loss'])),
                                        float("{:.4f}".format(data['results']['previous_rounds_test_loss']))))
            break
    
    return L_results_at_each_step, D_hparams
    