
import os,json

path_to_results = os.path.join(os.path.dirname(__file__), '../../results/exploration_results')


def get_current_dir_path(output_dir : str):
    """
    If directory with name 'output_dir' at location results/exploration_results doesn't exist, 
    a new directory with that name is created with subdirectories reports and plots.
    If 'output_dir' is None, location results/exploration_results is returned.
    
    Returns:
        Path of directory with this name
    """
    
    if output_dir:
        newpath = [ f'{path_to_results}/{output_dir}',  
                f'{path_to_results}/{output_dir}/reports',
                f'{path_to_results}/{output_dir}/plots']
        for path in newpath:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
        return path_to_results + '/' + output_dir
    else:
        return path_to_results
    
    
def scan_files(output_dir : str):
    """
    Scans files in specified dir of results/exploration_results to look for features that have already been tried, so
    we can continue exploration with remaining features. Also, retrieves information from previous step.
    
    Returns:
        List of tested features
        List of best features
        Previous validation, train and test losses
    """
    
    L_tested_features = []
    L_results = []
    max_length = 0

    L_best_features = []
    previous_val_loss = None
    previous_train_loss = None
    previous_test_loss = None

    for file_name in [file for file in os.listdir(get_current_dir_path(output_dir)) if file.endswith('.json')]:
        with open(f'{get_current_dir_path(output_dir)}/{file_name}') as json_file:
            data = json.load(json_file)
        
        L_results.append(data['results'])
        L_previous_rounds_best_features = data['results']['previous_rounds_best_features']

        # we determine which are exploration runs at current step based on length of best_features
        if len(L_previous_rounds_best_features) > max_length:
            max_length = len(L_previous_rounds_best_features)

    for result in L_results:
        # for each exploration run at current step we get tested feature and associated val loss, 
        # add them to list of tested features
        if len(result['previous_rounds_best_features']) == max_length:
            current_round_feature = result['current_round_feature_add']
            current_round_val_loss = result['current_round_val_loss']
            current_round_train_loss = result['current_round_train_loss']
            current_round_test_loss = result['current_round_test_loss']

            # Preventing duplicates
            if current_round_feature not in L_tested_features:
                L_tested_features.append((current_round_feature, current_round_val_loss, current_round_train_loss, 
                                          current_round_test_loss))

            L_best_features = result['previous_rounds_best_features']
            previous_val_loss = result['previous_rounds_val_loss']
            previous_train_loss = result['previous_rounds_train_loss']
            previous_test_loss = result['previous_rounds_test_loss']

    return L_tested_features, L_best_features, previous_val_loss, previous_train_loss, previous_test_loss


def get_min_loss(L_tested_features):
    """
    Gets feature that produced minimum validation loss from list of tested features
    
    Returns: 
        Best feature according to minimum validation loss
        Minimum validation loss, training and test losses
    """
    
    (min_loss_feature, val_loss, train_loss, test_loss) = sorted([feature for feature in L_tested_features], 
                                                                 key=lambda e: e[1])[0]
    return min_loss_feature, val_loss, train_loss, test_loss


def get_hparams_dict(cluster, learning_rate, batch_size, optimizer, model, hidden_size, nbr_layers, l1, l2):
    return {    "hyperparams" : {   "cluster": cluster,
                                    "learning_rate": learning_rate, 
                                    "batch_size": batch_size, 
                                    "optimizer": optimizer, 
                                    "model": model,
                                    "hidden_size": hidden_size,
                                    "nbr_layers": nbr_layers,
                                    "l1": l1,
                                    "l2": l2
                                }, 
                "results": {} }