from code.explore_features.explore import get_results, get_deterioration_metrics
from code.explore_features.explore_util import get_current_dir_path
from code.explore_features import run_experiment # leave here
import json
import matplotlib.pyplot as plt

from code.wait_time_prediction import configlib
from code.wait_time_prediction.configlib import config as C


parser = configlib.add_parser("Report exploration results config")
parser.add_argument('--synthetic_data', default=False, action='store_true',
                    help='Is data used for feature exploration synthetic or not.')


def select_baselines(is_data_synthetic : bool, cluster : str, model : str):
    """
    Select baseline according to model used, cluster and whether data is synthetic or not.
    
    Returns:
        Baseline when training model with no features
        Baseline when training model with all features
    """ 
    
    if is_data_synthetic:
        if cluster == 'cedar':
            # valid, train, test
            L_no_feature_baseline = [ 0.2611, 0.2603, 0.2305 ]
            if model == 'linear':
                L_all_feature_baseline = [ 0.0658, 0.0637, 0.0563 ]
            elif model == 'NN':
                L_all_feature_baseline = [ 0.0095, 0.0520, 0.0051 ]
        if cluster == 'graham':
            L_no_feature_baseline = [ 0.2343, 0.2424, 0.2592 ]
            if model == 'linear':
                L_all_feature_baseline = [ 0.0555, 0.0597, 0.0732 ]
            elif model == 'NN':
                L_all_feature_baseline = [ 0.0356, 0.0476, 0.0531 ]
    else:
        if cluster == 'cedar':
            L_no_feature_baseline = [ 0.6548, 0.6733, 0.5889 ]
            if model == 'linear':
                L_all_feature_baseline = [ 0.8315, 0.4046, 0.4224 ]
            elif model == 'NN':
                L_all_feature_baseline = [ 0.2483, 0.1979, 0.3107 ]
        if cluster == 'graham':
            L_no_feature_baseline = [ 0.7567, 0.5815, 0.4629 ]
            if model == 'linear':
                L_all_feature_baseline = [ 0.5560, 0.3540, 0.4335 ]
            elif model == 'NN':
                L_all_feature_baseline = [ 0.3301, 0.1867, 0.4290 ]
    
    return L_no_feature_baseline, L_all_feature_baseline


def report_results():
    """
    Creates a report with the results at each step of training the model with best (top) features.
    Each step means adding next top feature to previous ones when training the model (accumulation of features).
    Saves report at location results/explore_features/reports.
    """
    
    nbr_top_features = C['nbr_top_features']
    output_dir = C['output_dir']
    
    L_results_at_each_step, D_hparams = get_results(nbr_top_features, output_dir)
    
    # 1. Create dictionary of losses at each step + frequency and mean of loss deterioration
    DDD_deter_metrics = get_deterioration_metrics(nbr_top_features, output_dir)
    DD_report = {}
    DD_report['validation_loss_at_each_step'] = {}
    DD_report['training_loss_at_each_step'] = {}
    DD_report['test_loss_at_each_step'] = {}
    for result in L_results_at_each_step:
        DD_report['hparams'] = D_hparams
        DD_report['validation_loss_at_each_step'][' '.join(result[0])] = result[1]
        DD_report['training_loss_at_each_step'][' '.join(result[0])] = result[2]
        DD_report['test_loss_at_each_step'][' '.join(result[0])] = result[3]
    DD_report['deterioration_stats_at_each_step'] = DDD_deter_metrics

    path_to_report = f"{get_current_dir_path(output_dir)}/reports/{nbr_top_features}_features_report.json"
    
    with open(path_to_report, 'w') as f:
        json.dump(DD_report, f, sort_keys=True, indent=4)
        print(f'New file at {get_current_dir_path(output_dir)}/reports')
        
    # 2. Create plots for valid, train and test loss progression
    create_plots(nbr_top_features, output_dir, DD_report)
        
        
def create_plots(nbr_top_features : int, output_dir : str, DD_report):
    """
    Creates 3 plots : validation loss at each step, training loss at each step and test loss at each step.
    Saves them at location results/explore_features/plots.
    """
    
    L_no_feature_baseline, L_all_feature_baseline = select_baselines(C['synthetic_data'], 
                                                                     DD_report['hparams']['cluster'], 
                                                                     DD_report['hparams']['model'])

    plt_count = 0
    for segment in ( 'validation', 'training', 'test' ):
        segment_key = segment + "_loss_at_each_step"

        # First data point is using no features
        L_domain = [ 0 ]
        L_domain_features = [ 'none' ]
        L_image = [ L_no_feature_baseline[plt_count] ]
        
        # We get loss for each segment at each step of adding next top feature
        for (k, v) in sorted(DD_report[segment_key].items(), key=lambda e: len(e[0])):
            L_domain.append( len(k.split(" ")) )
            L_domain_features.append( k.split(" ")[-1] )
            L_image.append( v )

        fig = plt.figure()
        fig.patch.set_alpha(0)

        plt.plot(L_domain, L_image, linewidth=2)
        plt.title(f'{segment.title()} loss obtained by training model\nwith cumulative set of {nbr_top_features} best features')
        plt.axhline(y=L_all_feature_baseline[plt_count], label="all features", color='green', linestyle='--', linewidth=2)
        plt.text(0,L_all_feature_baseline[plt_count]+0.01, L_all_feature_baseline[plt_count], color='green')
        plt.xlabel('Accumulated features')
        plt.ylabel(f'{segment.title()} loss')
        plt.xticks(L_domain, L_domain_features, horizontalalignment='right', rotation=45)
        plt.tight_layout()
        plt.legend()

        fig.savefig(f"{get_current_dir_path(output_dir)}/plots/{nbr_top_features}_features_plot_{segment}.png")
        plt_count += 1
        
    print(f'New files at {get_current_dir_path(output_dir)}/plots')


if __name__ == "__main__":
    configlib.parse(save_fname="last_arguments.txt") # --> ligne importante
    print("Running with configuration:")
    configlib.print_config()
    
    report_results()
    
"""
python3 report_results.py --output_dir=7NN --synthetic_data

"""