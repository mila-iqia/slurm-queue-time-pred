import os
import pandas as pd
import csv
import numpy as np
import dateutil.parser as parser

path_to_fake_jobs = os.path.join(os.path.dirname(__file__), './sbatch_estimations')
path_to_csv = os.path.join(os.path.dirname(__file__), './true_start_time_vs_slurm_estimate.csv')

def get_true_and_estimate_start_times():
    """
    From text files with raw outputs from sbatch command, retrieves SLURM estimates and actual start time of fake jobs.
    
    Returns:
        True and estimate start times for fake jobs
    """
    
    L_true_and_estimate_start_times = []
    
    for file_name in [file for file in os.listdir(path_to_fake_jobs) if file.endswith('.txt')]:
        with open(f'{path_to_fake_jobs}/{file_name}') as txt_file:
            lines = txt_file.readlines()
            
        if 'Actual start time:\n' in lines:
            
            # Get job unique id from filename
            fake_job_id = file_name.split('.')[0].split('_')[0]
            
            # Get job configuration
            fake_job_config = eval(lines[0].split(" ", 2)[-1].strip())
            
            # Get actual start time of job
            true_time = parser.parse(lines[-1].strip()).replace(tzinfo=None)
            
            # get estimate for line that starts with sbatch, estimate is at 6th index
            estimate_time = parser.parse([line for line in lines if line.startswith('sbatch: Job')][0].split()[6])
            
            L_true_and_estimate_start_times.append((fake_job_id, fake_job_config, true_time, estimate_time))
            
    return L_true_and_estimate_start_times
            
            
def compare():
    """For each fake job, computes difference between estimate and actual start time while saving job configuration.""" 
    L_true_and_estimate_start_times = get_true_and_estimate_start_times()
    L_fake_job_data = []
    L_fake_job_features = []
   
    for (job_id, job_config, true, estimate) in L_true_and_estimate_start_times:
        
        # Add true and estimate to dictionary
        job_config['true_start_time'] = true
        job_config['estimate_start_time'] = estimate 
        
        # Get difference between true and estimate start time in seconds
        diff_in_s = abs(true - estimate).total_seconds()
        job_config['diff_in_s'] = diff_in_s
        
        # Add unique id at beginning of dictionary
        job_config = {"unique_id": job_id, **job_config}
        
        L_fake_job_data.append(job_config)
    
    L_fake_job_features = list(L_fake_job_data[0].keys())  
    L_fake_job_data = sorted(L_fake_job_data, key=lambda e: e['diff_in_s'], reverse=True)
    
    write_to_csv(L_fake_job_features, L_fake_job_data)
    compute_MSE()
    
    
def write_to_csv(fieldnames, rows):
    """Writes fake jobs' id, configuration and difference between estimate and actual start time in a csv file.""" 
    with open(f'{path_to_csv}', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

      
def compute_MSE():
    """
    Computes MSE for SLURM estimates for all fake jobs in true_start_time_vs_slurm_estimate.csv file and prints it.
    Difference between estimates and actual values is on a log10 scale because wait time predictions are also.
    """
    df = pd.read_csv(f'{path_to_csv}', index_col=0, header=0)
    diff = np.array(np.log10(df['diff_in_s'])).mean()
    print('Differences between SLURM predictions and real start times: ', "{:.4f}".format(diff))


if __name__ == "__main__":
    compare()
 
    
"""
python3 compare_estimate_to_true_start_time.py

"""
