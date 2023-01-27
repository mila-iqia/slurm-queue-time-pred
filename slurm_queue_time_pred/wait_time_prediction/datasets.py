
import numpy as np
import math
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
import glob
import os
import pickle5 as pickle
from slurm_queue_time_pred.wait_time_prediction.util import normalize, count_files_in_dir

# Change this to your chosen path to data
path_to_data = os.path.join(os.path.dirname(__file__), '../../data/data_on_submit_days')

# First test day must be at least 3, so there is data for validation and training
FIRST_TEST_DAY = 3

# Date features
date_features = ['submit_time', 'eligible_time', 'poll_time']


def get_total_days(cluster : str):
    return count_files_in_dir(os.path.join(path_to_data, cluster))
    
    
def csv_to_pickle(test_day : None, cluster : str):
    """Loads data from csv files into pickle files according to days for training, validation and testing."""
    
    files = sorted(glob.glob(f"{path_to_data}/{cluster}/*.csv"))
    
    rng = np.random.RandomState(seed=46)
    
    if not test_day:
        
        # Split data into train, valid and test sets
        # Test set: 10% of data, randomly sampled, with seed
        test_files = rng.choice(files, size=math.ceil(0.1*len(files)), replace=False)
        
        # Valid set: 10% of data, randomly sampled, with seed
        # Sort files to ensure that the same files are selected for valid and test sets
        valid_files = rng.choice(sorted(list(set(files) - set(test_files))), size=math.ceil(0.1*len(files)), replace=False)
        
        # Train set: remaining data
        train_files = list(set(files) - set(test_files) - set(valid_files))
    else:
        assert test_day <= get_total_days(cluster) and test_day >= FIRST_TEST_DAY

        # Valid set: 20% of days from beginning to test day
        valid_files = rng.choice(files[:test_day - 1], size=math.ceil(0.2*len(files[:test_day - 1])), replace=False)    
        
        # Train set: rest of days from beginning to test day
        train_files = list(set(files[:test_day - 1]) - set(valid_files))
        
        # Test files: day after test day
        test_files = [files[test_day - 1]]
    
    # Assert that train, valid and test sets don't have any overlap
    assert len(set(train_files).intersection(set(valid_files))) == 0
    assert len(set(train_files).intersection(set(test_files))) == 0
    assert len(set(valid_files).intersection(set(test_files))) == 0
    
    for (segment, files) in [('train', train_files), ('valid', valid_files), ('test', test_files)]:
        li = []
        
        for filename in files:
            df = pd.read_csv(filename, index_col=0, header=0)
            li.append(df)
        
        df = pd.concat(li, axis=0, ignore_index=True)
        df = df.replace([np.inf, -np.inf], np.nan, inplace=False).dropna()

        # Drop all features that are part of "ground truth"
        df = df.drop(['start_time', 'time_limit', 'time_limit_sec', 'req_tres', 'submit_wait', 'submit_wait_sec', 
            'eligible_wait', 'eligible_wait_sec', 'run_time', 'run_time_sec', 'poll_wait', 'q_core_run_hours', 
            'q_p_core_run_hours', 'q_a_core_run_hours','q_p_a_core_run_hours', 'end_time', 'job_augment_id', 
            'job_summary_id', 'state', 'priority_x'], inplace=False, axis=1, errors='ignore')

        # Feature to predict
        y_df = df[['poll_wait_sec']]
        X_df = df.drop('poll_wait_sec', inplace=False, axis=1)
        # X_df = df
            
        # Create pickle files
        if segment == 'train':
            y_df.to_pickle(f"{path_to_data}/y_train_{cluster}_{test_day}.pickle")
            X_df.to_pickle(f"{path_to_data}/X_train_{cluster}_{test_day}.pickle")
        elif segment == 'valid':
            y_df.to_pickle(f"{path_to_data}/y_valid_{cluster}_{test_day}.pickle")
            X_df.to_pickle(f"{path_to_data}/X_valid_{cluster}_{test_day}.pickle")
        elif segment == 'test':
            y_df.to_pickle(f"{path_to_data}/y_test_{cluster}_{test_day}.pickle")
            X_df.to_pickle(f"{path_to_data}/X_test_{cluster}_{test_day}.pickle")
            

def create_datasets(X_train, y_train, X_valid, y_valid, X_test, y_test, 
                    features : list, batch_size : int, mix_train_valid : bool):
    """Generates dataloaders from train, valid and test sets.
    
    Returns:
        Train, valid and test loaders
    """
    
    if mix_train_valid:
        # Mix train and valid data into one set
        X_train_and_valid = pd.concat([X_train, X_valid], axis=0, ignore_index=True)
        y_train_and_valid = pd.concat([y_train, y_valid], axis=0, ignore_index=True)
        
        train_and_valid_set = SLURMDataset(X_train_and_valid, y_train_and_valid, features)
        
        train_size = int(0.8 * len(train_and_valid_set))
        valid_size = len(train_and_valid_set) - train_size

        train_set, valid_set = random_split(train_and_valid_set, [train_size, valid_size], 
                                            generator=torch.Generator().manual_seed(46))
    else:   
        # No mix
        train_set = SLURMDataset(X_train, y_train, features)
        valid_set = SLURMDataset(X_valid, y_valid, features, mus=train_set.mus, sigmas=train_set.sigmas)
    
    test_set = SLURMDataset(X_test, y_test, features, mus=train_set.mus, sigmas=train_set.sigmas)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    return train_loader, valid_loader, test_loader


def load_data(test_day : int, cluster : str):
    """Tries to load data from pickle file if file is available. If not, creates pickle file from csv.
    
    Returns:
        Train, valid and test sets
    """
    # Check if pickle file exists
    if not os.path.isfile(f"{path_to_data}/X_train_{cluster}_{test_day}.pickle"):
        print(f"Pickle files not found. Creating pickle files...")
        csv_to_pickle(test_day, cluster)

    with open(f"{path_to_data}/X_train_{cluster}_{test_day}.pickle", "rb") as db:
        X_train = pickle.load(db)
    with open(f"{path_to_data}/y_train_{cluster}_{test_day}.pickle", "rb") as db:
        y_train = pickle.load(db)
    with open(f"{path_to_data}/X_valid_{cluster}_{test_day}.pickle", "rb") as db:
        X_valid = pickle.load(db)
    with open(f"{path_to_data}/y_valid_{cluster}_{test_day}.pickle", "rb") as db:
        y_valid = pickle.load(db)
    with open(f"{path_to_data}/X_test_{cluster}_{test_day}.pickle", "rb") as db:
        X_test = pickle.load(db)
    with open(f"{path_to_data}/y_test_{cluster}_{test_day}.pickle", "rb") as db:
        y_test = pickle.load(db)
                                    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


class SLURMDataset(Dataset):
    """Custom Dataset class."""
    
    def __init__(self, X, y, features, mus=None, sigmas=None, transform=None):
        self.X = X.loc[:, features] # extract only selected features
        self.y = np.log10(y.values)
        self.features = features
        self.transform = transform
        self.date_features_idx = []
        self.mus = mus
        self.sigmas = sigmas
        
        # If feature in features is a date, convert to seconds since 1970
        for idx, feature in enumerate(self.features):
            if feature in date_features:
                self.X[feature] = pd.to_datetime(self.X[feature]).apply(lambda x: x.timestamp())
                
                # Mark as date feature for future denormalization if needed
                self.date_features_idx.append(idx)
        
        # Normalize features  
        self.mus, self.sigmas, self.X = normalize(self.X.values, self.mus, self.sigmas)
        
        # Convert to tensors
        self.X = torch.tensor(self.X).float()
        self.y = torch.tensor(self.y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
           
        return self.X[idx], self.y[idx]
    