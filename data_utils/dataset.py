import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import numpy as np
import json
import os
import pandas as pd

from typing import Literal
from tqdm import tqdm


METADATA_PATH = None
AUDIO_EXT = '.npy' #Audio Files are pre-processed into 1-channel 16Khz numpy arrays
SR = 16000
IGNORE_INDEX = -1 #label to ignore during gradient computation, if necessary

class ProjectDataset(Dataset):
    #Instantiate class vars

    def __init__(self, *args, **kwargs):
        #Instance vars

        #Class Vars
         
        # Load data into a NumPy memory-mapped array
        self.data, self.indices = self._load_data()

        raise NotImplementedError('method not implemented')

    def __len__(self):
        raise NotImplementedError('method not implemented')
    
    def __getitem__(self, idx):
        raise NotImplementedError('method not implemented')
    
    def _zeropad(self, X, num):
        if len(X) >= num:
            return X
        return torch.cat([X, torch.zeros((num - X.shape[0],))])


    def _load_data(self):
        '''
        Assuming each file contains data that can be loaded into a NumPy array
        Modify this function based on your data loading logic
        '''
        file_list = None
        #file_list = [file_list[-1]] + file_list[1:-1] #to account for split_{}.npy
        data_list = [np.load(file_path, mmap_mode='r') for file_path in file_list]

        indices = np.cumsum([0] + [x.shape[0] for x in data_list])

        return (data_list, indices)
     
    @staticmethod
    def from_dirs(dirs: list[str], *args, **kwargs) -> bool:
        
        raise NotImplementedError('method not implemented')

if __name__ == '__main__':
    '''
    Unit Testing
    '''
    import time
    from random import randint

    TEST_DIR = None #Directory to load test samples
    X = None # Load Test dataset
    print('Num of Test Samples: ', len(X))

    lim = 100 #len(X)

    start = time.time()

    print("Testing....")
    for x in tqdm(range(lim)):
        '''
        Test running through <lim> samples in dataset
        add more tests as necessary test size of elements etc.
        
        ex: assert datapoint.shape == (N, ...)
            assert label.shape == (N, ...)
        '''
        continue
    end = time.time()

    print(f'time per execution: {((((end - start)/lim)))/60:.2e}m')


