from typing import Literal

import torch.utils.data as Data
import torch
import pytorch_lightning as pl
from tqdm import tqdm

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ProjectDataModule(pl.LightningDataModule):
    #Class Variables

    def __init__(self, dataset, batch_size=32, num_workers=6) -> None:
        super().__init__()

        self.dataset = dataset


        self.batch_size = batch_size
        self.num_workers = num_workers

        #Implement logic for train/test/validation splits

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        raise NotImplementedError('Not Implemented yet')

    def train_dataloader(self):
        return Data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return Data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return Data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

if __name__ == '__main__':
    '''
    Unit Testing
    '''
    import time

    TEST_DIR = None #directory of unit test samples

    X = None #Unit Test Dataset

    datamod = ProjectDataModule(X, batch_size=512)
    dataloader = datamod.train_dataloader()

    lim = len(dataloader)
    start = time.time()
    print("Testing....")
    labels = []
    for item, label in tqdm(dataloader, total=lim):
        '''
        Test running through all batches
        add more tests as necessary test size of batch elements etc.
        
        ex: assert test_batch.shape == (N, ...)
            assert label.shape == (N, ...)
        '''
        labels.append(label)
    end = time.time()

    labels = torch.cat(labels, axis=0).flatten()

    print(f'time per execution: {((len(dataloader)*((end - start)/lim)))/60:.2e}m')
