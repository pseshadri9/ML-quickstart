import pytorch_lightning as pl
import torch
from .metrics import *
from tqdm import tqdm
from data_utils import DEVICE

from data_utils.dataset import IGNORE_INDEX

OUTPUT = 'output'
LOSS = 'loss'

class Model(pl.LightningModule): 
    def __init__(self, *args, **kwargs):
        '''
        args:
            fill in
        '''
        super(Model, self).__init__()
        self.save_hyperparameters()
          
        # Define model architecture
        self.params = None
        
        # Defining learning rate
        self.lr = None
          
        # Define base loss fn
        self._loss = None
            

        #Cache Validation outputs 
        self.outs = list()
    
    def activation(self, X):
        #Define activation fn

        return NotImplementedError('method not implemented')
    
    def loss(self, X, y):
        '''
        Loss fn wrapper for augmentations before passing for base loss fn
        
        '''
        return self._loss(X, y)


    def forward(self, x):
        """
        :param x: [batch_size, ... ]

        :return output: [batch_size, ....]
        """

        raise NotImplementedError('method not implemented')
    
    def predict(self, x):
        """
        :param x: [batch_size, ... ]

        :return output: [batch_size, ....]
        """
        raise NotImplementedError('method not implemented')
    
    def configure_optimizers(self):
        #return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-9)
        raise NotImplementedError('method not implemented')

    
    def common_step(self, batch):
        """
        common forward step for train, test, val steps
        """
        X, y = batch

        pred = self.forward(X)

        return {LOSS: self.loss(pred, y), OUTPUT: pred}

    def training_step(self, batch, batch_idx):
        
        model_output = self.common_step(batch)
        
        self.log('train_loss', model_output[LOSS].detach().cpu().item(), prog_bar=True, sync_dist=True)

        return model_output[LOSS]

    def validation_step(self, batch, batch_idx):
        model_output = self.common_step(batch)

        self.outs.append((self.activation(model_output[OUTPUT]), batch[1]))

        self.log('val_loss', model_output[LOSS].detach().cpu().item(), prog_bar=True, sync_dist=True)

        return model_output[LOSS]
    
    
    def test_step(self, batch, batch_idx):
        model_output = self.common_step(batch) 
        self.outs.append((self.activation(model_output[OUTPUT]), batch[1]))

        self.log('test_loss', 0, prog_bar=True, sync_dist=True)

        return model_output[LOSS]
    
    def on_validation_epoch_end(self):
        pred, targ, rep = self.common_eval(stage='val')
        return rep
    
    def on_test_epoch_end(self):
        pred_onset, targ_onset, rep = self.common_eval(stage='test')
        return rep
    
    def common_eval(self, stage='val'):
        pred, targ = self.collect_test_batches()
        if len(pred.shape) > 1:
            pred = pred.flatten()
            targ = targ.flatten()

        pred = pred[targ != IGNORE_INDEX].tolist()
        targ = targ[targ != IGNORE_INDEX].tolist()
        
        #Calculate metrics below
        rep = self.metric_report(targ, pred, output_dict=True)
        self.log_nested_dict(rep, stage=stage)
        self.outs.clear()

        return pred, targ, rep
        
    '''
    Helper Fn's below
    '''
    def log_nested_dict(self, X: dict, stage='val'):
        for k, v in X.items():
            if type(v) == dict:
                d = {'/'.join([k, k2]):v2 for k2, v2 in v.items()}
                self.log_nested_dict(d, stage=stage)
            else:
                self.log(f'{stage} {k}', v, prog_bar=True, sync_dist=True)

    def collect_test_batches(self):
        with torch.no_grad():
            X_total = None
            y_total = None
            for batch in self.outs:
                X, y = batch
                if X_total is None:
                    X_total = X.detach().cpu()
                    y_total = y.detach().cpu()
                X_total = torch.cat((X_total, X.detach().cpu()), axis=0)
                y_total = torch.cat((y_total, y.detach().cpu()), axis=0)
        
        return X_total, y_total