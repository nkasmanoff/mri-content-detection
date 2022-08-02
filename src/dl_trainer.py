from .models.models import BrainClassifier, BasicBlock
from src.models.loss import contrast_loss_fn, masked_orientation_loss_fn
from .dat.mri_dataloader import create_datasets
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from test_tube import HyperOptArgumentParser
from argparse import ArgumentParser 
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report



class Autolabel(pl.LightningModule):
    """MRI AutoLabel model used to identiy an image's contrast and orientation.
    """
    def __init__(self, hparams=None):
        super(Autolabel,self).__init__()
        self.__check_hparams(hparams)
        self.hparams = hparams
        self.prepare_data()

        self.contrast_codes = pd.read_csv('../helpers/contrasts_OHE_codes.csv',index_col=0) 
        self.orientation_codes = pd.read_csv('../helpers/orientation_OHE_codes.csv',index_col=0)
        self.unlabeled_orientation = self.orientation_codes.iloc[self.orientation_codes.index == 'UNLABELED'].values[0].argmax()
        self._model = BrainClassifier(BasicBlock,in_ch=1,ch=self.n_ch,nblocks=2,n_contrasts = self.contrast_codes.shape[1],n_orientations = self.orientation_codes.shape[1])

    def forward(self,x):
        x_contrast, x_orientation = self._model(x)# returns the predicted mass of the galaxy at the center of this cube.
        return x_contrast, x_orientation

    def _run_step(self, batch, batch_idx,step_name):

        img, contrast_true, orientation_true, _ , _  = batch
        contrast_pred, orientation_pred  = self(img)
        contrast_loss = contrast_loss_fn(contrast_pred, torch.max(contrast_true, 1)[1],weight=self.contrast_weights)
        orientation_loss = masked_orientation_loss_fn(orientation_pred,orientation_true,unlabeled = self.unlabeled_orientation,weight=self.orientation_weights)
        loss = contrast_loss + orientation_loss 

        return loss, contrast_true,contrast_pred, orientation_true, orientation_pred


    def training_step(self, batch, batch_idx):
        """
        Log the loss for the training set.
        """
        train_loss, _, _, _, _, _ = self._run_step(batch, batch_idx, step_name='train')
        train_tensorboard_logs = {'train_loss': train_loss}
        return {'loss': train_loss, 'log': train_tensorboard_logs}


    def validation_step(self, batch, batch_idx):
        """
        Log the loss, accuracy, precision, recall, f1 score, and confusion matrix for the validation set.
        """
        val_log_dict = {}
        val_loss, contrast_true,contrast_pred, orientation_true, orientation_pred, fatsat_true = self._run_step(batch, batch_idx, step_name='valid')
        y_pred = contrast_pred.argmax(dim=1).detach().cpu()
        y_true = contrast_true.argmax(dim=1).detach().cpu()
        val_contrast_acc = torch.from_numpy(np.array([accuracy_score(y_pred,y_true)]))
        val_log_dict['val_loss'] = val_loss
        val_log_dict['val_contrast_acc'] = val_contrast_acc
        report = classification_report(y_pred, y_true,output_dict=True)
        for key in range(len(self.contrast_codes)):        
            key = str(key)
            contrast = self.contrast_codes.index[self.contrast_codes.values.argmax(axis=1) == int(key)].values[0]
            if key in report.keys():
                if report[key]['support'] > 0:
                    contrast_recall = report[key]['recall']
                    contrast_precision = report[key]['precision']
                    contrast_f1score = report[key]['f1-score']
                else:    
                    contrast_recall = np.nan #none of this contrast were in the batch. ignore! 
                    contrast_precision = np.nan
                    contrast_f1score = np.nan
            else:
                contrast_recall = np.nan
                contrast_precision = np.nan
                contrast_f1score = np.nan

            val_log_dict[contrast + '_recall'] = torch.from_numpy(np.array(contrast_recall))        
            val_log_dict[contrast + '_precision'] = torch.from_numpy(np.array(contrast_precision))        
            val_log_dict[contrast + '_f1score'] = torch.from_numpy(np.array(contrast_f1score))        
        
        return val_log_dict 
        
    def validation_epoch_end(self, outputs):
        pass 
    def test_step(self, batch, batch_idx):
        pass
    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer =  torch.optim.Adam(self.parameters(), lr = self.learning_rate ,weight_decay = self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = 4)
        return [optimizer], [scheduler] 
    
    def prepare_data(self):
        # the dataloaders are run batch by batch where this is run fully and once before beginning training
        self.train_loader, self.valid_loader, self.test_loader = create_datasets(batch_size=self.batch_size, weighted_trainer = self.weighted_trainer, batch = self.batch, seed = self.seed)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader

    def __check_hparams(self, hparams):
        self.learning_rate = hparams.learning_rate if hasattr(hparams, 'learning_rate') else 0.001
        self.weight_decay = hparams.weight_decay if hasattr(hparams, 'weight_decay') else 0.
        self.batch_size = hparams.batch_size if hasattr(hparams, 'batch_size') else 4
        self.subsample = hparams.subsample if hasattr(hparams, 'subsample') else True
        self.seed = hparams.seed if hasattr(hparams, 'seed') else 32
        self.n_ch = hparams.n_ch if hasattr(hparams, 'n_ch') else 1
        self.weighted_trainer = hparams.weighted_trainer if hasattr(hparams, 'weighted_trainer') else False
        self.batch = hparams.batch if hasattr(hparams, 'batch') else 'merged'
        self.auto_lr_find = hparams.auto_lr_find if hasattr(hparams, 'auto_lr_find') else True


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(parents=[parent_parser], add_help=False)
        parser.opt_list('--learning_rate', type=float, default=3e-4, options=[.001,.0001,.00001], tunable=False)
        parser.opt_list('--weight_decay', type = float,default = 3e-4, options = [.0005,.001,.0001],tunable = False)
        parser.opt_list('--n_ch', type = int, default = 128, options = [16,32,64],tunable = False)
        # fixed parameters       
        parser.opt_list('--auto_lr_find', type=bool, default = True)
        parser.add_argument('--batch_size', type=int, default=12) 
        parser.add_argument('--seed', type=int, default = 42)
        parser.add_argument('--subsample',type=bool, default = True)
        parser.add_argument('--weighted_trainer',type=bool, default = False)
        parser.add_argument('--batch',type=str, default = 'merged') 
        return parser

     
 
if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = Autolabel.add_model_specific_args(parser)
    args = parser.parse_args()
    model = Autolabel(args)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)
