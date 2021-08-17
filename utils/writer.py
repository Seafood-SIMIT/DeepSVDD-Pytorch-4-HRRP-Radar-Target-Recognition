import numpy as np
from tensorboardX import SummaryWriter

from .plotting import plot_spectrogram_to_numpy


class MyWriter(SummaryWriter):
    def __init__(self, hp, logdir):
        super(MyWriter, self).__init__(logdir)
        self.hp = hp

    def log_training(self, train_loss_sum,  
                     train_loss_ever, 
                     train_acc, 
                     train_step):
        
        self.add_scalar('train/train_accury', train_acc ,train_step)
        
        self.add_scalar('train/train_loss_sum', train_loss_sum,train_step)
        
        self.add_scalar('train/train_everloss', train_loss_ever,train_step)
        
        
    def log_evaluation(self, acc, test_loss_ever, 
                       test_loss_sum, 
                       step):

        self.add_scalar('test/acc', acc , step)
        
        self.add_scalar('test/test_loss_ever', test_loss_ever, step)

        self.add_scalar('test/test_loss_sum', test_loss_sum, step)
        
        