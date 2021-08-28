from utils.base_trainer import BaseTrainer

from model.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np

class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(0.0, device=self.device,requires_grad = True)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device,requires_grad = True) if c is not None else None
        self.nu = nu
        self.init_flag=0
        # Optimization parameters
        self.warm_up_n_epochs = 4  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self,trainloader, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)
        net.summary()
        #summary(net,input_size=[32],batch_size=1)
        #print(net)
        # Get train data loader
        train_loader = trainloader

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')
        # Set loss
        criterion = torch.nn.MSELoss()
        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        
        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            #scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            idx=0
            for data in train_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.reshape(-1).to(self.device)
                # Zero the network parameter gradients
                optimizer.zero_grad()
                #print(inputs[60][0:10])
                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - net.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - net.R** 2
                    scores = -1*torch.tanh(scores)
                    #loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                    #print(scores.shape,'labels_shape',labels.shape)
                    loss = criterion(labels,scores)
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()
                idx = idx+1
                #if idx%20 == 0:
                    #print('Network OUtput',outputs[0])
                    #print('True is ',labels[0], 'While process', scores[0])
                    #print('c is ',net.c, 'R', net.R)
                # Update hypersphere radius R on mini-batch distances
                # if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                #     self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def test(self,testloader, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)
        criterion = torch.nn.MSELoss()
        # Get test data loader
        test_loader = testloader

        # Testing
        logger.info('Starting testing...')
        #logger.info('Hyper param: R- %.3f,c-%.3f' % (self.R, self.c))
        #print(net.R, 'hyper',net.c)
        start_time = time.time()
        idx_label_score = []
        net.eval()
        idx=0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.reshape(-1).to(self.device)
                outputs = net(inputs)
                dist = torch.sum((outputs - net.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - net.R**2
                    scores = -1*torch.tanh(scores)
                    loss = criterion(labels,scores)
                else:
                    scores = dist
                #predict = -1*torch.tanh(scores)
                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))
                idx= idx+1
        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        #predict = np.array(predict)
        scores = np.array(scores)
        #print(labels[0:10])
        self.test_auc = roc_auc_score(labels, scores)
        self.test_acc = sum(np.sign(scores)==labels)/len(labels)
        logger.info('Radius and c: {:.2f},{}'.format(net.R.cpu().data, net.c.cpu().data))
        logger.info('Test set ROC_AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test set ACC: {:.2f}%'.format(100. * self.test_acc))
        logger.info('Finished testing.')

    
    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros((1,net.rep_dim), device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                #看论文，他为什么要用两个一样的网络，我不理解
                c += torch.sum(outputs, dim=0)
        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        #c = torch.nn.Parameter(c)
        c = torch.zeros((1,net.rep_dim), device=self.device,requires_grad=True)
        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
