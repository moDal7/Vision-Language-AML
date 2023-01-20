import torch
from torch import nn
from models.base_model import DomainDisentangleModel


class EntropyLoss(nn.Module): # entropy loss as described in the paper 'Domain2Vec: Domain Embedding for Unsupervised Domain Adaptation', inherits from nn.Module and uses torch functions to preserve autograd
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return -torch.sum(torch.sum(torch.log(x), 0)/x.shape[0])

class DomainDisentangleExperiment: # See point 2. of the project
    
    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')
        self.weights = torch.ones(3)

        # Setup model
        self.model = DomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.loss_ce = torch.nn.CrossEntropyLoss()
        self.loss_entropy = EntropyLoss()
        self.loss_MSE = torch.nn.MSELoss()
        

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, data):
        raise NotImplementedError('[TODO] Implement DomainDisentangleExperiment.')

    def validate(self, loader):
        raise NotImplementedError('[TODO] Implement DomainDisentangleExperiment.')