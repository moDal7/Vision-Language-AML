from torch import nn
from models.base_model import DomainDisentangleModel
from time import gmtime, strftime
import torch
import torch.nn.functional as funct
import logging
import random
import numpy
import wandb

class EntropyLoss(nn.Module): # entropy loss as described in the paper 'Domain2Vec: Domain Embedding for Unsupervised Domain Adaptation', inherits from nn.Module and uses torch functions to preserve autograd
    def __init__(self):
        super().__init__()

    def forward(self, x):
        entropy = funct.log_softmax(x, dim=1)
        soft_sum = -1.0 * entropy.sum()/x.size(0)
        return soft_sum 
        
class DomainDisentangleExperiment: # See point 2. of the project
    
    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        self.time = strftime('%m-%d_%H:%M:%S', gmtime())
        # Initialize wandb
        wandb.init(
            entity="vision-and-language2023", 
            project="vision-and-language",
            tags=["domain_disentangle", opt['experiment'], opt['target_domain']],
            name=f"{opt['experiment']}_{opt['target_domain']}_{self.time}"
        )

        # initialize wandb config
        config = wandb.config
        config.backbone = "Resnet18"
        config.experiment = opt['experiment']
        config.target_domain = opt['target_domain']
        config.max_iterations = opt['max_iterations']
        config.batch_size = opt['batch_size']
        config.learning_rate = opt['lr']
        config.validate_every = opt['validate_every']
        config.clip_finetune = opt['clip_finetune']

        if (opt['weights']): #load weights from command line argument
            self.weights = torch.Tensor(opt['weights'])
            config.weights = self.weights

        else:
            self.weights = torch.tensor([4, 1, 0.01, 0.01, 2])
            config.weights = self.weights
        logging.info(f'INITIAL WEIGHTS : {self.weights}')
        logging.basicConfig(filename=f'training_logs/log.txt', format='%(message)s', level=logging.INFO, filemode='a')
        # weights explanation:
        # weights[0] = weight of category classifier
        # weights[1] = weight of domain classifier 
        # weights[2] = alpha of category entropy
        # weights[3] = alpha of domain entropy
        # weights[4] = weight of reconstructor loss
        # weights[5] = if present weight of clip

        if opt["determ"]:
            random.seed(0)
            numpy.random.seed(0)
            torch.manual_seed(0)
            torch.use_deterministic_algorithms(True)

        # Setup model
        self.model = DomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True
        wandb.watch(self.model, log="all")

        # Setup optimization procedure 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.loss_ce_cat = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_ce_dom = torch.nn.CrossEntropyLoss()
        self.loss_entropy = EntropyLoss()
        self.loss_MSE = torch.nn.MSELoss()

        # Validation loss
        self.criterion = torch.nn.CrossEntropyLoss()
        

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)
        wandb.save('model.pt')

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, data):
        
        x, y, dom = data
        x = x.to(self.device)
        y = y.to(self.device)
        dom = dom.to(self.device)


        logits = self.model(x, 4)
        loss_0 = self.weights[0] * self.loss_ce_cat(logits[1], y)
        loss_1 = self.weights[1] * self.loss_ce_dom(logits[3], dom)
        loss_2 = self.weights[2] * self.loss_entropy(logits[2])
        loss_3 = self.weights[3] * self.loss_entropy(logits[4])
        loss_4 = self.weights[4] * self.loss_MSE(logits[5], logits[0]) 

        loss_final = loss_0 + loss_1 + loss_2 + loss_3 + loss_4
        self.optimizer.zero_grad()
        loss_final.backward()
        self.optimizer.step()

        wandb.log({"loss_ce_cat": loss_0})
        wandb.log({"loss_ce_dom": loss_1})
        wandb.log({"loss_entropy_cat": loss_2})
        wandb.log({"loss_entropy_dom": loss_3})
        wandb.log({"loss_reconstructor": loss_4})
        wandb.log({"loss_final": loss_final})
  
        return loss_final.item()

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                _ = _.to(self.device)

                logits = self.model(x, 4)
                loss += self.criterion(logits[1], y)
                pred = torch.argmax(logits[1], dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss