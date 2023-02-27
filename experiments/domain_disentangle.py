import torch
from torch import nn
from models.base_model import DomainDisentangleModel
import torch.functional as funct
import logging
import random
import numpy

class EntropyLoss(nn.Module): # entropy loss as described in the paper 'Domain2Vec: Domain Embedding for Unsupervised Domain Adaptation', inherits from nn.Module and uses torch functions to preserve autograd
    def __init__(self):
        super().__init__()

    def forward(self, x):
        entropy = funct.softmax(x, dim=1) * funct.log_softmax(x, dim=1)
        soft_sum = -1.0 * (entropy.sum()/x.size(0))
        return soft_sum 
        
class DomainDisentangleExperiment: # See point 2. of the project
    
    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        if (opt['weights']): #load weights from command line argument
            self.weights = torch.Tensor(opt['weights'])
        else:
            self.weights = torch.tensor([4, 1, 0.01, 0.01, 2])
        logging.info(f'INITIAL WEIGHTS : {self.weights}')
        logging.basicConfig(filename=f'training_logs/log.txt', format='%(message)s', level=logging.INFO, filemode='a')
        # weights explanation:
        # weights[0] = weight of category losses (category cross-entropy, category entropy)
        # weights[1] = weight of domain losses (domain cross-entropy, domain entropy)
        # weights[2] = weight of reconstructor loss
        # weights[3] = alpha of category entropy
        # weights[4] = alpha of domain entropy
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

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, data, debug = False, i = False):

        x, y, dom = data
        x = x.to(self.device)
        y = y.to(self.device)
        dom = dom.to(self.device)

        if ( debug and i%500 == 0 ):
            logging.info(f'[TRAIN - iteration {i}] ')

        # #step 0     #TODO: only step 4 for now
        # logits = self.model(x, 0) 
        # loss_0 = self.loss_ce(logits, y)

        # self.optimizer.zero_grad()
        # loss_0.backward()
        # self.optimizer.step()
        
        # if ( debug and i%500 == 0 ):
        #     logging.info(f'[TRAIN - iteration {i}] logits size step 0 : {logits.size()}')
        #     logging.info(f'[TRAIN - iteration {i}] logits step 0 : {logits}')
        #     logging.info(f'[TRAIN - iteration {i}] loss_0 : {loss_0}')

        # #step 1
        # logits = self.model(x, 1) 
        # loss_1 = self.loss_ce(logits, dom)

        # self.optimizer.zero_grad()
        # loss_1.backward()
        # self.optimizer.step()

        # if ( debug and i%500 == 0 ):
        #     logging.info(f'[TRAIN - iteration {i}] logits size step 1 : {logits.size()}')
        #     logging.info(f'[TRAIN - iteration {i}] logits step 1 : {logits}')
        #     logging.info(f'[TRAIN - iteration {i}] loss_1 : {loss_1}')
        
        # #step 2
        # #freezing layers for the adversarial stepe of the training
        # for param in self.model.category_encoder.parameters():
        #     param.requires_grad = False
        # for param in self.model.category_classifier.parameters():
        #     param.requires_grad = False
        # for param in self.model.domain_encoder.parameters():
        #     param.requires_grad = False
        # for param in self.model.domain_classifier.parameters():
        #     param.requires_grad = False
        # self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opt['lr'])
        # logits = self.model(x, 2) 
        # loss_2 = self.loss_entropy(smax(logits))

        # self.optimizer.zero_grad()
        # loss_2.backward()
        # self.optimizer.step()

        # if ( debug and i%500 == 0 ):
        #     logging.info(f'[TRAIN - iteration {i}] logits step 2 size : {logits.size()}')
        #     logging.info(f'[TRAIN - iteration {i}] logits step 2 : {logits}')
        #     logging.info(f'[TRAIN - iteration {i}] loss_2 : {loss_2}')

        # #step 3
        # logits = self.model(x, 3) 
        # loss_3 = self.loss_entropy(smax(logits))

        # self.optimizer.zero_grad()
        # loss_3.backward()
        # self.optimizer.step()

        # if ( i%500 == 0 and debug):
        #     logging.info(f'[TRAIN - iteration {i}] logits step 3 size : {logits.size()}')
        #     logging.info(f'[TRAIN - iteration {i}] logits step 3 : {logits}')
        #     logging.info(f'[TRAIN - iteration {i}] loss_3 : {loss_3}')

        #step 4
        #for param in self.model.category_encoder.parameters():
        #    param.requires_grad = True
        #for param in self.model.category_classifier.parameters():
        #    param.requires_grad = True
        #for param in self.model.domain_encoder.parameters():
        #    param.requires_grad = True
        #for param in self.model.domain_classifier.parameters():
        #    param.requires_grad = True

        logits = self.model(x, 4)
        loss_0 = self.loss_ce_cat(logits[1], y)
        loss_1 = self.loss_ce_dom(logits[3], dom)
        loss_2 = self.loss_entropy(logits[2])
        loss_3 = self.loss_entropy(logits[4])
        loss_4 = self.loss_MSE(logits[5], logits[0]) 

        loss_final = self.weights[0] * (loss_0 + self.weights[3] * loss_2) + self.weights[1] * (loss_1 + self.weights[4] * loss_3) + self.weights[2] * loss_4
        self.optimizer.zero_grad()
        loss_final.backward()
        self.optimizer.step()

        if ( debug and i%500 == 0 ):
            logging.info(f'[TRAIN - iteration {i}] logits size step 4 : ')
            for j in range(6):
                logging.info(f'logits[{j}]: {logits[j].size()}')
            logging.info(f'[TRAIN - iteration {i}] logits step 4 : {logits}')
            logging.info(f'[TRAIN - iteration {i}] loss_0 : {loss_0}')
            logging.info(f'[TRAIN - iteration {i}] loss_1 : {loss_1}')
            logging.info(f'[TRAIN - iteration {i}] loss_2 : {loss_2}')
            logging.info(f'[TRAIN - iteration {i}] loss_3 : {loss_3}')
            logging.info(f'[TRAIN - iteration {i}] loss_4 : {loss_4}')
            logging.info(f'[TRAIN - iteration {i}] loss_final : {loss_final}')
                
        return loss_final.item()

    def validate(self, loader, debug = False, i=3):
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

                if ( debug and i%500 == 0 ):
                    logging.info(f'[VALIDATION - iteration {i}] ')
                    for elem in logits:
                      logging.info(f'[VALIDATION - iteration {i}] logits size : {elem.size()}')
                      logging.info(f'[VALIDATION - iteration {i}] logits : {elem}')
                    logging.info(f'[VALIDATION - iteration {i}] loss : {loss}')

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss