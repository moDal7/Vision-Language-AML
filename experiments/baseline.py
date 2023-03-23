import torch
import wandb
from time import gmtime, strftime
from models.base_model import BaselineModel

class BaselineExperiment: # See point 1. of the project
    
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

        # Setup model
        self.model = BaselineModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True
        wandb.watch(self.model, log="all")

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
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
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.model(x)
        loss = self.criterion(logits, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        wandb.log({"loss": loss})
        return loss.item()

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss += self.criterion(logits, y)
                pred = torch.argmax(logits, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss