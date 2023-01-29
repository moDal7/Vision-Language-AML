import torch
from torch import nn
from models.base_model import ClipDisentangleModel
import clip

# Load CLIP model and freeze it
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

clip_model, _ = clip.load('ViT-B/32', device='cpu') # load it first to CPU to ensure you're using fp32 precision.
clip_model = clip_model.to(device)

class EntropyLoss(nn.Module): # entropy loss as described in the paper 'Domain2Vec: Domain Embedding for Unsupervised Domain Adaptation', inherits from nn.Module and uses torch functions to preserve autograd
    def __init__(self):
        super().__init__()

    def forward(self, x):
        softmax_batch = -torch.sum(torch.sum(torch.log(x), 0)/x.shape[0])
        
class CLIPDisentangleExperiment: # See point 4. of the project

    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')
        self.weights = torch.tensor([1, 1, 0.5, 0.2, 0.2, 1])

        # Setup model
        self.model = ClipDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.loss_ce = torch.nn.CrossEntropyLoss()
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
    
    def train_iteration(self, data):
        x, y, dom, desc = data
        x = x.to(self.device)
        y = y.to(self.device)
        dom = dom.to(self.device)
        desc = desc.to(self.device)
        smax = nn.Softmax(dim=1)
        
        # TODO TEST CLIP TEST NEEDED
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False

        tokenized_text = clip.tokenize(desc).to(device)
        text_features = clip_model.encode_text(tokenized_text)

        #step 0
        logits = self.model(x, 0) 
        loss_0 = self.loss_ce(logits, y)

        self.optimizer.zero_grad()
        loss_0.backward()
        self.optimizer.step()

        #step 1
        logits = self.model(x, 1) 
        loss_1 = self.loss_ce(logits, dom)

        self.optimizer.zero_grad()
        loss_1.backward()
        self.optimizer.step()
        
        #step 2
        #freezing layers for the adversarial stepe of the training
        for param in self.model.category_encoder.parameters():
            param.requires_grad = False
        for param in self.model.category_classifier.parameters():
            param.requires_grad = False
        for param in self.model.domain_encoder.parameters():
            param.requires_grad = False
        for param in self.model.domain_classifier.parameters():
            param.requires_grad = False
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opt['lr'])
        logits = self.model(x, 2) 
        loss_2 = self.loss_entropy(smax(logits))

        self.optimizer.zero_grad()
        loss_2.backward()
        self.optimizer.step()

        #step 3
        logits = self.model(x, 3) 
        loss_3 = self.loss_entropy(smax(logits))

        self.optimizer.zero_grad()
        loss_3.backward()
        self.optimizer.step()

        #step 4
        for param in self.model.category_encoder.parameters():
            param.requires_grad = True
        for param in self.model.category_classifier.parameters():
            param.requires_grad = True
        for param in self.model.domain_encoder.parameters():
            param.requires_grad = True
        for param in self.model.domain_classifier.parameters():
            param.requires_grad = True
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt['lr'])

        logits = self.model(x, 4)
        loss_0 = self.loss_ce(logits[1], y)
        loss_1 = self.loss_ce(logits[3], dom)
        loss_2 = self.loss_entropy(smax(logits[2]))
        loss_3 = self.loss_entropy(smax(logits[4]))
        loss_4 = self.loss_MSE(logits[5], logits[0]) 
        # TODO TODO TODO loss clip
        loss_5 = self.loss_MSE(logits[6], text_features) 

        loss_final = self.weights[0] * (loss_0 + self.weights[3] * loss_2) + self.weights[1] * (loss_1 + self.weights[4] * loss_3) + self.weights[2] * loss_4 + self.weights[5] * loss_5
        self.optimizer.zero_grad()
        loss_final.backward()
        self.optimizer.step()
        
        return loss_final.item()

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, y, _, desc in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                _ = _.to(self.device)
                desc = desc.to(self.device)

                logits = self.model(x, 4)
                loss += self.criterion(logits[1], y)
                pred = torch.argmax(logits[1], dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss