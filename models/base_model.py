import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
from parse_args import parse_arguments

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        x = x.squeeze()
        if len(x.size()) < 2:
            x = x.unsqueeze(0)
        return x

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, 7)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.category_encoder(x)
        x = self.classifier(x)
        return x

class DomainDisentangleModel(nn.Module):
    def __init__(self):
        super(DomainDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.opt = parse_arguments()

        #TODO verify domain encoder
        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # category encoder as in baseline
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        if self.opt['dg'] and self.opt['pda']:
            self.domain_classifier = nn.Linear(512, 4)
        elif self.opt['dg'] and not self.opt['pda']:
            self.domain_classifier = nn.Linear(512, 3)
        else:
            self.domain_classifier = nn.Linear(512, 2)

        self.category_classifier = nn.Linear(512, 7)

        # reconstructor
        self.feature_reconstructor = nn.Sequential( # test reconstructor
            #nn.Conv1d(1024, 512, 2),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )


    def forward(self, x, step):
        
        # training the category classifier
        if step==0:
            x = self.feature_extractor(x)
            x = self.category_encoder(x)
            x = self.category_classifier(x)
            return x

        # training the domain classifier
        if step==1:
            x = self.feature_extractor(x)
            x = self.domain_encoder(x)
            x = self.domain_classifier(x)
            return x
        
        # adversarial training the disentangler fooling the category classifier 
        if step==2:
            x = self.feature_extractor(x)
            x = self.domain_encoder(x)
            x = self.category_classifier(x)
            return x

        # adversarial training the disentangler fooling the domain classifier 
        if step==3:
            x = self.feature_extractor(x)
            x = self.category_encoder(x)
            x = self.domain_classifier(x)
            return x

        # end to end, feature reconstructor, return three results for different losses computation
        if step==4:
            x = self.feature_extractor(x)
            x1 = self.category_encoder(x)
            x1_class = self.category_classifier(x1)
            x1_adv = self.domain_classifier(x1)
            x2 = self.domain_encoder(x)
            x2_class = self.domain_classifier(x2)
            x2_adv = self.category_classifier(x2) 
            x_rec = self.feature_reconstructor(torch.cat((x1,x2), 1)) 

            return x, x1_class, x1_adv, x2_class, x2_adv, x_rec
        
class ClipDisentangleModel(nn.Module):
    def __init__(self):
        super(ClipDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.opt = parse_arguments()

        #TODO verify domain encoder
        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # category encoder as in baseline
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        if self.opt['dg'] and self.opt['pda']:
            self.domain_classifier = nn.Linear(512, 4)
        elif self.opt['dg'] and not self.opt['pda']:
            self.domain_classifier = nn.Linear(512, 3)
        else:
            self.domain_classifier = nn.Linear(512, 2)

        self.category_classifier = nn.Linear(512, 7)

        # reconstructor
        self.feature_reconstructor = nn.Sequential( # test reconstructor
            #nn.Conv1d(1024, 512, 2),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def forward(self, x, step):
        
        # training the category classifier
        if step==0:
            x = self.feature_extractor(x)
            x = self.category_encoder(x)
            x = self.category_classifier(x)
            return x

        # training the domain classifier
        if step==1:
            x = self.feature_extractor(x)
            x = self.domain_encoder(x)
            x = self.domain_classifier(x)
            return x
        
        # adversarial training the disentangler fooling the category classifier 
        if step==2:
            x = self.feature_extractor(x)
            x = self.domain_encoder(x)
            x = self.category_classifier(x)
            return x

        # adversarial training the disentangler fooling the domain classifier 
        if step==3:
            x = self.feature_extractor(x)
            x = self.category_encoder(x)
            x = self.domain_classifier(x)
            return x

        # end to end, feature reconstructor, return three results for different losses computation
        if step==4:
            x = self.feature_extractor(x)
            x1 = self.category_encoder(x)
            x1_class = self.category_classifier(x1)
            x1_adv = self.domain_classifier(x1)
            x2 = self.domain_encoder(x)
            x2_class = self.domain_classifier(x2)
            x2_adv = self.category_classifier(x2) 
            x_rec = self.feature_reconstructor(torch.cat((x1,x2), 1)) # test reconstructor

            return x, x1_class, x1_adv, x2_class, x2_adv, x_rec, x2