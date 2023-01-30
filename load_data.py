from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import json

CATEGORIES = {
    'dog': 0,
    'elephant': 1,
    'giraffe': 2,
    'guitar': 3,
    'horse': 4,
    'house': 5,
    'person': 6,
}

LABEL_FILE_PATH = "data/image_descriptions.json"


class PACSDatasetBaseline(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, y = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y

def read_lines(data_path, domain_name):
    examples = {}
    with open(f'{data_path}/{domain_name}.txt') as f:
        lines = f.readlines()

    for line in lines: 
        line = line.strip().split()[0].split('/')
        category_name = line[3]
        category_idx = CATEGORIES[category_name]
        image_name = line[4]
        image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
        if category_idx not in examples.keys():
            examples[category_idx] = [image_path]
        else:
            examples[category_idx].append(image_path)
    return examples

def get_descriptions() -> dict:
    f = open(LABEL_FILE_PATH, "r")
    desc_json = json.loads(f.read())
    dict_description = {}

    for elem in desc_json:
        description = str()
        path = f'data/PACS/kfold/{elem["image_name"]}'
        for descr in elem["descriptions"]:
            if description=="":
                description = descr
            else:
                description = description + ", " + descr
        final_descr = description
        dict_description.update({path: final_descr})

    return dict_description

def build_splits_baseline(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines(opt['data_path'], source_domain)
    target_examples = read_lines(opt['data_path'], target_domain)
    
    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation

    train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category_idx] * val_split_length)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
            else:
                val_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetBaseline(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetBaseline(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader


class PACSDatasetDomDisentangle(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, y, dom = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y, dom


def build_splits_domain_disentangle(opt):

    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines(opt['data_path'], source_domain)
    target_examples = read_lines(opt['data_path'], target_domain)

    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}
    val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation

    # target_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in target_examples.items()}
    # target_total_examples = sum(target_category_ratios.values())
    # target_category_ratios = {category_idx: c / target_total_examples for category_idx, c in target_category_ratios.items()}
    # val_split_length_target = target_total_examples * 0.2 # 20% of the training split used for validation

    train_examples = []
    val_examples = []
    #train_examples_dom = []
    #val_examples_dom = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category_idx] * val_split_length)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                train_examples.append([example, category_idx, 0]) # each triplet is [path_to_img, class_label, domain]
            else:
                val_examples.append([example, category_idx, 0]) # each triplet is [path_to_img, class_label, domain]
    
    # for category_idx, examples_list in target_examples.items():
    #     split_idx = round(target_category_ratios[category_idx] * val_split_length_target)
    #     for i, example in enumerate(examples_list):
    #         if i > split_idx:
    #             train_examples_dom.append([example, category_idx, 1]) # each triplet is [path_to_img, class_label, domain]
    #         else:
    #             val_examples_dom.append([example, category_idx, 1]) # each triplet is [path_to_img, class_label, domain]
    
    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_examples.append([example, category_idx, 1]) # each triplet is [path_to_img, class_label, domain]

    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetDomDisentangle(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    #train_dom_loader = DataLoader(PACSDatasetDomDisentangle(train_examples_dom, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetDomDisentangle(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False) 
    #val_dom_loader = DataLoader(PACSDatasetDomDisentangle(val_examples_dom, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetDomDisentangle(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader #train_dom_loader,val_dom_loader

class PACSDatasetClipDisentangle(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, y, dom, description = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y, dom, description


def build_splits_clip_disentangle(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']
    descriptions = get_descriptions()

    source_examples = read_lines(opt['data_path'], source_domain)
    target_examples = read_lines(opt['data_path'], target_domain)

    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}
    val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation

    train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category_idx] * val_split_length)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                if example in descriptions.keys():
                    train_examples.append([example, category_idx, 0, descriptions[example]]) # each triplet is [path_to_img, class_label, domain]
                else:
                    train_examples.append([example, category_idx, 0])
            else:
                val_examples.append([example, category_idx]) # each triplet is [path_to_img, class_label, domain]
    
    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_examples.append([example, category_idx]) # each triplet is [path_to_img, class_label, domain]

    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetClipDisentangle(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetClipDisentangle(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False) 
    test_loader = DataLoader(PACSDatasetClipDisentangle(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader 

