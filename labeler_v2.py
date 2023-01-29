from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('domain', type=str)
parser.add_argument('--to_label', type=str, default='data/to_label/')
parser.add_argument('--image_path', type=str, default='data/PACS/kfold/')
parser.add_argument('--clip_labels_path', type=str, default='data/clip_labels.txt') 
parser.add_argument('--checkpoint_path', type=str, default='data/checkpoint.txt')

opt = vars(parser.parse_args())

image_path = opt['image_path']
clip_labels_path =opt['clip_labels_path']
checkpoint_path =opt['checkpoint_path']
to_label_path = opt['to_label']

if __name__ == '__main__':

    target_domain = opt['domain']
    with open(to_label_path+target_domain+'_to_label.txt', 'r') as file:
        to_label = [elem.rstrip().split(' ')[0] for elem in file.readlines()]
    
    
    if(not os.path.isfile(checkpoint_path)):
        checkpoint = 0
    else:
        with open(checkpoint_path, 'r') as file:
            checkpoint = int(file.readline()) + 1
    #for image in to_label:
    for i in range(checkpoint, len(to_label)):
        path = image_path + to_label[i]
        print(f"Opening: {path}")
        
        im = Image.open(path)
        plt.imshow(im)
        plt.show(block = False)
        plt.axis(False)

        lod = input('Level of detail, if the image is rich in detail or it is a raw representation: ')
        edges = input('Edges, description of the contours of the object: ')
        sat = input('Color Saturation, the intensity and brilliance of colors in the image: ')
        shades = input('Color Shades, if there are shades of colors in the image: ')
        bg = input('Background, description of the background: ')
        instances = input('Single Instance, if the image is composed by a single instance or multiple instances: ')
        txt = input('Text, if there is text in the image: ')
        texture = input('Texture, if there is a visual pattern repeated: ')
        persp = input('Perspective, if the three dimensional proportions of the object parts are realistic: ')
        plt.clf()
        plt.close()
        line = "{'image_name': '%s', 'descriptions': ['%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s']}, " % (to_label[i], 
                lod, edges, sat, shades, bg, instances, txt, texture, persp)
        
        with open(clip_labels_path, 'a') as f:
            f.write(line)
        with open(checkpoint_path, 'w') as f:
            f.write(str(i))
        i += 1
        