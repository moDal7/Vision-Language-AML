from PIL import Image
import matplotlib.pyplot as plt

to_label = []

IMAGE_PATH = "data/PACS/kfold/"
CLIP_LABELS_PATH ="data/clip_labels.txt"

if __name__ == '__main__':

    target_domain = input("Target domain: ")
    with open(f'data/to_label/'+target_domain+'_to_label.txt') as f:
        lines = f.readlines()

        for line in lines: 
            line = line.strip().split()[0]
            to_label.append(line)
    
    for image in to_label:
        path = IMAGE_PATH + image
        print(f"Opening: {path}")
        
        im = Image.open(path)
        plt.imshow(im)
        plt.show(block = False)

        lod = input('Level of detail, if the image is rich in detail or it is a raw representation: ')
        edges = input('Edges, description of the contours of the object: ')
        sat = input('Color Saturation, the intensity and brilliance of colors in the image: ')
        shades = input('Color Shades, if there are shades of colors in the image: ')
        bg = input('Background, description of the background: ')
        instances = input('Single Instance, if the image is composed by a single instance or mulitple instances: ')
        txt = input('Text, if there is text in the image: ')
        texture = input('Texture, if there is a visual pattern repeated: ')
        persp = input('Perspective, if the three dimensional proportions of the object parts are realistic: ')
        plt.clf()
        plt.close()
        line = "{'image_name': '%s', 'descriptions': ['%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s']}, " % (image, 
                lod, edges, sat, shades, bg, instances, txt, texture, persp)
        
        with open(CLIP_LABELS_PATH, 'a') as f:
            f.write(line)
        