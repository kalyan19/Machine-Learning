import urllib.request
import cv2
import numpy as np
import os
from PIL import Image


IMAGE_LINKS = ['http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n00017222', \
              'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n00015388', \
              'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n09287968']

#IMAGE_LINKS = ['http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n09618957']
DIR = 'temp'
FILE_EXT = {0: '.png', 1: '.jpg'}



"""
Pulls down images and stores them as desired format
"""
def store_raw_images():

    img_num = 0

    if not os.path.exists(DIR):
        os.makedirs(DIR)

    for image_link in IMAGE_LINKS:
        image_urls = urllib.request.urlopen(image_link).read().decode()
        for i in image_urls.split('\n'):
            try:
                print(i)
                urllib.request.urlretrieve(i, DIR +"/"+str(img_num)+FILE_EXT[1])
                im = Image.open('{}/{}'.format(DIR, str(img_num)+FILE_EXT[1]))
                im.save(  os.path.splitext('{}/{}'.format(DIR, str(img_num)))[0]  + FILE_EXT[0])
                os.remove(DIR +"/"+str(img_num)+FILE_EXT[1])
                img_num += 1

            except Exception as e:
                print(str(e))

"""
Removes bad images like "Image not found"
"""
def filter_bad_images():
    match = False
    for file_type in [DIR]:
        for img in os.listdir(file_type):
            if img.endswith(FILE_EXT[0]):
                for bad_img in os.listdir('filter'):
                    try:
                        current_image_path = str(file_type)+'/'+str(img)
                        bad_img = cv2.imread('filter/'+str(bad_img))
                        question = cv2.imread(current_image_path)
                        if bad_img.shape == question.shape and not(np.bitwise_xor(bad_img,question).any()):
                            print('Deleting ', current_image_path)
                            os.remove(current_image_path)
                    except Exception as e:
                        print(str(e))

"Rename images in order after filtered"
def rename_images():
    count = 0
    for file_type in [DIR]:
        imgs = os.listdir(file_type)
        imgs = sorted(imgs,key=lambda x: int(os.path.splitext(x)[0]))
        #print(lsorted)
        #print('imgs',imgs)
        for img in imgs:
            if img.endswith(FILE_EXT[0]):
                os.rename(str(file_type) + "/" + str(img), str(file_type) + "/" +  os.path.splitext(str(count))[0] + FILE_EXT[0])
                count += 1

def main():
    #store_raw_images()
    #filter_bad_images()
    rename_images()

main()
