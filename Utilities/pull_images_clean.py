import urllib.request
import cv2
import numpy as np
import os
from PIL import Image


FILTER = 'filter'

# give absolute path

"""
Pulls down images and stores them as desired format
target_dir - abs location to save files
image_link - url with list of urls of images
input_type - what the input images are
output_type - what the images will be saved as
start - the starting number for images to be saved as
"""
def store_raw_images(target_dir, image_link, input_type = '.jpg', output_type = '.png', start = 0):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for image_link in target_dir:
        image_urls = urllib.request.urlopen(image_link).read().decode()
        for i in image_urls.split('\n'):
            try:
                urllib.request.urlretrieve(i, os.path.join(target_dir, str(start)+input_type))
                im = Image.open(os.path.join(target_dir, str(start)+input_type))
                im.save(os.path.splitext(os.path.join(target_dir, str(start)+input_type))[0]  + output_type)
                os.remove(os.path.join(target_dir, str(start)+input_type))
                start += 1
            except Exception as e:
                print(str(e))

    return start

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
