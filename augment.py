import cv2
from PIL import Image
import numpy as np
import skimage
from matplotlib import pyplot as plt
from skimage.util import random_noise
import os

class ImageAugment:
    def add_gaussian_noise(self,image):
        gauss = random_noise(image, mode='gaussian', seed=None, clip=True)
        return gauss

    def add_salt_pepper_noise(self,image):
        sp = random_noise(image, mode='s&p', seed=None, clip=True)
        return sp

    def add_poisson_noise(self,image):
        poisson = random_noise(image, mode='poisson', seed=None, clip=True)
        return poisson

    def add_speckle_noise(self,image):
        speckle = random_noise(image, mode='speckle', seed=None, clip=True)
        return speckle

    def flip_vertical(self,image):
        flipVertical = cv2.flip(image, 0)
        return flipVertical

    def flip_horizontal(self,image):
        flipHorizontal = cv2.flip(image, 1)
        return flipHorizontal

    def do_augmentation(self,image):
        image = self.add_gaussian_noise(image)
        image = self.add_salt_pepper_noise(image)
        image = self.add_poisson_noise(image)
        image = self.add_speckle_noise(image)
        image = self.flip_vertical(image)
        image = self.flip_horizontal(image)
        return image

class Generator(ImageAugment):
    def __init__(self,path):
        self.path = path

    def image_generator(self):
        path = self.path
        file_list = []
        for root,dirs,files in os.walk(path):
            for file in files:
                file_list.append(os.path.join(root,file))
        i=0
        while(True):
            yield file_list[i]
            i+=1

    def on_next(self):
        gen_obj = self.image_generator()
        return next(gen_obj)

    def augment_and_show(self):
        print(self.on_next())
        while True:
            image = cv2.imread(self.on_next())
            image = self.do_augmentation(image)
            cv2.imshow('image', image)
            cv2.waitKey(0)

    def label_generator(self,feat,labels):
        i=0
        while (True):
            yield feat[i],labels[i]
            i+=1


# image_path = './data/image1.jpg'
# I = cv2.imread(image_path, 1)
#
# aug = ImageAugment()
# gauss = aug.add_gaussian_noise(I)
# sp = aug.add_salt_pepper_noise(I)
# poisson = aug.add_poisson_noise(I)
# speckle = aug.add_speckle_noise(I)
# flipv = aug.flip_vertical(I)
# fliph = aug.flip_horizontal(I)
#
image_gen = Generator('./data')
# augmented = image_gen.do_augmentation(I)

gen = image_gen.augment_and_show()

# plt.subplot(421), plt.imshow(I), plt.title('Origin')
# plt.subplot(422), plt.imshow(gauss), plt.title('Gaussian')
# plt.subplot(423), plt.imshow(sp), plt.title('Salt and Pepper')
# plt.subplot(424), plt.imshow(poisson), plt.title('Poisson')
# plt.subplot(425), plt.imshow(speckle), plt.title('Speckle')
# plt.subplot(426), plt.imshow(flipv), plt.title('Flip Vertical')
# plt.subplot(427), plt.imshow(fliph), plt.title('Flip Horizontal')
# plt.subplot(428), plt.imshow(augmented), plt.title('Augmented')
#
# plt.show()
