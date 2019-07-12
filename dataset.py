import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils import is_image_file, load_img
import glob
import random
import os
import numpy as np
from PIL import Image
import cv2
import pickle
from tqdm import tqdm

class MSRA10K(data.Dataset):
    def __init__(self, root, visual_effect, training=True):
        super(MSRA10K, self).__init__()

        self.files_A = []
        self.files_B = []

        self.root = os.path.join(root, 'MSRA10K_Imgs_GT/Imgs/')
        self.effect = visual_effect
        self.training = training

        if not os.path.isdir(self.root):
            print('path not exist')
            assert False
        
        self.image = glob.glob(self.root + '/*.jpg')
        self.gt = glob.glob(self.root + '/*.png')

        if len(self.image) != len(self.gt):
            print('label and image not the same')
            assert False

        if self.training:
            for i in tqdm(range(9500)):
                if i%2 == 0:
                    self.files_A.append(self.image[i])
                else:
                    self.files_B.append(self.image[i])
        else:
            self.files_A = self.image[9500:]


        transform_list = [transforms.Resize(224, Image.BICUBIC), 
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(180, resample=False, expand=False, center=None),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):


        # Normal image
        img_A = cv2.imread(self.files_A[index])

        # Visual Effect image
        if self.training:
            img_B = cv2.imread(self.files_B[index])
            maskB = cv2.imread(self.files_B[index].replace('jpg', 'png'))

            if self.effect == 'black-background':
                effect_A = np.zeros_like((img_A))
                effect_B = np.zeros_like((img_B))
            elif self.effect == 'color-selectivo':
                effect_A = cv2.imread(self.files_A[index], 0)
                effect_A = np.tile(effect_A[:, :, None], [1, 1, 3])
                effect_B = cv2.imread(self.files_B[index], 0)
                effect_B = np.tile(effect_B[:, :, None], [1, 1, 3])
            elif self.effect == 'defocus':
                kernel = np.ones((11, 11),np.float32)/121
                effect_A = cv2.filter2D(img_A,-1,kernel)
                effect_B = cv2.filter2D(img_B,-1,kernel)
    
            stich = np.where(maskB==255, img_B, effect_B)

            normal_img = self.fill_blank(img_A)
            background = self.fill_blank(effect_A)
            effect_img = self.fill_blank(stich)

            angle = random.randint(-180,180)
            flip  = random.randint(-1,2)
            normal_img = self.rotate_and_flip(normal_img, angle, flip=flip)
            background = self.rotate_and_flip(background, angle, flip=flip)
            effect_img = self.rotate_and_flip(effect_img, random.randint(-180,180), flip=random.randint(-1,2))

        else:

            if self.effect == 'black-background':
                effect_A = np.zeros_like((img_A))
            elif self.effect == 'color-selectivo':
                effect_A = cv2.imread(self.files_A[index], 0)
                effect_A = np.tile(effect_A[:, :, None], [1, 1, 3])
            elif self.effect == 'defocus':
                kernel = np.ones((11, 11),np.float32)/121
                effect_A = cv2.filter2D(img_A,-1,kernel)

            maskA = cv2.imread(self.files_A[index].replace('jpg', 'png'))
            stich = np.where(maskA==255, img_A, effect_A)


            normal_img = self.fill_blank(img_A)
            background = self.fill_blank(effect_A)
            effect_img = self.fill_blank(stich)


        normal_img = self.transform(Image.fromarray(np.uint8(normal_img)))
        background = self.transform(Image.fromarray(np.uint8(background)))
        effect_img = self.transform(Image.fromarray(np.uint8(effect_img)))

        return normal_img, background, effect_img


    def __len__(self):
        return len(self.files_A)

    def fill_blank(self, crop_img_A):

        # h, w, c = crop_img_A.shape
        # s = max(h, w)

        # l_h = int((s-h)/2)
        # l_w = int((s-w)/2)

        # blank = np.zeros((s, s, 3))

        # blank[l_h:l_h+h, l_w:l_w+w, :] = crop_img_A

        blank = cv2.resize(crop_img_A,(224,224),interpolation=cv2.INTER_CUBIC)

        return blank

    def rotate_and_flip(self, image, angle, center=None, scale=1.0, flip=2):

        if flip!=2:
            image = cv2.flip(image, flip)

        (h, w) = image.shape[:2]
    
        if center is None:
            center = (w / 2, h / 2)
    
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
    
        return rotated

class ECSSD(data.Dataset):
    def __init__(self, root, visual_effect, training=True):
        super(ECSSD, self).__init__()

        self.files_A = []
        self.files_B = []

        self.root = os.path.join(root, 'ECSSD/')
        self.effect = visual_effect
        self.training = training

        if not os.path.isdir(self.root):
            print('path not exist')
            assert False
        
        self.image = glob.glob(self.root + '/images/*.jpg')
        self.gt = glob.glob(self.root + '/ground_truth_mask/*.png')

        if len(self.image) != len(self.gt):
            print('label and image not the same')
            assert False

        if self.training:
            for i in tqdm(range(950)):
                if i%2 == 0:
                    self.files_A.append(self.image[i])
                else:
                    self.files_B.append(self.image[i])
        else:
            self.files_A = self.image[950:]


        transform_list = [transforms.Resize(224, Image.BICUBIC), 
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(180, resample=False, expand=False, center=None),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):


        # Normal image
        img_A = cv2.imread(self.files_A[index])

        # Visual Effect image
        if self.training:
            img_B = cv2.imread(self.files_B[index])
            maskB = cv2.imread(self.files_B[index].replace('jpg', 'png').replace('images', 'ground_truth_mask'))

            if self.effect == 'black-background':
                effect_A = np.zeros_like((img_A))
                effect_B = np.zeros_like((img_B))
            elif self.effect == 'color-selectivo':
                effect_A = cv2.imread(self.files_A[index], 0)
                effect_A = np.tile(effect_A[:, :, None], [1, 1, 3])
                effect_B = cv2.imread(self.files_B[index], 0)
                effect_B = np.tile(effect_B[:, :, None], [1, 1, 3])
            elif self.effect == 'defocus':
                kernel = np.ones((11, 11),np.float32)/121
                effect_A = cv2.filter2D(img_A,-1,kernel)
                effect_B = cv2.filter2D(img_B,-1,kernel)
    
            stich = np.where(maskB==255, img_B, effect_B)

            normal_img = self.fill_blank(img_A)
            background = self.fill_blank(effect_A)
            effect_img = self.fill_blank(stich)

            angle = random.randint(-180,180)
            flip  = random.randint(-1,2)
            normal_img = self.rotate_and_flip(normal_img, angle, flip=flip)
            background = self.rotate_and_flip(background, angle, flip=flip)
            effect_img = self.rotate_and_flip(effect_img, random.randint(-180,180), flip=random.randint(-1,2))

        else:

            if self.effect == 'black-background':
                effect_A = np.zeros_like((img_A))
            elif self.effect == 'color-selectivo':
                effect_A = cv2.imread(self.files_A[index], 0)
                effect_A = np.tile(effect_A[:, :, None], [1, 1, 3])
            elif self.effect == 'defocus':
                kernel = np.ones((11, 11),np.float32)/121
                effect_A = cv2.filter2D(img_A,-1,kernel)

            maskA = cv2.imread(self.files_A[index].replace('jpg', 'png').replace('images', 'ground_truth_mask'))
            stich = np.where(maskA==255, img_A, effect_A)


            normal_img = self.fill_blank(img_A)
            background = self.fill_blank(effect_A)
            effect_img = self.fill_blank(stich)


        normal_img = self.transform(Image.fromarray(np.uint8(normal_img)))
        background = self.transform(Image.fromarray(np.uint8(background)))
        effect_img = self.transform(Image.fromarray(np.uint8(effect_img)))

        return normal_img, background, effect_img


    def __len__(self):
        return len(self.files_A)

    def fill_blank(self, crop_img_A):

        # h, w, c = crop_img_A.shape
        # s = max(h, w)

        # l_h = int((s-h)/2)
        # l_w = int((s-w)/2)

        # blank = np.zeros((s, s, 3))

        # blank[l_h:l_h+h, l_w:l_w+w, :] = crop_img_A

        blank = cv2.resize(crop_img_A,(224,224),interpolation=cv2.INTER_CUBIC)

        return blank

    def rotate_and_flip(self, image, angle, center=None, scale=1.0, flip=2):

        if flip!=2:
            image = cv2.flip(image, flip)

        (h, w) = image.shape[:2]
    
        if center is None:
            center = (w / 2, h / 2)
    
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
    
        return rotated


