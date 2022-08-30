import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import pandas as pd

import matplotlib as mpl

mpl.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.misc import imresize

import os
import glob
import csv
import cv2

from utils import imutils
from utils import myutils


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class GazeFollowLoader(Dataset):
    def __init__(self, data_dir, csv_path, depth_data_dir, transform1, transform2, transform3, input_size=224, output_size=64,
                 test=False):
        if test:
            column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'meta1', 'meta2',
                            'left_x_min', 'left_y_min','left_x_max','left_y_max', 'right_x_min', 'right_y_min','right_x_max','right_y_max']
            df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            df = df[['path', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max',
                    'bbox_y_max', 'left_x_min', 'left_y_min','left_x_max','left_y_max', 'right_x_min', 
                    'right_y_min','right_x_max','right_y_max']].groupby(['path', 'eye_x'])
            self.keys = list(df.groups.keys())
            self.X_test = df
            self.length = len(self.keys)
        else:
            column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'inout', 'meta1', 'meta2',
                            'left_x_min', 'left_y_min','left_x_max','left_y_max', 'right_x_min', 'right_y_min','right_x_max','right_y_max']
            df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            df = df[df['inout'] != -1]  # only use "in" or "out "gaze. (-1 is invalid, 0 is out gaze)
            df.reset_index(inplace=True)
            self.y_train = df[['bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'eye_x', 'eye_y', 'gaze_x',
                               'gaze_y', 'inout', 'left_x_min', 'left_y_min','left_x_max','left_y_max', 'right_x_min', 'right_y_min','right_x_max','right_y_max']]
            self.X_train = df['path']
            self.length = len(df)

        self.data_dir = data_dir
        self.depth_data_dir = depth_data_dir
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.test = test

        self.input_size = input_size
        self.output_size = output_size
        

    def __getitem__(self, index):
        if self.test:
            g = self.X_test.get_group(self.keys[index])
            cont_gaze = []
            for i, row in g.iterrows():
                path = row['path']
                depth_path = self.depth_data_dir + path[path.find("/"):path.rfind(".")]+".jpg"
                x_min = row['bbox_x_min']
                y_min = row['bbox_y_min']
                x_max = row['bbox_x_max']
                y_max = row['bbox_y_max']
                eye_x = row['eye_x']
                eye_y = row['eye_y']
                gaze_x = row['gaze_x']
                gaze_y = row['gaze_y']
                left_x_min = row['left_x_min'] 
                left_y_min = row['left_y_min'] 
                left_x_max = row['left_x_max'] 
                left_y_max = row['left_y_max']  
                right_x_min = row['right_x_min']  
                right_y_min = row['right_y_min'] 
                right_x_max = row['right_x_max'] 
                right_y_max = row['right_y_max'] 
                cont_gaze.append([gaze_x, gaze_y])  # all ground truth gaze are stacked up
            for j in range(len(cont_gaze), 20):
                cont_gaze.append([-1, -1])  # pad dummy gaze to match size for batch processing
            cont_gaze = torch.FloatTensor(cont_gaze)
            gaze_inside = True # always consider test samples as inside
        else:
            path = self.X_train.iloc[index]
            depth_path = self.depth_data_dir + path[path.find("/"):path.rfind(".")]+".jpg"
            x_min, y_min, x_max, y_max, eye_x, eye_y, gaze_x, gaze_y, inout, left_x_min, left_y_min,left_x_max,left_y_max, right_x_min, right_y_min,right_x_max,right_y_max = self.y_train.iloc[index]
            gaze_inside = bool(inout)

        # expand face bbox a bit
        """k = 0.1
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += k * abs(y_max - y_min)"""
        x_min = max(0,x_min)
        y_min = max(0, y_min)
        

        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert('RGB')
        img_depth = cv2.imread(depth_path)
        #img_depth = cv2.imread(depth_path,  cv2.IMREAD_GRAYSCALE)
        width, height = img.size
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

        if self.test:
            imsize = torch.IntTensor([width, height])
        else:
            ## data augmentation

            # Jitter (expansion-only) bounding box size
            """if np.random.random_sample() <= 0.5:
                k = np.random.random_sample() * 0.2
                x_min -= k * abs(x_max - x_min)
                y_min -= k * abs(y_max - y_min)
                x_max += k * abs(x_max - x_min)
                y_max += k * abs(y_max - y_min)"""

            # Random Crop
            if np.random.random_sample() <= 0.5:
                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                crop_x_min = np.min([gaze_x * width, x_min, x_max])
                crop_y_min = np.min([gaze_y * height, y_min, y_max])
                crop_x_max = np.max([gaze_x * width, x_min, x_max])
                crop_y_max = np.max([gaze_y * height, y_min, y_max])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)

                # Crop it
                img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)
                img_depth = img_depth[int(crop_y_min):int(crop_y_min)+int(crop_height), int(crop_x_min):int(crop_x_min)+int(crop_width),:]
                #img_depth = img_depth[int(crop_y_min):int(crop_y_min)+int(crop_height), int(crop_x_min):int(crop_x_min)+int(crop_width)]
                
                # Record the crop's (x, y) offset
                offset_x, offset_y = crop_x_min, crop_y_min

                # convert coordinates into the cropped frame
                x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y
                # if gaze_inside:
                gaze_x, gaze_y = (gaze_x * width - offset_x) / float(crop_width), \
                                 (gaze_y * height - offset_y) / float(crop_height)
                # else:
                #     gaze_x = -1; gaze_y = -1

                width, height = crop_width, crop_height

            # Random flip
            if np.random.random_sample() <= 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                #img_depth = img_depth[:,::-1]
                img_depth = img_depth[:,::-1,:]
                BB_width = x_max - x_min
                x_max_2 = width - x_min
                x_min_2 = width - x_max
                x_max = x_max_2
                x_min = x_min_2
                if left_x_min != -1:
                  left_x_max_2 = BB_width - left_x_min
                  left_x_min_2 = BB_width - left_x_max
                  left_x_max = left_x_max_2
                  left_x_min = left_x_min_2

                  right_x_max_2 = BB_width - right_x_min
                  right_x_min_2 = BB_width - right_x_max
                  right_x_max = right_x_max_2
                  right_x_min = right_x_min_2

                  tmpright_x_min, tmpright_y_min,tmpright_x_max,tmpright_y_max = left_x_min, left_y_min,left_x_max,left_y_max
                  left_x_min, left_y_min,left_x_max,left_y_max =  right_x_min, right_y_min,right_x_max,right_y_max
                  right_x_min, right_y_min,right_x_max,right_y_max = tmpright_x_min, tmpright_y_min,tmpright_x_max,tmpright_y_max

                gaze_x = 1 - gaze_x

            # Random color change
            if np.random.random_sample() <= 0.5:
                img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        head_channel = imutils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                    resolution=self.input_size, coordconv=False).unsqueeze(0)

        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        if left_x_min != -1:
          left_eye = face.crop((int(left_x_min), int(left_y_min),int(left_x_max),int(left_y_max)))
          right_eye = face.crop((int(right_x_min), int(right_y_min),int(right_x_max),int(right_y_max)))
        else:
          left_eye = Image.new("RGB", (36, 60), "black")
          right_eye = Image.new("RGB", (36, 60), "black")

        if self.transform1 is not None:
          img = self.transform1(img)
          face = self.transform1(face)
          img_depth = self.transform2(img_depth)
          left_eye, right_eye = self.transform3(left_eye), self.transform3(right_eye)

        # generate the heat map used for deconv prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
        if self.test:  # aggregated heatmap
            num_valid = 0
            for gaze_x, gaze_y in cont_gaze:
                if gaze_x != -1:
                    num_valid += 1
                    gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                         3,
                                                         type='Gaussian')
            gaze_heatmap /= num_valid
        else:
            # if gaze_inside:
            gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                 3,
                                                 type='Gaussian')

        if self.test:
            return img, img_depth, face, left_eye, right_eye, head_channel, gaze_heatmap, cont_gaze, imsize, path
        else:
            return img, img_depth, face, left_eye, right_eye, head_channel, gaze_heatmap, gaze_inside

    def __len__(self):
        return self.length