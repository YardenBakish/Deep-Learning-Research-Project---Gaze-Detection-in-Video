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


class VideoAttTargetLoader(Dataset):
    def __init__(self, data_dir, annotation_dir, depth_data_dir, transform1, transform2, transform3, input_size=224, output_size=64,
                 test=False, imshow=False, seq_len_limit=400):
        shows = glob.glob(os.path.join(annotation_dir, '*'))
        self.all_sequence_paths = []
        for s in shows:
            sequence_annotations = glob.glob(os.path.join(s, '*', '*.txt'))
            self.all_sequence_paths.extend(sequence_annotations)
        self.data_dir = data_dir
        self.depth_data_dir = depth_data_dir
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.input_size = input_size
        self.output_size = output_size
        self.test = test
        self.imshow = imshow
        self.length = len(self.all_sequence_paths)
        self.seq_len_limit = seq_len_limit

    def __getitem__(self, index):
        sequence_path = self.all_sequence_paths[index]
        df = pd.read_csv(sequence_path, header=None, index_col=False,
                         names=['path', 'xmin', 'ymin', 'xmax', 'ymax', 'gazex', 'gazey', 'left_x_min', 'left_y_min','left_x_max','left_y_max', 'right_x_min', 'right_y_min','right_x_max','right_y_max'])
        show_name = sequence_path.split('/')[-3]
        clip = sequence_path.split('/')[-2]
        seq_len = len(df.index)

        # moving-avg smoothing
        window_size = 11 # should be odd number
        df['xmin'] = myutils.smooth_by_conv(window_size, df, 'xmin')
        df['ymin'] = myutils.smooth_by_conv(window_size, df, 'ymin')
        df['xmax'] = myutils.smooth_by_conv(window_size, df, 'xmax')
        df['ymax'] = myutils.smooth_by_conv(window_size, df, 'ymax')
        
        """df['left_x_min'] = myutils.smooth_by_conv(window_size, df, 'left_x_min')
        df['left_y_min'] = myutils.smooth_by_conv(window_size, df, 'left_y_min')
        df['left_x_max'] = myutils.smooth_by_conv(window_size, df, 'left_x_max')
        df['left_y_max'] = myutils.smooth_by_conv(window_size, df, 'left_y_max')


        df['right_x_min'] = myutils.smooth_by_conv(window_size, df, 'right_x_min')
        df['right_y_min'] = myutils.smooth_by_conv(window_size, df, 'right_y_min')
        df['right_x_max'] = myutils.smooth_by_conv(window_size, df, 'right_x_max')
        df['right_y_max'] = myutils.smooth_by_conv(window_size, df, 'right_y_max')"""

        if not self.test:
            # cond for data augmentation
            cond_jitter = np.random.random_sample()
            cond_flip = np.random.random_sample()
            cond_color = np.random.random_sample()
            if cond_color < 0.5:
                n1 = np.random.uniform(0.5, 1.5)
                n2 = np.random.uniform(0.5, 1.5)
                n3 = np.random.uniform(0.5, 1.5)
            cond_crop = np.random.random_sample()

            # if longer than seq_len_limit, cut it down to the limit with the init index randomly sampled
            if seq_len > self.seq_len_limit:
                sampled_ind = np.random.randint(0, seq_len - self.seq_len_limit)
                seq_len = self.seq_len_limit
            else:
                sampled_ind = 0

            if cond_crop < 0.5:
                sliced_x_min = df['xmin'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_x_max = df['xmax'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_y_min = df['ymin'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_y_max = df['ymax'].iloc[sampled_ind:sampled_ind+seq_len]

                sliced_gaze_x = df['gazex'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_gaze_y = df['gazey'].iloc[sampled_ind:sampled_ind+seq_len]

                check_sum = sliced_gaze_x.sum() + sliced_gaze_y.sum()
                all_outside = check_sum == -2*seq_len

                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                if all_outside:
                    crop_x_min = np.min([sliced_x_min.min(), sliced_x_max.min()])
                    crop_y_min = np.min([sliced_y_min.min(), sliced_y_max.min()])
                    crop_x_max = np.max([sliced_x_min.max(), sliced_x_max.max()])
                    crop_y_max = np.max([sliced_y_min.max(), sliced_y_max.max()])
                else:
                    crop_x_min = np.min([sliced_gaze_x.min(), sliced_x_min.min(), sliced_x_max.min()])
                    crop_y_min = np.min([sliced_gaze_y.min(), sliced_y_min.min(), sliced_y_max.min()])
                    crop_x_max = np.max([sliced_gaze_x.max(), sliced_x_min.max(), sliced_x_max.max()])
                    crop_y_max = np.max([sliced_gaze_y.max(), sliced_y_min.max(), sliced_y_max.max()])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Get image size
                path = os.path.join(self.data_dir, show_name, clip, df['path'].iloc[0])
                img = Image.open(path)
                img = img.convert('RGB')
                width, height = img.size

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)
        else:
            sampled_ind = 0


        faces, left_eyes, right_eyes, images, depth_images, head_channels, heatmaps, paths, gazes, imsizes, gaze_inouts = [], [], [], [], [], [], [], [], [], [], []
        index_tracker = -1
        for i, row in df.iterrows():
            index_tracker = index_tracker+1
            if not self.test:
                if index_tracker < sampled_ind or index_tracker >= (sampled_ind + self.seq_len_limit):
                    continue

            face_x1 = row['xmin']  # note: Already in image coordinates
            face_y1 = row['ymin']  # note: Already in image coordinates
            face_x2 = row['xmax']  # note: Already in image coordinates
            face_y2 = row['ymax']  # note: Already in image coordinates
            gaze_x = row['gazex']  # note: Already in image coordinates
            gaze_y = row['gazey']  # note: Already in image coordinates
            left_x_min = int(row['left_x_min'])
            left_y_min = int(row['left_y_min']) 
            left_x_max = int(row['left_x_max']) 
            left_y_max = int(row['left_y_max'])  
            right_x_min = int(row['right_x_min'] ) 
            right_y_min = int(row['right_y_min']) 
            right_x_max = int(row['right_x_max'] )
            right_y_max = int(row['right_y_max']) 

            impath = os.path.join(self.data_dir, show_name, clip, row['path'])
            #print("img path",impath)
            path = row['path']
            depth_impath = None
            if os.path.exists( os.path.join(self.depth_data_dir, show_name, clip,  path[:path.rfind(".")] +".jpg")):
              depth_impath =  os.path.join(self.depth_data_dir, show_name, clip,  path[:path.rfind(".")] +".jpg")
            else:
              print(self.depth_data_dir, show_name, clip,  path[:path.rfind(".")] +".jpg")
              continue
            #print("depth img path", depth_impath)
            img = Image.open(impath)
            img = img.convert('RGB')
            img_depth = cv2.imread(depth_impath)
            #img_depth = cv2.imread(depth_impath,  cv2.IMREAD_GRAYSCALE)
            if img_depth is None:
              print(self.depth_data_dir, show_name, clip,  path[:path.rfind(".")] +".jpg")
              continue 
            #img_depth = cv2.imread(depth_impath)


            width, height = img.size
            imsize = torch.FloatTensor([width, height])
            # imsizes.append(imsize)

            face_x1, face_y1, face_x2, face_y2 = map(float, [face_x1, face_y1, face_x2, face_y2])
            gaze_x, gaze_y = map(float, [gaze_x, gaze_y])
            if gaze_x == -1 and gaze_y == -1:
                gaze_inside = False
            else:
                if gaze_x < 0: # move gaze point that was sliglty outside the image back in
                    gaze_x = 0
                if gaze_y < 0:
                    gaze_y = 0
                gaze_inside = True

            if not self.test:

                # Random Crop
                if cond_crop < 0.5:
                    crop_y_min = max(crop_y_min,0)
                    crop_x_min = max(crop_x_min,0)
                    #print(int(crop_y_min),int(crop_y_min)+int(crop_height), int(crop_x_min),int(crop_x_min)+int(crop_width))

                    # Crop it
                    img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)
                    img_depth = img_depth[int(crop_y_min):int(crop_y_min)+int(crop_height), int(crop_x_min):int(crop_x_min)+int(crop_width),:]
                    #img_depth = img_depth[int(crop_y_min):int(crop_y_min)+int(crop_height), int(crop_x_min):int(crop_x_min)+int(crop_width)]


                    # Record the crop's (x, y) offset
                    offset_x, offset_y = crop_x_min, crop_y_min

                    # convert coordinates into the cropped frame
                    face_x1, face_y1, face_x2, face_y2 = face_x1 - offset_x, face_y1 - offset_y, face_x2 - offset_x, face_y2 - offset_y
                    if gaze_inside:
                        gaze_x, gaze_y = (gaze_x- offset_x), \
                                         (gaze_y - offset_y)
                    else:
                        gaze_x = -1; gaze_y = -1

                    width, height = crop_width, crop_height

                # Flip?
                if cond_flip < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    img_depth = img_depth[:,::-1,:]
                    #img_depth = img_depth[:,::-1]
                    BB_width = face_x2 - face_x1
                    x_max_2 = width - face_x1
                    x_min_2 = width - face_x2
                    face_x2 = x_max_2
                    face_x1 = x_min_2

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
                    if gaze_x != -1 and gaze_y != -1:
                        gaze_x = width - gaze_x

                # Random color change
                if cond_color < 0.5:
                    img = TF.adjust_brightness(img, brightness_factor=n1)
                    img = TF.adjust_contrast(img, contrast_factor=n2)
                    img = TF.adjust_saturation(img, saturation_factor=n3)

            # Face crop
            face = img.copy().crop((int(face_x1), int(face_y1), int(face_x2), int(face_y2)))
            if left_x_min != -1:
              left_eye = face.copy().crop((int(left_x_min), int(left_y_min),int(left_x_max),int(left_y_max)))
              right_eye = face.copy().crop((int(right_x_min), int(right_y_min),int(right_x_max),int(right_y_max)))
            else:
              left_eye = Image.new("RGB", (36, 60), "black")
              right_eye = Image.new("RGB", (36, 60), "black")

          
            # Head channel image
            head_channel = imutils.get_head_box_channel(face_x1, face_y1, face_x2, face_y2, width, height,
                                                        resolution=self.input_size, coordconv=False).unsqueeze(0)
            
            
            
            if self.transform1 is not None:
                img = self.transform1(img)
                face = self.transform1(face)
                img_depth = self.transform2(img_depth)
                left_eye, right_eye = self.transform3(left_eye), self.transform3(right_eye)


            # Deconv output
            if gaze_inside:
                gaze_x /= float(width) # fractional gaze
                gaze_y /= float(height)
                gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
                gaze_map = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                 3,
                                                 type='Gaussian')
                gazes.append(torch.FloatTensor([gaze_x, gaze_y]))
            else:
                gaze_map = torch.zeros(self.output_size, self.output_size)
                gazes.append(torch.FloatTensor([-1, -1]))
            faces.append(face)
            left_eyes.append(left_eye)
            right_eyes.append(right_eye)
            depth_images.append(img_depth)
            images.append(img)
            head_channels.append(head_channel)
            heatmaps.append(gaze_map)
            gaze_inouts.append(torch.FloatTensor([int(gaze_inside)]))


        faces = torch.stack(faces)
        left_eyes = torch.stack(left_eyes)
        right_eyes = torch.stack(right_eyes)
        images = torch.stack(images)
        depth_images = torch.stack(depth_images)
        head_channels = torch.stack(head_channels)
        heatmaps = torch.stack(heatmaps)
        gazes = torch.stack(gazes)
        gaze_inouts = torch.stack(gaze_inouts)
        # imsizes = torch.stack(imsizes)
        # print(faces.shape, images.shape, head_channels.shape, heatmaps.shape)

        if self.test:
            return images, depth_images, faces,left_eyes,right_eyes, head_channels, heatmaps, gazes, gaze_inouts
        else: # train
            return images, depth_images, faces,left_eyes,right_eyes, head_channels, heatmaps, gaze_inouts

    def __len__(self):
        return self.length
