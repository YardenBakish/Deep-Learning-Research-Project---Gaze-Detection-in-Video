import imageio
import matplotlib.pyplot as plt
import cv2
import torch
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
from scipy.misc import imresize
import numpy as np
import os
from pathlib import Path
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from PIL import Image
import glob
import matplotlib.patches as patches
from VideoAttTargetModel import Complete_Model
from torchvision import transforms
from utils import imutils
from utils import myutils, evaluation
import pandas as pd
import argparse 
import subprocess
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


cmd = f'mkdir output/'
subprocess.call(cmd,shell = True)

parser = argparse.ArgumentParser()
parser.add_argument("--person", default = "left", choices = ["left", "right"], type=str, help="select 'left' for left person and 'right' for right person")
args = parser.parse_args()


annotations_dir = "data_demo/images/"
imgs_dir  = "data_demo/images"
depth_imgs_dir = "data_demo/DepthMaps/"
output_dir = "output/"
csv_path = ""
if args.person == "left":
  csv_path = "data_demo/s00.txt"
else:
  csv_path = "data_demo/s01.txt"
model_weights = "workshop_model.pth.tar"

column_names = ['path', 'xmin', 'ymin', 'xmax', 'ymax', 'gazex', 'gazey', 'left_x_min', 'left_y_min','left_x_max','left_y_max', 'right_x_min', 'right_y_min','right_x_max','right_y_max']
df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
transform1 = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
transform2 = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)),transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
transform3 = transforms.Compose([transforms.Resize((36, 60)),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


path_list = df['path'].tolist()
xmin_list = df['xmin'].tolist()
ymin_list = df['ymin'].tolist()
xmax_list = df['xmax'].tolist()
ymax_list = df['ymax'].tolist()
pre_trans_img = []
pre_trans_depth_img = []

window_size = 11 # should be odd number
df['xmin'] = myutils.smooth_by_conv(window_size, df, 'xmin')
df['ymin'] = myutils.smooth_by_conv(window_size, df, 'ymin')
df['xmax'] = myutils.smooth_by_conv(window_size, df, 'xmax')
df['ymax'] = myutils.smooth_by_conv(window_size, df, 'ymax')


model = Complete_Model(num_lstm_layers = 2)
model.cuda(0)
model_dict = model.state_dict()
pretrained_dict = torch.load(model_weights)
pretrained_dict = pretrained_dict['state_dict']
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.train(False)
chunk_size = 3

with torch.no_grad():
  faces, left_eyes, right_eyes, images, depth_images, head_locations = [], [], [], [], [], []
  for ind in df.index:    
    row = df.iloc[[ind]]
    #print(row['path'].iloc[0])
    face_x1 = row['xmin'].iloc[0]  
    face_y1 = row['ymin'].iloc[0] 
    face_x2 = row['xmax'].iloc[0]  
    face_y2 = row['ymax'].iloc[0]  
    gaze_x = row['gazex'].iloc[0]  
    gaze_y = row['gazey'].iloc[0]
    path = row['path'].iloc[0]
    left_x_min = int(row['left_x_min'].iloc[0])
    left_y_min = int(row['left_y_min'].iloc[0]) 
    left_x_max = int(row['left_x_max'].iloc[0]) 
    left_y_max = int(row['left_y_max'].iloc[0])  
    right_x_min = int(row['right_x_min'].iloc[0] ) 
    right_y_min = int(row['right_y_min'].iloc[0]) 
    right_x_max = int(row['right_x_max'].iloc[0] )
    right_y_max = int(row['right_y_max'].iloc[0]) 
    impath   = os.path.join(imgs_dir,path)
    #print(impath, face_x1,face_y1,face_x2,face_y2)
    depth_impath   = os.path.join(depth_imgs_dir,path[:path.rfind(".")] +".jpg")
    img = Image.open(impath)
    img = img.convert('RGB')    
    pre_trans_img.append(img)

    img_depth = cv2.imread(depth_impath)
    pre_trans_depth_img.append(img_depth)
    width, height = img.size
    imsize = torch.FloatTensor([width, height])
    # imsizes.append(imsize)

    face_x1, face_y1, face_x2, face_y2 = map(float, [face_x1, face_y1, face_x2, face_y2])
    

    face = img.copy().crop((int(face_x1), int(face_y1), int(face_x2), int(face_y2)))
    if left_x_min != -1:
      left_eye = face.copy().crop((int(left_x_min), int(left_y_min),int(left_x_max),int(left_y_max)))
      right_eye = face.copy().crop((int(right_x_min), int(right_y_min),int(right_x_max),int(right_y_max)))
    else:
      left_eye = Image.new("RGB", (36, 60), "black")
      right_eye = Image.new("RGB", (36, 60), "black")
    # Head channel image
    head_location = imutils.get_head_box_channel(face_x1, face_y1, face_x2, face_y2, width, height,
                                                resolution=224, coordconv=False).unsqueeze(0)
    img = transform1(img)
    face = transform1(face)
    img_depth =transform2(img_depth)
    left_eye, right_eye = transform3(left_eye), transform3(right_eye)


    faces.append(face)
    depth_images.append(img_depth)
    images.append(img)
    head_locations.append(head_location)
    left_eyes.append(left_eye)
    right_eyes.append(right_eye)

  faces = torch.stack(faces)
  left_eyes = torch.stack(left_eyes)
  right_eyes = torch.stack(right_eyes)
  images = torch.stack(images)
  depth_images = torch.stack(depth_images)
  head_locations = torch.stack(head_locations)

  faces = faces[None, :]
  left_eyes = left_eyes[None, :]
  right_eyes = right_eyes[None, :]
  images = images[None, :]
  depth_images = depth_images[None, :]
  head_locations = head_locations[None, :]
  lengths = [df.shape[0]]


  frame_sequence =  pack_padded_sequence(images, lengths, batch_first=True)
  frame_sequence, pad_sizes = frame_sequence.data, frame_sequence.batch_sizes

  depth_frame_sequence =  pack_padded_sequence(depth_images, lengths, batch_first=True)
  depth_frame_sequence, pad_depth_sizes = depth_frame_sequence.data, depth_frame_sequence.batch_sizes

  head_loc_sequence= (pack_padded_sequence(head_locations, lengths, batch_first=True)).data
  face_sequence= (pack_padded_sequence(faces, lengths, batch_first=True)).data
  left_eye_sequence= (pack_padded_sequence(left_eyes, lengths, batch_first=True)).data
  right_eye_sequence= (pack_padded_sequence(right_eyes, lengths, batch_first=True)).data
  hx = (torch.zeros((2, 1, 512, 7, 7)).cuda(0),
        torch.zeros((2, 1, 512, 7, 7)).cuda(0)) # (num_layers, batch_size, feature dims)
  last_index = 0
  previous_hx_size = 1

  for i in range(0, lengths[0], chunk_size):
      # In this for loop, we read batched images across the time dimension
          # we step forward N = chunk_size frames args
      pad_sizes_slice = pad_sizes[i:i + chunk_size]
      curr_length = np.sum(pad_sizes_slice.cpu().detach().numpy())
      # slice padded data
      frame_sequence_slice = frame_sequence[last_index:last_index + curr_length].cuda(0)
      depth_frame_sequence_slice = depth_frame_sequence[last_index:last_index + curr_length].cuda(0)
      head_loc_sequence_slice = head_loc_sequence[last_index:last_index + curr_length].cuda(0)
      face_sequence_slice = face_sequence[last_index:last_index + curr_length].cuda(0)
      left_eye_sequence_slice = left_eye_sequence[last_index:last_index + curr_length].cuda(0)
      right_eye_sequence_slice = right_eye_sequence[last_index:last_index + curr_length].cuda(0)
      last_index += curr_length

      # detach previous hidden states to stop gradient flow
      prev_hx = (hx[0][:, :min(pad_sizes_slice[0], previous_hx_size), :, :, :].detach(),
                  hx[1][:, :min(pad_sizes_slice[0], previous_hx_size), :, :, :].detach())

      # forward pass
      deconv, inout_val, hx = model(frame_sequence_slice, depth_frame_sequence_slice, head_loc_sequence_slice, face_sequence_slice, \
                                                left_eye_sequence_slice, right_eye_sequence_slice, hidden_scene=prev_hx, batch_sizes=pad_sizes_slice)
      #print(deconv.shape, inout_val.shape)
      previous_hx_size = pad_sizes_slice[-1]
      for j in range(deconv.shape[0]):
        # heatmap modulation
        new_id = i +j
        raw_hm = deconv[j].cpu().detach().numpy() * 255
        raw_hm = raw_hm.squeeze()
        inout = inout_val[j].cpu().detach().numpy()
        inout = 1 / (1 + np.exp(-inout))
        inout = (1 - inout) * 255
        norm_map = imresize(raw_hm, (height, width)) - inout

        xmin, y_min, x_max, y_max = int(xmin_list[new_id]),int(ymin_list[new_id]),int(xmax_list[new_id]),int(ymax_list[new_id])
        rect = patches.Rectangle((xmin, y_min), x_max-xmin, y_max-y_min, linewidth=2, edgecolor=(0,1,0), facecolor='none') 
        
        frame = pre_trans_img[new_id]
        depth_map = pre_trans_depth_img[new_id]
        
        plt.close()
        fig = plt.figure()
        
        fig.add_subplot(2, 2, 1)
        plt.imshow(frame)
        plt.axis('off')
        plt.title("Scene")

        ax = plt.gca()
        ax.add_patch(rect)
        if inout < 200:
          pred_x, pred_y = evaluation.argmax_pts(raw_hm)
          norm_p = [pred_x/64, pred_y/64]
          circ = patches.Circle((norm_p[0]*width, norm_p[1]*height), height/50.0, facecolor=(0,1,0), edgecolor='none')
          ax.add_patch(circ)
          plt.plot((norm_p[0]*width,(xmin+x_max)/2), (norm_p[1]*height,(y_min+y_max)/2), '-', color=(0,1,0,1))
      
        fig.add_subplot(2, 2, 2)
        plt.imshow(depth_map)
        plt.axis('off')
        plt.title("Depth Map")
        ax = plt.gca()
        #ax.add_patch(rect)
        if inout < 100:
          plt.imshow(norm_map, cmap = 'rainbow', alpha=0.2, vmin=0, vmax=255)
        plt.tight_layout(pad=0.00)
        plt.savefig(f"output/{'%03d' % new_id}", bbox_inches="tight")
        print(f"output/{'%03d' % new_id} saved")
   
print("running demo has finished")
