import numpy as np
import argparse
import math
import random
from PIL import Image
import os
import dlib
import math
import torch
import pandas as pd
import torchvision.transforms as transforms
import sys
import imageio
from functools import partial
import cv2
import multiprocessing as mp


def get_args():
    parser = argparse.ArgumentParser(description='eyePatches')
    parser.add_argument('-train_input', type=str, help='input txt file path')
    parser.add_argument('-data', type=str,choices = ["train","test","video"])
    parser.add_argument('-test_input', type=str, help='input txt file path')
    parser.add_argument('-validation_input', type=str, help='input txt file path')

    parser.add_argument('-output_dir', type=str, help='output dir for files')   
    args = parser.parse_args()
    return args


def extract_eye_patch(image_path, output_dir, fh, sp, detections):
  print(image_path)
  #src = cv2.imread(image_path)
  #print(src.shape[0])
  #if src.shape[0] < 200 or src.shape[1] < 200:
  #  return -1
  #img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
  #detections = detector(img, 1)
  #print(detections)
  #datFile = '/content/shape_predictor_68_face_landmarks.dat'
  #sp = dlib.shape_predictor(datFile)
  faces = dlib.full_object_detections()

  for det in detections:
      faces.append(sp(img, det))

  if (len(faces) != 1):
    return -1
  # Bounding box and eyes
                      # Convert out of dlib format

  right_eyes = [[face.part(i) for i in range(36, 42)] for face in faces]
  right_eyes = [[(i.x, i.y) for i in eye] for eye in right_eyes]          # Convert out of dlib format

  left_eyes = [[face.part(i) for i in range(42, 48)] for face in faces]
  left_eyes = [[(i.x, i.y) for i in eye] for eye in left_eyes] 

  #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

  if(len(left_eyes) != 1 or len(right_eyes) != 1):
    return -1

  

  k = 0.3
  right_x_max = 0
  right_y_max = 0
  right_x_min = 0
  right_y_min = 0
  for eye in right_eyes:
      right_x_max = max(eye, key=lambda x: x[0])[0]
      right_y_max = max(eye, key=lambda x: x[1])[1]
      right_x_min = min(eye, key=lambda x: x[0])[0]
      right_y_min = min(eye, key=lambda x: x[1])[1]

      right_x_min -= k * abs(right_x_max - right_x_min)
      right_y_min -= 3*k * abs(right_y_max - right_y_min)
      right_x_max += k * abs(right_x_max - right_x_min)
      right_y_max += k * abs(right_y_max - right_y_min)

      """if (right_x_max - right_x_min <= 12 or right_y_max - right_y_min <= 12):
        print("here")
        return -1"""
      #cv2.rectangle(img, (int(left_x_min), int(left_y_min)),(int(left_x_max), int(left_y_max)),(0, 0, 255), 1)


  left_x_max = 0
  left_y_max = 0
  left_x_min = 0
  left_y_min = 0
  for eye in left_eyes:
      left_x_max = max(eye, key=lambda x: x[0])[0]
      left_y_max = max(eye, key=lambda x: x[1])[1]
      left_x_min = min(eye, key=lambda x: x[0])[0]
      left_y_min = min(eye, key=lambda x: x[1])[1]
      
      left_x_min -= k * abs(left_x_max - left_x_min)
      left_y_min -= 3*k * abs(left_y_max - left_y_min)
      left_x_max += k * abs(left_x_max - left_x_min)
      left_y_max += k * abs(left_y_max - left_y_min)

      """if (left_x_max - left_x_min <= 12 or left_y_max - left_y_min <= 12):
          return -1"""
  #fh.write(f"{int(left_x_min)} {int(left_y_min)} {int(left_x_max)} {int(left_y_max)} ")
  #fh.write(f"{int(right_x_min)} {int(right_y_min)} {int(right_x_max)} {int(right_y_max)}")


def extract_bbox(image_path, detector, sp):
  
  path, bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max, inout = image_path
  
  if inout == -190909:
    return None
  bbox_x_min = max(bbox_x_min,0)
  bbox_y_min = max(bbox_y_min,0)
  if bbox_x_min > bbox_x_max:
    print("HEY1")
    bbox_x_min, bbox_x_max = bbox_x_max, bbox_x_min
  if bbox_y_min > bbox_y_max:
    print("HEY2")
    bbox_y_min, bbox_y_max = bbox_y_max, bbox_y_min
  print(path, bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max, inout)
  try:
    src = cv2.imread(path)
    src = src[int(bbox_y_min):int(bbox_y_max),int(bbox_x_min):int(bbox_x_max)]
  #if src.shape[0] < 200 or src.shape[1] < 200:
    #return None
  
    img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
  except:
    return None

  detections = detector(img, 1)
  faces = dlib.full_object_detections()

  for det in detections:
      faces.append(sp(img, det))

  if (len(faces) != 1):
    return None
  # Bounding box and eyes
                      # Convert out of dlib format

  right_eyes = [[face.part(i) for i in range(36, 42)] for face in faces]
  right_eyes = [[(i.x, i.y) for i in eye] for eye in right_eyes]          # Convert out of dlib format

  left_eyes = [[face.part(i) for i in range(42, 48)] for face in faces]
  left_eyes = [[(i.x, i.y) for i in eye] for eye in left_eyes] 

  #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

  if(len(left_eyes) != 1 or len(right_eyes) != 1):
    return None

  k = 0.3
  right_x_max = 0
  right_y_max = 0
  right_x_min = 0
  right_y_min = 0
  for eye in right_eyes:
      right_x_max = max(eye, key=lambda x: x[0])[0]
      right_y_max = max(eye, key=lambda x: x[1])[1]
      right_x_min = min(eye, key=lambda x: x[0])[0]
      right_y_min = min(eye, key=lambda x: x[1])[1]

      right_x_min -= k * abs(right_x_max - right_x_min)
      right_y_min -= 3*k * abs(right_y_max - right_y_min)
      right_x_max += k * abs(right_x_max - right_x_min)
      right_y_max += 2*k * abs(right_y_max - right_y_min)

      #if (right_x_max - right_x_min <= 10 or right_y_max - right_y_min <= 10):
        #return None
      #cv2.rectangle(img, (int(left_x_min), int(left_y_min)),(int(left_x_max), int(left_y_max)),(0, 0, 255), 1)
  
  left_x_max = 0
  left_y_max = 0
  left_x_min = 0
  left_y_min = 0
  for eye in left_eyes:
      left_x_max = max(eye, key=lambda x: x[0])[0]
      left_y_max = max(eye, key=lambda x: x[1])[1]
      left_x_min = min(eye, key=lambda x: x[0])[0]
      left_y_min = min(eye, key=lambda x: x[1])[1]
      
      left_x_min -= k * abs(left_x_max - left_x_min)
      left_y_min -= 3*k * abs(left_y_max - left_y_min)
      left_x_max += k * abs(left_x_max - left_x_min)
      left_y_max += 2*k * abs(left_y_max - left_y_min)

      #if (left_x_max - left_x_min <= 10 or left_y_max - left_y_min <= 10):
        #return None

  return [int(left_x_min),int(left_y_min),int(left_x_max) ,int(left_y_max), int(right_x_min),int(right_y_min),int(right_x_max) ,int(right_y_max)]


  

def main(args):
  print("HERE")
  op = args.data
  output_dir = args.output_dir
  train_path = args.train_input
  test_path = args.test_input
  validation_path = args.validation_input
  detector = dlib.get_frontal_face_detector()
  if train_path != None:
    datFile = '/content/attention_proggress/shape_predictor_68_face_landmarks.dat'
    sp = dlib.shape_predictor(datFile)
    fh_train = open(train_path, 'r')
    x = os.path.basename(train_path)
    fh_output_train = open(os.path.join(output_dir, x), 'w')
    print(os.path.join(output_dir, x))
    Lines = fh_train.readlines()
    fh_train.close()
    column_names = []
    if op == "test":
      column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'meta']
    if op == "train":
      column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'inout', 'meta']

    if op == "video":
       column_names = ['path','bbox_x_min','bbox_y_min','bbox_x_max','bbox_y_max','gaze_x','inout']

    df = pd.read_csv(train_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
    imgs_path = df[['path','bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max','inout']]
    #print(imgs_path["path"].iloc[0])

    imgs_path = imgs_path.values.tolist()
    bbox = None
    with mp.Pool(mp.cpu_count() -1) as p:
      bbox = p.map(partial(extract_bbox, detector = detector, sp = sp), imgs_path)
    for i in range(len(Lines)):
      if i%1000 ==0:
        print(i,"/", len(Lines))
      fh_output_train.write(Lines[i][:-1])
      if bbox[i] == None:
        fh_output_train.write(",-1,-1,-1,-1,-1,-1,-1,-1\n")
      else:
        for coor in bbox[i]:
          fh_output_train.write(","+str(coor))
        fh_output_train.write("\n")

    fh_output_train.close()




if __name__ == '__main__':
    args = get_args()
    main(args)
