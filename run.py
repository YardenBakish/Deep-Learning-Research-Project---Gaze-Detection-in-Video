"""
Yarden Bakish
208539270
"""


import torch
from torchvision import transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
from VideoAttTargetModel import Complete_Model
from GazeFollowModel import Static_Model
from GazeFollowLoader import GazeFollowLoader
from VideoAttTarget_loader import VideoAttTargetLoader
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
from lib.pytorch_convolutional_rnn import convolutional_rnn

from utils import imutils, evaluation, misc

import argparse
import os
from datetime import datetime
import shutil
import numpy as np
from scipy.misc import imresize
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


##########################################
#FILE PATHS
##########################################
gazefollow_train_data = "dataset/gazefollow_extended/"
gazefollow_train_label = "dataset/gazefollow_train_modified.txt"
gazefollow_val_data = "dataset/gazefollow_extended/"
gazefollow_val_label = "dataset/gazefollow_test_modified.txt"

gazefollow_train_depth_data = "dataset/gazefollow_extended/depth_train"
gazefollow_test_depth_data = "dataset/gazefollow_extended/depth_test"
network_name = 'WORKSHOP'

videoattentiontarget_train_data = "dataset/videoattentiontarget/images"
videoattentiontarget_train_label = "refined_attn_video/train"
videoattentiontarget_val_data = "dataset/videoattentiontarget/images"
videoattentiontarget_val_label = "refined_attn_video/test"
video_depth_imgs    = "dataset/videoattentiontarget/depth"

##########################################
#CMD ARGUMENTS
##########################################
parser = argparse.ArgumentParser()
parser.add_argument("--init_weights", type=str, help="initial weights")
parser.add_argument("--Dataset", choices = ['GazeFollow', 'VideoAttTarget'], required=True, help="which model to train")
parser.add_argument("--mode", type=str, choices = ["train", "test"],default="test", help="train or test the model")
args = parser.parse_args()


##########################################
#MAIN
##########################################


##########################################
#GazeFollow

#Train or test a spatial model (no lstm layers) on the GazeFollow dataset 
##########################################

def runGazeFollow():
  #HYPERPAREMETERS
  epochs = 70
  batch_size = 48
  lr = 2.5e-4
  input_resolution = 224
  output_resolution = 64

  best_AUC = 0
  best_min_dist = 100

  np.random.seed(1)

  device = torch.device('cuda', 0)
  model = Static_Model()
  model.cuda().to(device)

  cudnn.benchmark = True

  transform_scene = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  transform_depth = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)),transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
  transform_eyes = transforms.Compose([transforms.Resize((36, 60)),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

  train_dataset = GazeFollowLoader(gazefollow_train_data, gazefollow_train_label, gazefollow_train_depth_data,
                    transform_scene,transform_depth,transform_eyes, input_size=input_resolution, output_size=output_resolution)
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True, 
                                              pin_memory=True,
                                              num_workers=0)

  val_dataset = GazeFollowLoader(gazefollow_val_data, gazefollow_val_label, gazefollow_test_depth_data,
                    transform_scene,transform_depth,transform_eyes,input_size=input_resolution, output_size=output_resolution, test=True)
  val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              pin_memory=True,
                                              num_workers=0)

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  remain_epoch = epochs
  
  checkpoint_tar = "/content/drive/MyDrive/workshop/checkpoint_gaze_depth_eyes.pth.tar"
  checkpoint_tar_best = "/content/drive/MyDrive/workshop/checkpoint_gaze_depth_eyes_best.pth.tar"
  
  #choose between user-provided weights or default: initial_weights_for_spatial_training.pt
  if args.init_weights:
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.init_weights)
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
  else:
    checkpoint = torch.load(checkpoint_tar)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    AUC = checkpoint['AUC']
    min_dist = checkpoint['min_dist']
    avg_dist = checkpoint['avg_dist']
    remain_epoch = checkpoint['remain_epoch']

    best_checkpoint = torch.load(checkpoint_tar_best)
    best_AUC = best_checkpoint['AUC']
    best_min_dist = best_checkpoint['min_dist']
    best_avg_dist = best_checkpoint['avg_dist']

  #LOSS FUNCTIONS: mse_loss for heatmap regression and bce for identifying out-of-frame targets 
  mse_loss      = nn.MSELoss(reduce=False) 
  bcelogit_loss = nn.BCEWithLogitsLoss()
  
  #MODE (test\train)
  #(1) test model
  if args.mode == "test":
    AUC, min_dist, avg_dist = validateGazeFollow(model, val_loader)
    print("\tAUC:{:.4f}\tmin dist:{:.4f}\tavg dist:{:.4f}".format(
    AUC,min_dist, avg_dist))
  #(2) train model
  else:
    for ep in range(0, remain_epoch):
      #train model for one epoch
      total_loss = trainGazeFollow(train_loader, model, mse_loss,bcelogit_loss, optimizer, ep)
      # evaluate model by AUC, min distance and average distance
      AUC, min_dist, avg_dist = validateGazeFollow(model, val_loader)
      #back up model
      state = {'epoch': ep+1,
          'optimizer': optimizer.state_dict(),
          'state_dict': model.state_dict(),
          'AUC': torch.mean(torch.tensor(AUC)),
          'min_dist': torch.mean(torch.tensor(min_dist)),
          'avg_dist': torch.mean(torch.tensor(avg_dist)),
          'remain_epoch': remain_epoch - (ep + 1)}
      torch.save(state, checkpoint_tar) 




def trainGazeFollow(train_loader, model, mse_loss,bcelogit_loss, optimizer, epoch):
  """
  Function: trainGazeFollow
  Input: data loader, static model, mse loss function, bce loss function, optimizer, current epoch
  Output: None

  Description: trains the static model for one epoch
  """
  device = torch.device('cuda', 0)
  
  for i, (frame, frame_depth, face_image, left_eye, right_eye, head_location, gt_heatmap, gt_gaze_inside) in enumerate(train_loader):
    model.train(True)
    frames = frame.cuda().to(device)
    frames_depth = frame_depth.cuda().to(device)
    head_loc = head_location.cuda().to(device)
    faces_img = face_image.cuda().to(device)
    left_eyes = left_eye.cuda().to(device)
    right_eyes = right_eye.cuda().to(device)
    gt_heatmap = gt_heatmap.cuda().to(device)
    heatmap_pred, inout_pred = model(frames,frames_depth, head_loc, faces_img, left_eyes, right_eyes)
    heatmap_pred = heatmap_pred.squeeze(1)

    l2_loss = mse_loss(heatmap_pred, gt_heatmap)*10000 #multiplied by a factor to prevent underflow
    l2_loss = torch.mean(l2_loss, dim=1) 
    l2_loss = torch.mean(l2_loss, dim=1)
    gt_gaze_inside = gt_gaze_inside.cuda(device).to(torch.float)
    l2_loss = torch.mul(l2_loss, gt_gaze_inside) #l2 loss is computed only for in-frame targets instances
    l2_loss = torch.sum(l2_loss)/torch.sum(gt_gaze_inside)
        
    BCE_loss = bcelogit_loss(inout_pred.squeeze(), gt_gaze_inside.squeeze())*100

    total_loss = l2_loss + BCE_loss

    total_loss.backward() 

    optimizer.step()
    optimizer.zero_grad()

    print("Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss: (l2){:.4f} (bce){:.4f}".format(epoch, i+1, len(train_loader), l2_loss, BCE_loss))
  return 0

      
        
def validateGazeFollow(model,test_loader):
  """
  Function: validateGazeFollow
  Input: static model, validation data loader
  Output: None
  Description: evaluates the static model
  """  
  model.train(False)
  AUC = []; min_dist = []; avg_dist = []   
  with torch.no_grad():
    for i, (frame, frame_depth, face_img, left_eye, right_eye, head_location, gt_heatmap, cont_gaze, imsize, _) in enumerate(test_loader):
      frames = frame.cuda().to(0)
      frame_depth = frame_depth.cuda().to(0)
      head_loc = head_location.cuda().to(0)
      faces = face_img.cuda().to(0)
      left_eyes = left_eye.cuda().to(0)
      right_eyes = right_eye.cuda().to(0)
      gt_heatmap = gt_heatmap.cuda().to(0)
      heatmap_pred, inout_pred = model(frames,frame_depth,head_loc, faces, left_eyes, right_eyes)
      heatmap_pred = heatmap_pred.squeeze(1)

      # go through each data point and record AUC, min dist, avg dist
      for b_i in range(len(cont_gaze)):
        # remove padding and recover valid ground truth points
        valid_gaze = cont_gaze[b_i]
        valid_gaze = valid_gaze[valid_gaze != -1].view(-1,2)
        # AUC: area under curve of ROC
        multi_hot = imutils.multi_hot_targets(cont_gaze[b_i], imsize[b_i])
        scaled_heatmap = imresize(heatmap_pred[b_i].cpu(), (imsize[b_i][1].cpu(), imsize[b_i][0].cpu()), interp = 'bilinear')
        auc_score = evaluation.auc(scaled_heatmap, multi_hot)
        AUC.append(auc_score)
        # min distance: minimum among all possible pairs of <ground truth point, predicted point>
        pred_x, pred_y = evaluation.argmax_pts(heatmap_pred[b_i].cpu())
        norm_p = [pred_x/float(64), pred_y/float(64)]
        all_distances = []
        for gt_gaze in valid_gaze:
            all_distances.append(evaluation.L2_dist(gt_gaze, norm_p))
        min_dist.append(min(all_distances))
        # average distance: distance between the predicted point and human average point
        mean_gt_gaze = torch.mean(valid_gaze, 0)
        avg_distance = evaluation.L2_dist(mean_gt_gaze, norm_p)
        avg_dist.append(avg_distance)

  return (
    torch.mean(torch.tensor(AUC)),
    torch.mean(torch.tensor(min_dist)),
    torch.mean(torch.tensor(avg_dist)))



##########################################
#VideoAttentionTarget

#Train or test the complete temporal model on the videoAtttarget dataset 
##########################################


def runVideoAttTarget():
  #HYPERPARAMETERS
  epochs = 3
  chunk_size = 3
  batch_size = 8
  lr = 5e-5
  input_resolution = 224
  output_resolution = 64
  remain_epoch = epochs
  best_AUC = 0
  best_dist = 100
  best_in_vs_out_pred = -1
  transform_scene = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  transform_depth = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)),transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
  transform_eyes = transforms.Compose([transforms.Resize((36, 60)),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


  train_dataset = VideoAttTargetLoader(videoattentiontarget_train_data, videoattentiontarget_train_label, video_depth_imgs,
                                        transform1=transform_scene,transform2=transform_depth,transform3=transform_eyes, test=False, seq_len_limit=50)
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                              collate_fn=video_pack_sequences)
  val_dataset = VideoAttTargetLoader(videoattentiontarget_val_data, videoattentiontarget_val_label, video_depth_imgs,
                                        transform1=transform_scene,transform2=transform_depth,transform3=transform_eyes, test=True, seq_len_limit=50)
  val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              collate_fn=video_pack_sequences)
  
  np.random.seed(1)

  device = torch.device('cuda', 0)

  num_lstm_layers = 2

  model = Complete_Model(num_lstm_layers = num_lstm_layers)
  model.cuda(device)
  optimizer = None

  optimizer = torch.optim.Adam([
          {'params': model.convlstm_scene.parameters(), 'lr': lr},
          {'params': model.deconv1.parameters(), 'lr': lr},
          {'params': model.deconv2.parameters(), 'lr': lr},
          {'params': model.deconv3.parameters(), 'lr': lr},
          {'params': model.conv4.parameters(), 'lr': lr},
          {'params': model.fc_inout.parameters(), 'lr': lr*5},
          ], lr = 0)
  
  checkpoint_tar = "/content/drive/MyDrive/workshop/video_gaze_depth_eyes2.pth.tar"
  checkpoint_tar_best = "/content/drive/MyDrive/workshop/video_gaze_depth_eyes_best2.pth.tar"

  if args.init_weights:  
    model_dict = model.state_dict()
    trained_dict = torch.load(args.init_weights)

    trained_dict = trained_dict['state_dict']
    model_dict.update(trained_dict)
    model.load_state_dict(model_dict)

    optimizer = torch.optim.Adam([
                {'params': model.convlstm_scene.parameters(), 'lr': lr},
                {'params': model.deconv1.parameters(), 'lr': lr},
                {'params': model.deconv2.parameters(), 'lr': lr},
                {'params': model.deconv3.parameters(), 'lr': lr},
                {'params': model.conv4.parameters(), 'lr': lr},
                {'params': model.fc_inout.parameters(), 'lr': lr*5},
                ], lr = 0)
  else:
    checkpoint = torch.load(checkpoint_tar)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    remain_epoch = checkpoint['remain_epoch']
    epochs = remain_epoch
    AUC =      checkpoint['AUC']
    dist =checkpoint['dist'] 
    in_vs_out_pred =  checkpoint['in_vs_out_pred']
    print(epochs,"AUC",AUC, "dist",dist, "in_vs_out_pred", in_vs_out_pred)

    best_checkpoint = torch.load(checkpoint_tar_best)
    best_AUC = best_checkpoint['AUC']
    best_dist = best_checkpoint['dist']
    best_in_vs_out_pred = best_checkpoint['in_vs_out_pred']
    print(f'Best Results:\nAUC: {best_AUC}, min_dist: {best_dist}, avg_dist: {best_in_vs_out_pred}\n')


  #LOSS FUNCTIONS: mse_loss for heatmap regression and bce for identifying out-of-frame targets 

  mse_loss = nn.MSELoss(reduce=False) 
  bcelogit_loss = nn.BCEWithLogitsLoss()
  optimizer.zero_grad()
  #MODE
  if args.mode == "test":     
    AUC, dist, in_vs_out_pred = validateVideoAttTarget(model, val_loader)
    print("\tAUC:{:.4f}\tdist:{:.4f}\in_vs_out_pred:{:.4f}".format(
    AUC,dist, in_vs_out_pred))
  else:
    for ep in range(0, remain_epoch):
      
      total_loss = trainVideoAttTarget(model, train_loader,mse_loss, bcelogit_loss, optimizer, ep)

      AUC, dist, in_vs_out_pred = validateVideoAttTarget(model, val_loader)
      print("\tAUC:{:.4f}\tdist:{:.4f}\tin_vs_out_pred:{:.4f}".format(AUC,dist, in_vs_out_pred))

      state = {'epoch': ep+1,
      'optimizer': optimizer.state_dict(),
      'state_dict': model.state_dict(),
      'AUC': AUC,
      'dist': dist,
      'in_vs_out_pred': in_vs_out_pred,
      'remain_epoch': remain_epoch - (ep + 1)}
      torch.save(state, checkpoint_tar)

   
def trainVideoAttTarget(model, train_loader,mse_loss, bcelogit_loss, optimizer, ep):
  device = torch.device('cuda', 0)
  num_lstm_layers = 2
  total_loss = 0
  batch_size = 8
  chunk_size = 3
  for batch, (frame, depth_frame, face, left_eye, right_eye, head_location, gt_heatmap, gt_inout, lengths) in enumerate(train_loader):
    model.train(True)
    # freeze batchnorm layers
    for module in model.modules():
        if isinstance(module, torch.nn.modules.BatchNorm1d):
            module.eval()
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval()
        if isinstance(module, torch.nn.modules.BatchNorm3d):
            module.eval()

    frame_sequence =  pack_padded_sequence(frame, lengths, batch_first=True)
    frame_sequence, pad_sizes = frame_sequence.data, frame_sequence.batch_sizes
    depth_frame_sequence =  pack_padded_sequence(depth_frame, lengths, batch_first=True)
    depth_frame_sequence, pad_depth_sizes = depth_frame_sequence.data, depth_frame_sequence.batch_sizes
    head_loc_sequence= (pack_padded_sequence(head_location, lengths, batch_first=True)).data
    face_sequence= (pack_padded_sequence(face, lengths, batch_first=True)).data
    left_eye_sequence= (pack_padded_sequence(left_eye, lengths, batch_first=True)).data
    right_eye_sequence= (pack_padded_sequence(right_eye, lengths, batch_first=True)).data
    gt_heatmap_sequence= (pack_padded_sequence(gt_heatmap, lengths, batch_first=True)).data
    gt_inout_sequence = (pack_padded_sequence(gt_inout, lengths, batch_first=True)).data

    hx = (torch.zeros((num_lstm_layers, batch_size, 512, 7, 7)).cuda(device),
          torch.zeros((num_lstm_layers, batch_size, 512, 7, 7)).cuda(device)) # (num_layers, batch_size, feature dims)
    last_index = 0
    previous_hx_size = batch_size

    for i in range(0, lengths[0], chunk_size):
      # In this for loop, we read batched images across the time dimension
          # we step forward N = chunk_size frames args
      pad_sizes_slice = pad_sizes[i:i + chunk_size]
      curr_length = np.sum(pad_sizes_slice.cpu().detach().numpy())
      # slice padded data
      frame_sequence_slice = frame_sequence[last_index:last_index + curr_length].cuda(device)
      depth_frame_sequence_slice = depth_frame_sequence[last_index:last_index + curr_length].cuda(device)
      head_loc_sequence_slice = head_loc_sequence[last_index:last_index + curr_length].cuda(device)
      face_sequence_slice = face_sequence[last_index:last_index + curr_length].cuda(device)
      left_eye_sequence_slice = left_eye_sequence[last_index:last_index + curr_length].cuda(device)
      right_eye_sequence_slice = right_eye_sequence[last_index:last_index + curr_length].cuda(device)
      gt_heatmap_sequence_slice = gt_heatmap_sequence[last_index:last_index + curr_length].cuda(device)
      gt_inout_sequence_slice = gt_inout_sequence[last_index:last_index + curr_length].cuda(device)
      last_index += curr_length

      # detach previous hidden states to stop gradient flow
      prev_hx = (hx[0][:, :min(pad_sizes_slice[0], previous_hx_size), :, :, :].detach(),
                  hx[1][:, :min(pad_sizes_slice[0], previous_hx_size), :, :, :].detach())

      # forward pass
      deconv, inout_val, hx = model(frame_sequence_slice, depth_frame_sequence_slice, head_loc_sequence_slice, face_sequence_slice, \
                                                      left_eye_sequence_slice, right_eye_sequence_slice, hidden_scene=prev_hx, batch_sizes=pad_sizes_slice)


        #print(type(deconv), type(inout_val), type(hx))
        #print(deconv.shape, inout_val.shape, hx.shape)
        
      # compute loss
          # l2 loss computed only for inside case
      l2_loss = mse_loss(deconv.squeeze(1), gt_heatmap_sequence_slice) * 10000 
      l2_loss = torch.mean(l2_loss, dim=1)
      l2_loss = torch.mean(l2_loss, dim=1)
      gt_inout_sequence_slice = gt_inout_sequence_slice.cuda(device).to(torch.float).squeeze()
      l2_loss = torch.mul(l2_loss, gt_inout_sequence_slice) # zero out loss when it's outside gaze case
      l2_loss = torch.sum(l2_loss)/torch.sum(gt_inout_sequence_slice)
          # cross entropy loss for in vs out
      BCE_loss = bcelogit_loss(inout_val.squeeze(), gt_inout_sequence_slice.squeeze())*100

      total_loss = l2_loss + BCE_loss
      total_loss.backward() # loss accumulation

      # update model parameters
      optimizer.step()
      optimizer.zero_grad()

      previous_hx_size = pad_sizes_slice[-1]

      
      print("Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss: (l2){:.4f} (Xent){:.4f}".format(ep, batch+1, len(train_loader), l2_loss, BCE_loss))
  return 0



def validateVideoAttTarget(model, val_loader):
  """
  Function: validateVideoAttTarget
  Input: Complete model, validation data loader
  Output: None
  Description: evaluates the complete model 
  """  
  model.train(False)
  num_lstm_layers = 2
  batch_size = 8
  chunk_size = 3
  device = torch.device('cuda', 0)
  AUC = []; in_vs_out_groundtruth = []; in_vs_out_pred = []; distance = []
  chunk_size = 3
  with torch.no_grad():
      for batch, (frame, depth_frame, face, left_eye, right_eye, head_location, gt_heatmap, gazes, gt_inout, lengths) in enumerate(val_loader):
        print('\tprogress = ', batch+1, '/', len(val_loader))
        
        frame_sequence =  pack_padded_sequence(frame, lengths, batch_first=True)
        frame_sequence, X_pad_sizes = frame_sequence.data, frame_sequence.batch_sizes
        depth_frame_sequence =  pack_padded_sequence(depth_frame, lengths, batch_first=True)
        depth_frame_sequence, X_pad_depth_sizes = depth_frame_sequence.data, depth_frame_sequence.batch_sizes
        head_loc_sequence= (pack_padded_sequence(head_location, lengths, batch_first=True)).data
        face_sequence= (pack_padded_sequence(face, lengths, batch_first=True)).data
        left_eye_sequence= (pack_padded_sequence(left_eye, lengths, batch_first=True)).data
        right_eye_sequence= (pack_padded_sequence(right_eye, lengths, batch_first=True)).data
        cont_gaze_sequence = (pack_padded_sequence(gazes, lengths, batch_first=True)).data
        gt_heatmap_sequence= (pack_padded_sequence(gt_heatmap, lengths, batch_first=True)).data
        gt_inout_sequence = (pack_padded_sequence(gt_inout, lengths, batch_first=True)).data
        hx = (torch.zeros((num_lstm_layers, batch_size, 512, 7, 7)).cuda(device),
              torch.zeros((num_lstm_layers, batch_size, 512, 7, 7)).cuda(device)) # (num_layers, batch_size, feature dims)
        last_index = 0
        previous_hx_size = batch_size

        for i in range(0, lengths[0], chunk_size):
            # In this for loop, we read batched images across the time dimension
                # we step forward N = chunk_size frames args
            X_pad_sizes_slice = X_pad_sizes[i:i + chunk_size]
            curr_length = np.sum(X_pad_sizes_slice.cpu().detach().numpy())
            # slice padded data
            frame_sequence_slice = frame_sequence[last_index:last_index + curr_length].cuda(device)
            depth_frame_sequence_slice = depth_frame_sequence[last_index:last_index + curr_length].cuda(device)
            head_loc_sequence_slice = head_loc_sequence[last_index:last_index + curr_length].cuda(device)
            face_sequence_slice = face_sequence[last_index:last_index + curr_length].cuda(device)
            left_eye_sequence_slice = left_eye_sequence[last_index:last_index + curr_length].cuda(device)
            right_eye_sequence_slice = right_eye_sequence[last_index:last_index + curr_length].cuda(device)
            cont_gaze_sequence_slice = cont_gaze_sequence[last_index:last_index + curr_length].cuda(device)
            gt_heatmap_sequence_slice = gt_heatmap_sequence[last_index:last_index + curr_length].cuda(device)
            gt_inout_sequence_slice = gt_inout_sequence[last_index:last_index + curr_length].cuda(device)
            last_index += curr_length

            # detach previous hidden states to stop gradient flow
            prev_hx = (hx[0][:, :min(X_pad_sizes_slice[0], previous_hx_size), :, :, :].detach(),
                        hx[1][:, :min(X_pad_sizes_slice[0], previous_hx_size), :, :, :].detach())

            # forward pass
            deconv, inout_val, hx = model(frame_sequence_slice, depth_frame_sequence_slice, head_loc_sequence_slice, face_sequence_slice, \
                                                      left_eye_sequence_slice, right_eye_sequence_slice, hidden_scene=prev_hx, batch_sizes=X_pad_sizes_slice)
            for b_i in range(len(cont_gaze_sequence_slice)):
              if gt_inout_sequence_slice[b_i]: # ONLY for 'inside' cases
                # AUC: area under curve of ROC
                multi_hot = torch.zeros(64, 64)  # set the size of the output
                gaze_x = cont_gaze_sequence_slice[b_i, 0]
                gaze_y = cont_gaze_sequence_slice[b_i, 1]
                multi_hot = imutils.draw_labelmap(multi_hot, [gaze_x * 64, gaze_y * 64], 3, type='Gaussian')
                multi_hot = (multi_hot > 0).float() * 1 # make GT heatmap as binary labels
                multi_hot = misc.to_numpy(multi_hot)

                scaled_heatmap = imresize(deconv[b_i].squeeze().cpu(), (64, 64), interp = 'bilinear')
                auc_score = evaluation.auc(scaled_heatmap, multi_hot)
                AUC.append(auc_score)

                # distance: L2 distance between ground truth and argmax point
                pred_x, pred_y = evaluation.argmax_pts(deconv[b_i].squeeze().cpu())
                norm_p = [pred_x/64, pred_y/64]
                dist_score = evaluation.L2_dist(cont_gaze_sequence_slice[b_i].cpu(), norm_p).item()
                distance.append(dist_score)
            if np.isnan(inout_val.cpu().numpy()).any():
              continue
            in_vs_out_groundtruth.extend(gt_inout_sequence_slice.cpu().numpy())
            in_vs_out_pred.extend(inout_val.cpu().numpy())
            previous_hx_size = X_pad_sizes_slice[-1]
      
  return torch.mean(torch.tensor(AUC)), torch.mean(torch.tensor(distance)), evaluation.ap(in_vs_out_groundtruth, in_vs_out_pred)




def video_pack_sequences(in_batch):

    """
    Pad the variable-length input sequences to fixed length
    :param in_batch: the original input batch of sequences generated by pytorch DataLoader
    :return:
        out_batch (list): the padded batch of sequences
    """
    # Get the number of return values from __getitem__ in the Dataset
    num_returns = len(in_batch[0])

    # Sort the batch according to the sequence lengths. This is needed by torch func: pack_padded_sequences
    in_batch.sort(key=lambda x: -x[0].shape[0])
    shapes = [b[0].shape[0] for b in in_batch]

    # Determine the length of the padded inputs
    max_length = shapes[0]

    # Declare the output batch as a list
    out_batch = []
    # For each return value in each sequence, calculate the sequence-wise zero padding
    for r in range(num_returns):
        output_values = []
        lengths = []
        for seq in in_batch:
            values = seq[r]
            seq_size = values.shape[0]
            seq_shape = values.shape[1:]
            lengths.append(seq_size)
            padding = torch.zeros((max_length - seq_size, *seq_shape))
            padded_values = torch.cat((values, padding))
            output_values.append(padded_values)

        out_batch.append(torch.stack(output_values))
    out_batch.append(lengths)

    return out_batch

if __name__ == "__main__":
  if args.Dataset == "GazeFollow":
    runGazeFollow()
  else:
    runVideoAttTarget()
