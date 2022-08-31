import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
import math
from resnet import resnet18
from lib.pytorch_convolutional_rnn import convolutional_rnn
import numpy as np


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Static_Model(nn.Module):
    # Define a ResNet 50-ish arch
    def __init__(self, block = Bottleneck, layers_scene = [3, 4, 6, 3, 2], layers_face = [3, 4, 6, 3, 2], layers_depth = [3, 4, 6, 3, 2]):
        # Resnet Feature Extractor
        self.inplanes_scene = 64
        self.inplanes_face = 64
        self.inplanes_depth = 64
        super(Static_Model, self).__init__()
        # common
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # scene pathway
        self.conv1_scene = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_scene = nn.BatchNorm2d(64)
        self.layer1_scene = self._make_layer_scene(block, 64, layers_scene[0])
        self.layer2_scene = self._make_layer_scene(block, 128, layers_scene[1], stride=2)
        self.layer3_scene = self._make_layer_scene(block, 256, layers_scene[2], stride=2)
        self.layer4_scene = self._make_layer_scene(block, 512, layers_scene[3], stride=2)
        self.layer5_scene = self._make_layer_scene(block, 256, layers_scene[4], stride=1) # additional to resnet50


        # depth pathway
        self.conv1_depth = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_depth = nn.BatchNorm2d(64)
        self.layer1_depth = self._make_layer_depth(block, 64, layers_depth[0])
        self.layer2_depth = self._make_layer_depth(block, 128, layers_depth[1], stride=2)
        self.layer3_depth = self._make_layer_depth(block, 256, layers_depth[2], stride=2)
        self.layer4_depth = self._make_layer_depth(block, 512, layers_depth[3], stride=2)
        self.layer5_depth = self._make_layer_depth(block, 256, layers_depth[4], stride=1) # additional to resnet50
        self.fc_depth     = nn.Linear(1024, 784)
        # face pathway
        self.conv1_face = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1_face = nn.BatchNorm2d(64)
        self.layer1_face = self._make_layer_face(block, 64, layers_face[0])
        self.layer2_face = self._make_layer_face(block, 128, layers_face[1], stride=2)
        self.layer3_face = self._make_layer_face(block, 256, layers_face[2], stride=2)
        self.layer4_face = self._make_layer_face(block, 512, layers_face[3], stride=2)
        self.layer5_face = self._make_layer_face(block, 256, layers_face[4], stride=1) # additional to resnet50
      

        #eyes pathway 
        self.eyes_resnet = resnet18(pretrained=True)
        self.eyes_resnet.fc2 = nn.Linear(1000, 128)

        #fusion
        self.final_face = nn.Linear(1280, 1024)

        # attention
        self.attn = nn.Linear(1808, 1*7*7)
        self.attn2 = nn.Linear(2592, 1*7*7)

        self.FOV_features = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.FOV_features_bn1 = nn.BatchNorm2d(1024)

        # encoding for saliency
        self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        # encoding for in/out
        self.compress_conv1_inout = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1_inout = nn.BatchNorm2d(512)
        self.compress_conv2_inout = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2_inout = nn.BatchNorm2d(1)
        self.fc_inout = nn.Linear(49, 1)

        # decoding
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer_scene(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_scene != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_scene, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_scene, planes, stride, downsample))
        self.inplanes_scene = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_scene, planes))

        return nn.Sequential(*layers)

    def _make_layer_face(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_face != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_face, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_face, planes, stride, downsample))
        self.inplanes_face = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_face, planes))

        return nn.Sequential(*layers)


    def _make_layer_depth(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_depth != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_depth, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_depth, planes, stride, downsample))
        self.inplanes_depth = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_depth, planes))

        return nn.Sequential(*layers)


    def forward(self, images,depth_maps,head,face, left_eye, right_eye):

        face = self.conv1_face(face)
        face = self.bn1_face(face)
        face = self.relu(face)
        face = self.maxpool(face)
        face = self.layer1_face(face)
        face = self.layer2_face(face)
        face = self.layer3_face(face)
        face = self.layer4_face(face)
        face_feat = self.layer5_face(face)

        depth_im = torch.cat((depth_maps, head), dim=1)
        depth_im = self.conv1_depth(depth_im)
        depth_im = self.bn1_depth(depth_im)
        depth_im = self.relu(depth_im)
        depth_im = self.maxpool(depth_im)
        depth_im = self.layer1_depth(depth_im)
        depth_im = self.layer2_depth(depth_im)
        depth_im = self.layer3_depth(depth_im)
        depth_im = self.layer4_depth(depth_im)
        depth_feat = self.layer5_depth(depth_im)
        
        left_eye_feat = self.eyes_resnet(left_eye)
        right_eye_feat = self.eyes_resnet(right_eye)
  
        head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)
        face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)
        depth_feat_reduced = self.avgpool(depth_feat).view(-1, 1024)
        depth_feat_reduced  = self.fc_depth(depth_feat_reduced)

        base_face = torch.cat((face_feat_reduced, left_eye_feat, right_eye_feat), dim=1)
        base_face = self.final_face(base_face)


        attn_weights = self.attn2(torch.cat((head_reduced, base_face, depth_feat_reduced), 1))
        #attn_weights = self.attn(torch.cat((head_reduced, base_face), 1))
        attn_weights = attn_weights.view(-1, 1, 49)
        attn_weights = F.softmax(attn_weights, dim=2) 
        attn_weights = attn_weights.view(-1, 1, 7, 7)
        
        
        im = torch.cat((images, head), dim=1)
        im = self.conv1_scene(im)
        im = self.bn1_scene(im)
        im = self.relu(im)
        im = self.maxpool(im)
        im = self.layer1_scene(im)
        im = self.layer2_scene(im)
        im = self.layer3_scene(im)
        im = self.layer4_scene(im)
        scene_feat = self.layer5_scene(im)

        attn_applied_scene_feat = torch.mul(attn_weights, scene_feat) 
        scene_face_feat = self.FOV_features(torch.cat((depth_feat, face_feat), 1))
        scene_face_feat = self.FOV_features_bn1(scene_face_feat)
        scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)
        
        encoding_inout = self.compress_conv1_inout(scene_face_feat)
        encoding_inout = self.compress_bn1_inout(encoding_inout)
        encoding_inout = self.relu(encoding_inout)
        encoding_inout = self.compress_conv2_inout(encoding_inout)
        encoding_inout = self.compress_bn2_inout(encoding_inout)
        encoding_inout = self.relu(encoding_inout)
        encoding_inout = encoding_inout.view(-1, 49)
        encoding_inout = self.fc_inout(encoding_inout)


        encoding = self.compress_conv1(scene_face_feat)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)


        x = self.deconv1(encoding)
        x = self.deconv_bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.deconv_bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.deconv_bn3(x)
        x = self.relu(x)
        x = self.conv4(x)


        return x, encoding_inout

