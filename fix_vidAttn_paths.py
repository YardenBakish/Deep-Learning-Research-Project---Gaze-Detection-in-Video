import shutil
import subprocess
import os

train_dir = "dataset/videoattentiontarget/annotations/train"
test_dir = "dataset/videoattentiontarget/annotations/test"
img_dir = "dataset/videoattentiontarget/images"
for dir1 in os.listdir(train_dir):
    dir1_x = dir1.replace(" ", "_")
    dir1_x = dir1_x.replace("'", "")
    os.rename(os.path.join(train_dir,dir1), os.path.join(train_dir,dir1_x))

  
for dir1 in os.listdir(test_dir):
    dir1_x = dir1.replace(" ", "_")
    dir1_x = dir1_x.replace("'", "")
    os.rename(os.path.join(test_dir,dir1), os.path.join(test_dir,dir1_x))


for dir1 in os.listdir(img_dir):
    dir1_x = dir1.replace(" ", "_")
    dir1_x = dir1_x.replace("'", "")
    os.rename(os.path.join(img_dir,dir1), os.path.join(img_dir,dir1_x))