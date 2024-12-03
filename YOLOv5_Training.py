!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt

import torch
print("GPU disponível:", torch.cuda.is_available())

from google.colab import files
import zipfile

uploaded = files.upload()
for filename in uploaded.keys():
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data/')

data_yaml = """
train: data/images/train
val: data/images/val
nc: 2
names: ['classe1', 'classe2']
"""
with open('data.yaml', 'w') as f:
    f.write(data_yaml)

!python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --cfg yolov5s.yaml --weights yolov5s.pt
!python val.py --weights runs/train/exp/weights/best.pt --data data.yaml --img 640
!python export.py --weights runs/train/exp/weights/best.pt --img 640 --batch-size 1
