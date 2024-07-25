This is pytorch implementation of P2PNet paper(https://arxiv.org/abs/2107.12746)
<br>
To see the process, visit https://colab.research.google.com/drive/1L7JG72JShqbNJV8PY9c1BwK-U2mWd_uS#scrollTo=yVRQMTk2kIcW
<br>
<br>
For more information please refer to https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet
<br>
<br>
## Intro
Based on https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet
- changed preprocessing // preprocess.py

## Installation
```
$ git clone https://github.com/standfsk/Pytorch_P2PNet.git
$ cd Pytorch_P2PNet
$ pip install -r requirements.txt
```

## Dataset
- NWPU(https://gjy3035.github.io/NWPU-Crowd-Sample-Code/)
- QNRF(https://www.crcv.ucf.edu/data/ucf-qnrf/)
- ShanghaiTech Part A/B(https://github.com/desenzhou/ShanghaiTechDataset)
- UCF-CC(https://www.crcv.ucf.edu/data/ucf-cc-50/UCFCrowdCountingDataset_CVPR13.rar)

## Preprocess
```
$ mkdir data
$ mkdir ckpt
$ python preprocess.py --dataset UCF-CC_50
```

## Train
```
$ python train.py --dataset UCF_CC_50 --save_path ckpt/ucfcc --batch-size 6 --max-epoch 100 --gpu_id 0
```

## Test
```
$ python test.py --weight_path ckpt/ucfcc/best_model.pth --dataset ucfcc --gpu_id 0
```

## Demo
```
$ python video_demo.py --weight_path ckpt/ucfcc/best_model.pth --video_name sample.mp4 --gpu_id 0
```


