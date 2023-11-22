# ESDGUI-Update
<!--
 * @Author: mikami520 yxiao39@jh.edu
 * @Date: 2023-06-29 14:42:52
 * @LastEditors: mikami520 yxiao39@jh.edu
 * @LastEditTime: 2023-06-29 16:23:44
 * @FilePath: /ESDGUI-Update/README.md
 * @Description: README for ESDGUI
 * I Love IU
 * Copyright (c) 2023 by mikami520 yxiao39@jh.edu, All Rights Reserved. 
-->

## Installation
(1) create environment
``` 
git clone https://github.com/mikami520/ESDGUI-Update.git
cd ESDGUI-Update
conda env create -f gui.yml
conda activate ESDSafety
cd ..
```
(2) install dependencies
```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
cd ../ESDGUI-Update
```
## Usage
There are two main functions of this GUI work - ```video SAM segmentation (only)``` && ```integration of phase recognition, segmentation, annotation```
```
python canvas-video.py (video segmentation only)
```
or
```
python gui.py
```
## Pretained Models
#### Customized Models: [trained models](https://drive.google.com/drive/folders/1XcjfQ6Ced6L-XZc4KbOjVqu2c0VD3782?usp=sharing)

#### SAM Models:
- H-checkpoint: [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
    
    H checkpoint has best effect, but need more resources.VRAM needs at least 8G.
- L-checkpoint: [sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
    
    L checkpoint has normal effect and normal resources.VRAM needs at least 7G.
- B-checkpoint: [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
    
    B checkpoint has pool effect, but need less resources.VRAM needs at least 6G.

After downloading the models, place them in the **```ESDGUI-Update/runs```** folder. By now, only phase recognition and SAM inference are supported
