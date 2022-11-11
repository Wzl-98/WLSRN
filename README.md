# WLSRN:Lightweight Super-Resolution Network for SEM Image Based on Wavelet Transform
# Introduction
#### We propose a general approach to lightweight Super Resolution (SR) networks which is processed in two steps:<br>1) We propose Wavelet upscale based on inverse discrete wavelet transform (IDWT) for SR network, then we replace the original upscale method of SR network by Wavelet upscale and regard it as teacher network.<br>2) We replace the feature extraction module of teacher network by lightweight feature extraction module and regard it as student network.<br>3) We propose Wavelet distillation, the training process of student network is guided by teacher network through Wavelet distillation.<br><br>In this work, we apply this general approach to Residual Channel Attention Networks and obtain WLSRN.
![The relationship between RCAN, teacher network (TN) and WLSRN.](https://github.com/Wzl-98/WLSRN/tree/main/Fig/Fig1.png)<br> ![The framework of Wavelet upscale for grayscale images.](https://github.com/Wzl-98/WLSRN/tree/main/Fig/Fig2.png) 
#### Results on SEM dataset:
![Trade-off between performance (PSNR) vs. network weight (FLOPs and parameters) on SEM images (1280 × 960 pixels). The x-axis and y-axis denote the FLOPs and PSNR, the size of the circle represents the number of parameters.](https://github.com/Wzl-98/WLSRN/tree/main/Fig/Fig3.png) <br>
# Code
## Teacher network
### Environment
#### --python 3.6.0<br>--cuda 10.1<br>--torch 0.4.1
### Train
#### --cd Teacher/code<br><br>--python main.py --model Teacher --save Teacher --scale 2 --n_resgroups 10 --n_resblocks 20 --n_feats 64 --reset --save_results --print_model --patch_size 192 --ext img --n_colors 1 --batch_size 16 --epochs 1000
## Student network (WLSRN)
### Environment
#### --python 3.6.9<br>--cuda 10.1<br>--torch 1.1.0
### Train
#### --cd WLSRN/code<br><br>--python train.py --ckp_dir WLSRN_train --scale 2 --teacher [Teacher] --model WLSRN --alpha 0.7 --epochs 1000 --save_results --save_gt --ext img --n_colors 1 --batch_size 16 --patch_size 192
# Acknowledgement
#### With greatly appreciation for [RCAN](https://github.com/yulunzhang/RCAN) and [Knowledge-Distillation-for-Super-resolution](https://github.com/Vincent-Hoo/Knowledge-Distillation-for-Super-resolution)
# Citation
#### Coming soon.
