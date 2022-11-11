# WLSRN
Lightweight Super-Resolution Network for SEM Image Based on Wavelet Transform
#Introduction
##We propose a general approach to lightweight Super Resolution (SR) networks which is processed in two steps:<br>
1) We propose Wavelet upscale based on inverse discrete wavelet transform (IDWT) for SR network, then we replace the original upscale method of SR network by Wavelet upscale and regard it as teacher network.<br>
2) We replace the feature extraction module of teacher network by lightweight feature extraction module and regard it as student network.<br>
3) We propose Wavelet distillation, the training process of student network is guided by teacher network through Wavelet distillation.<br>
In this work, we apply this general approach to Residual Channel Attention Networks and obtain WLSRN. 
