# Geometry-Aware Reference Synthesis for Multi-View Image Super-Resolution

This is the official implementation of the paper [Geometry-Aware Reference Synthesis for Multi-View Image Super-Resolution](https://arxiv.org/abs/2207.08601), ACM MM 2022.


## Setup
We use NVIDIA RTX 3090, cuda 11.1 and the follwing python packages:
* pytorch=1.9.0
* torchvision=0.10.0
* pillow=8.3.1

Our code borrows from [FreeViewSynthesis](https://github.com/isl-org/FreeViewSynthesis), please build the Python extension needed for preprecessing,

```
cd ext/preprocess
cmake -DCMAKE_BUILD_TYPE=Release .
make 
```

## Run Geometry-Aware Reference Synthesis (GARS) module to synthesize references
We preprocess and reorganize the folder structure for three public datasets, [Tanks and Temples](https://github.com/isl-org/FreeViewSynthesis), [BlendedMVS](https://github.com/YoYo000/BlendedMVS), and [GTAV](https://phuang17.github.io/DeepMVS/mvs-synth.html), for training and evaluation.
In addition, we provide the preprocessing code for [Tanks and Temples](https://github.com/isl-org/FreeViewSynthesis) datasets in [preprocess_Tanks_Temples.py](GARS/preprocess_Tanks_Temples.py).
You can download our preprocessed version from [baidudisk](https://pan.baidu.com/s/1_mhkSEkliNDfriJ_rqQuTQ), code: ey16, and place them in the [GARS](GARS) folder. 
Then run:
```
cd GARS

# Synthesize references for Tanks and Temples:
bash GARS_Tanks_Temples.sh  // The CPU processes the Courthouse, Palace, and Church scenes due to a large number of images.

# Synthesize references for BlendedMVS:
bash GARS_BlendedMVS.sh  

# Synthesize references for GTAV:
bash GARS_GTAV.sh  
```
Finallly, we will get Tanks_and_Temples_HLBIC_FR_MVRs_NVRs, BlendedMVS_HLBIC_FR_MVRs_NVRs, and GTAV_720_HLBIC_FR_MVRs_NVRs floders in the [GARS](GARS) folder. You can directly download the GARS results from [baidudisk](https://pan.baidu.com/s/1wyjYn8zBUF6uL_hyxAVu-A), code: aq9t.

## Train Dynamic High-Frequency Search (DHFS) network
You can prepare dataset and updapte config.py with your own path.
```
cd exp

# Training the GAN-Based model, set use_perceptual_gan_loss=True in config.py
CUDA_VISIBLE_DEVICES=0 python exp.py --cmd retrain

# Training the Pixel-Based model, set use_perceptual_gan_loss=False in config.py
CUDA_VISIBLE_DEVICES=0 python exp.py --cmd retrain
```

## Test
You can download our pretrained GAN-based model [baidudisk](https://pan.baidu.com/s/1kctbqTaAL4n_OjcNqzFB4A) (ctrq) and Pixel-based model [baidudisk](https://pan.baidu.com/s/1uE2fYQvHtMfpQN5fCGf_UA) (jha1), and place them in [exp/experiments](exp/experiments).
Then run:
```
cd exp

# Test the GAN-Based model, set use_perceptual_gan_loss=True in config.py
# Test the Pixel-Based model, set use_perceptual_gan_loss=False in config.py

# for Tanks and Temples, set blendedmvs = False and gtav = False in config.py
CUDA_VISIBLE_DEVICES=0 python exp.py --cmd eval --iter 139999

# for BlendedMVS, set blendedmvs = True and gtav = False in config.py
CUDA_VISIBLE_DEVICES=0 python exp.py --cmd eval --iter 139999

# for GTAV, set blendedmvs = False and gtav = True in config.py
CUDA_VISIBLE_DEVICES=0 python exp.py --cmd eval --iter 129999
```

## Metrics
We provide the code to calculate PSNR, SSIM and LPIPS. Remember to update your own path.
```
cd metric

# for Tanks and Temples:
CUDA_VISIBLE_DEVICES=0 python result_tanks_psnr_ssim_lpips.py

# for BlendedMVS:
CUDA_VISIBLE_DEVICES=0 python result_blendedmvs_psnr_ssim_lpips.py

# for GTAV:
CUDA_VISIBLE_DEVICES=0 python result_gtav_psnr_ssim_lpips.py
```

## Citation
If you find this code useful, please cite our paper:
```
@misc{https://doi.org/10.48550/arxiv.2207.08601,
  title = {Geometry-Aware Reference Synthesis for Multi-View Image Super-Resolution},
  author = {Cheng, Ri and Sun, Yuqi and Yan, Bo and Tan, Weimin and Ma, Chenxi},
  publisher = {arXiv},
  year = {2022},
}
```

## Credit
Our code borrows from [FreeViewSynthesis](https://github.com/isl-org/FreeViewSynthesis), [RCAN](https://github.com/yulunzhang/RCAN), and [C2-Matching
](https://github.com/yumingj/C2-Matching).
