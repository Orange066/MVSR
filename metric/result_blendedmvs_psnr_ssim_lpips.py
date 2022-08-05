import os
from PIL import Image
import cv2
import math
import numpy as np
import shutil as shutil

import torch
import lpips
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--result_path", type=str, default="./")
parser.add_argument("--iter", type=int, default=139999)
parser.add_argument("--save_every", action='store_true', default=False)
args = parser.parse_args()

loss_fn_alex = lpips.LPIPS(net='alex')
use_cuda = True
if use_cuda ==True:
    loss_fn_alex = loss_fn_alex.cuda()


paths = [
    '../exp/experiments/backbone_perceptual_gan_ref_6_patch_[80, 160]_complement/',
         ]
gt_path = '../GARS/BlendedMVS_HLBIC_FR_MVRs_NVRs/LR/x4/'
datasets = os.listdir(gt_path)
datasets.sort()
datasets_renames = []
for dataset in datasets:
    datasets_renames.append('tat_all_' + dataset + '_0.5_n6')

save_paths = [
    '../exp/experiments/backbone_perceptual_gan_ref_6_patch_[80, 160]_complement/',
         ]
iter =  [ str(args.iter), ]
sr_names = [ 'BlendedMVS', ]
save_every = args.save_every

if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)
file_txt=open(os.path.join(args.result_path, 'blendedmvs.txt'),mode='w')

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

total_str_rgb = ''
total_str_y = ''

for path_count, path in enumerate(paths):

    file_txt.write('#' * 80 + '\n')
    file_txt.write(path + '\n')

    total_psnr_rgb =[]
    total_ssim_rgb = []
    total_psnr_y = []
    total_ssim_y = []
    total_lpips = []

    str_rgb = sr_names[path_count]
    str_y = sr_names[path_count]
    str_rgb_sr = sr_names[path_count]
    str_y_sr = sr_names[path_count]
    str_rgb_ip = sr_names[path_count]
    str_y_ip = sr_names[path_count]

    for dataset_count, dataset in enumerate(datasets):
        tmp_total_count = 0.0
        tmp_total_psnr_rgb = 0.0
        tmp_total_ssim_rgb = 0.0
        tmp_total_psnr_y = 0.0
        tmp_total_ssim_y = 0.0
        tmp_total_lpips = 0.0

        sub_savepath = os.path.join(save_paths[path_count], iter[path_count], dataset)
        if not os.path.exists(sub_savepath):
            os.makedirs(sub_savepath)

        image_l = []
        subdir = os.path.join(path, datasets_renames[dataset_count], iter[path_count])
        print('subdir', subdir)
        subsubdirs = os.listdir(subdir)
        subsubdirs.sort()
        for subsubdir in subsubdirs:
            image_l.append(os.path.join(subdir, subsubdir, 's0000_es.png'))


        gt_l = []
        subdir = os.path.join(gt_path, datasets[dataset_count], 'dense', 'ibr3d_pw_0.50')
        subsubdirs = os.listdir(subdir)
        subsubdirs.sort()
        for subsubdir_count, subsubdir in enumerate(subsubdirs):
            gt_l.append(os.path.join(subdir, subsubdir, 'hr_' + str(subsubdir_count).zfill(8) + '.png'))

        for i, gt_image_path in enumerate(gt_l):
            gt_image = Image.open(gt_image_path)
            sr_image = Image.open(image_l[i])

            copy_path = sub_savepath + '/' + str(i).zfill(4) + '.png'
            shutil.copy(image_l[i], copy_path)

            gt_image = np.array(gt_image)
            sr_image = np.array(sr_image)

            psnr_rgb = calculate_psnr(sr_image, gt_image)
            ssim_rgb = calculate_ssim(sr_image, gt_image)
            gt_image = rgb2ycbcr(gt_image, only_y=True)
            sr_image = rgb2ycbcr(sr_image, only_y=True)
            psnr_y = calculate_psnr(sr_image, gt_image)
            ssim_y = calculate_ssim(sr_image, gt_image)

            # lpips
            gt_image = Image.open(gt_image_path)
            sr_image = Image.open(image_l[i])
            gt_image = np.array(gt_image)
            sr_image = np.array(sr_image)
            gt_image = torch.unsqueeze(torch.from_numpy(gt_image).permute(2, 0, 1), dim=0)
            sr_image = torch.unsqueeze(torch.from_numpy(sr_image).permute(2, 0, 1), dim=0)
            gt_image = (gt_image / 255.) * 2 - 1
            sr_image = (sr_image / 255.) * 2 - 1
            if use_cuda == True:
                gt_image = gt_image.cuda().float()
                sr_image = sr_image.cuda().float()
            else:
                gt_image = gt_image.float()
                sr_image = sr_image.float()
            with torch.no_grad():
                lpips = loss_fn_alex(sr_image, gt_image)
                lpips = float(lpips.cpu())

            tmp_total_count = tmp_total_count + 1
            tmp_total_psnr_rgb = tmp_total_psnr_rgb + psnr_rgb
            tmp_total_ssim_rgb = tmp_total_ssim_rgb + ssim_rgb
            tmp_total_psnr_y = tmp_total_psnr_y + psnr_y
            tmp_total_ssim_y = tmp_total_ssim_y + ssim_y
            tmp_total_lpips = tmp_total_lpips + lpips

            if save_every == True:
                file_txt.write(image_l[i] + '\n')
                file_txt.write('psnr_rgb:' + str(psnr_rgb) + '\nssim_rgb:' + str(ssim_rgb) + '\npsnr_y:' + str(
                    psnr_y) + '\nssim_y:' + str(ssim_y) + '\nlpips:' + str(lpips) + '\n')
            print(image_l[i])
            print('psnr_rgb:', psnr_rgb)
            print('ssim_rgb:', ssim_rgb)
            print('psnr_y:', psnr_y)
            print('ssim_y:', ssim_y)
            print('lpips:', lpips)

        total_psnr_rgb.append(round(tmp_total_psnr_rgb / tmp_total_count, 2))
        total_ssim_rgb.append(round(tmp_total_ssim_rgb / tmp_total_count, 4))
        total_psnr_y.append(round(tmp_total_psnr_y / tmp_total_count, 2))
        total_ssim_y.append(round(tmp_total_ssim_y / tmp_total_count, 4))
        total_lpips.append(round(tmp_total_lpips / tmp_total_count, 3))

        print('*' * 80)
        print(dataset)
        print('psnr_rgb:', round(tmp_total_psnr_rgb / tmp_total_count, 2))
        print('ssim_rgb:', round(tmp_total_ssim_rgb / tmp_total_count, 4))
        print('psnr_y:', round(tmp_total_psnr_y / tmp_total_count, 2))
        print('ssim_y:', round(tmp_total_ssim_y / tmp_total_count, 4))
        print('lpips:', round(tmp_total_lpips / tmp_total_count, 3))

        str_rgb = str_rgb + ' & ' + str(round(tmp_total_psnr_rgb / tmp_total_count, 2))
        str_rgb = str_rgb + ' / ' + str(round(tmp_total_ssim_rgb / tmp_total_count, 4))
        str_rgb = str_rgb + ' / ' + str(round(tmp_total_lpips / tmp_total_count, 3))
        str_y = str_y + ' & ' + str(round(tmp_total_psnr_y / tmp_total_count, 2))
        str_y = str_y + ' / ' + str(round(tmp_total_ssim_y / tmp_total_count, 4))
        str_y = str_y + ' / ' + str(round(tmp_total_lpips / tmp_total_count, 3))

        file_txt.write('*' * 80 + '\n')
        file_txt.write(dataset + '\n')
        file_txt.write('psnr_rgb :' + str(tmp_total_psnr_rgb / tmp_total_count) + '\n')
        file_txt.write('ssim_rgb :' + str(tmp_total_ssim_rgb / tmp_total_count) + '\n')
        file_txt.write('psnr_y :' + str(tmp_total_psnr_y / tmp_total_count) + '\n')
        file_txt.write('ssim_y :' + str(tmp_total_ssim_y / tmp_total_count) + '\n')
        file_txt.write('ssim_y :' + str(tmp_total_lpips / tmp_total_count) + '\n')

    str_rgb = str_rgb + ' & ' + str(round(sum(total_psnr_rgb) / len(datasets), 2))
    str_rgb = str_rgb + ' / ' + str(round(sum(total_ssim_rgb) / len(datasets), 4))
    str_rgb = str_rgb + ' / ' + str(round(sum(total_lpips) / len(datasets), 3))
    str_y = str_y + ' & ' + str(round(sum(total_psnr_y) / len(datasets), 2))
    str_y = str_y + ' / ' + str(round(sum(total_ssim_y) / len(datasets), 4))
    str_y = str_y + ' / ' + str(round(sum(total_lpips) / len(datasets), 3))

    str_rgb = str_rgb + ' \\\\ \n'
    str_y = str_y + ' \\\\ \n'

    total_str_rgb = total_str_rgb + str_rgb
    total_str_y = total_str_y + str_y

    print('*' * 80)
    print('total result')
    print('psnr_rgb:', str(round(sum(total_psnr_rgb) / len(datasets), 2)))
    print('ssim_rgb:', str(round(sum(total_ssim_rgb) / len(datasets), 4)))
    print('psnr_y:', str(round(sum(total_psnr_y) / len(datasets), 2)))
    print('ssim_y:', str(round(sum(total_ssim_y) / len(datasets), 4)))
    print('lpips:', str(round(sum(total_lpips) / len(datasets), 3)))

    file_txt.write('*' * 80 + '\n')
    file_txt.write('total result' + '\n')
    file_txt.write('psnr_rgb :' + str(round(sum(total_psnr_rgb) / len(datasets), 2)) + '\n')
    file_txt.write('ssim_rgb :' + str(round(sum(total_ssim_rgb) / len(datasets), 4)) + '\n')
    file_txt.write('psnr_y :' + str(round(sum(total_psnr_y) / len(datasets), 2)) + '\n')
    file_txt.write('ssim_y :' + str(round(sum(total_ssim_y) / len(datasets), 4)) + '\n')
    file_txt.write('lpips :' + str(round(sum(total_lpips) / len(datasets), 3)) + '\n')

file_txt.write('*' * 80 + '\n')
file_txt.write('total_str_rgb' + '\n')
file_txt.write(total_str_rgb)

file_txt.write('*' * 80 + '\n')
file_txt.write('total_str_y' + '\n')
file_txt.write(total_str_y)

file_txt.close()