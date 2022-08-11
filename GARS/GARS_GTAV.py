import sys
sys.path.append("../")
import ext
import numpy as np
import PIL
from PIL import Image
import os

import glob
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import shutil as shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="./GTAV_720_HLBIC_FR/LR/x4/")
parser.add_argument("--save_path", type=str, default="./GTAV_720_HLBIC_FR_MVRs_NVRs/LR/x4/")
parser.add_argument("--pad_width", type=int, default=16)
parser.add_argument("--reference_num", type=int, default=6)
parser.add_argument("--use_cuda", action='store_true', default=False)
args = parser.parse_args()


use_cuda = args.use_cuda

def pad(im, pad_width, scene_use_cuda):
    h, w = im.shape[-2:]
    mh = h % pad_width
    ph = 0 if mh == 0 else pad_width - mh
    mw = w % pad_width
    pw = 0 if mw == 0 else pad_width - mw
    shape = [s for s in im.shape]
    shape[-2] += ph
    shape[-1] += pw
    im_p = torch.zeros(shape).float()
    if scene_use_cuda == True:
        im_p = im_p.cuda()
    im_p[..., :h, :w] = im
    im = im_p
    return im

def load(p, height=None, width=None):
    if p.suffix == ".npy":
        return np.load(p)
    elif p.suffix in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
        im = PIL.Image.open(p)
        im = np.array(im)
        if (
            height is not None
            and width is not None
            and (im.shape[0] != height or im.shape[1] != width)
        ):
            raise Exception("invalid size of image")
        im = (im.astype(np.float32) / 255) * 2 - 1
        im = im.transpose(2, 0, 1)
        return im
    else:
        raise Exception("invalid suffix")

pad_width = args.pad_width
reference_num = args.reference_num
threshold = 0.95
k1, k2, L = 0.01, 0.03, 255
C1 = (k1 * L) ** 2
C2 = (k2 * L) ** 2
C3 = C2 / 2

path = args.path
save_path = args.save_path

subpaths = os.listdir(path)
subpaths.sort()
for subpath in subpaths:

    root = os.path.join(path, subpath, 'dense', 'ibr3d_pw_0.50')
    save_path_tmp = os.path.join(save_path, subpath, 'dense', 'ibr3d_pw_0.50')

    scene_use_cuda = True

    print(root)

    # copy gt and bic
    gt_root = root.replace('/LR/', '/HR/')
    bic_root = root.replace('/LR/', '/Bic/')
    files = os.listdir(gt_root)
    files.sort()
    for i, file in enumerate(files):
        save_path_gt = os.path.join(save_path_tmp, str(i).zfill(4))
        if not os.path.exists(save_path_gt):
            os.makedirs(save_path_gt)
        src_root = os.path.join(gt_root, file)
        tgt_root = os.path.join(save_path_gt, src_root[src_root.rfind('/') + 1:]).replace('/im_', '/hr_')
        shutil.copy(src_root, tgt_root)
    files = os.listdir(bic_root)
    files.sort()
    for i, file in enumerate(files):
        save_path_bic = os.path.join(save_path_tmp, str(i).zfill(4))
        if not os.path.exists(save_path_bic):
            os.makedirs(save_path_bic)
        src_root = os.path.join(bic_root, file)
        tgt_root = os.path.join(save_path_bic, src_root[src_root.rfind('/') + 1:]).replace('/im_', '/bic_')
        shutil.copy(src_root, tgt_root)


    #######
    #######
    #######
    dm_paths = sorted(glob.glob(os.path.join(root, 'dm*.npy')))
    png_paths = sorted(glob.glob(os.path.join(root, 'im_0*.png')))


    bic_paths = sorted(glob.glob(os.path.join(root.replace('LR', 'Bic'), 'im_0*.png')))
    Ks = np.load(os.path.join(root, 'Ks.npy'))
    Rs = np.load(os.path.join(root, 'Rs.npy'))
    ts = np.load(os.path.join(root, 'ts.npy'))
    src_dms = np.array([load(Path(ii)) for ii in dm_paths])
    src_dms = torch.from_numpy(src_dms).float()
    src_dms_clone = src_dms.clone()
    src_dms = torch.unsqueeze(src_dms, dim=1)
    src_dms = F.interpolate(src_dms, scale_factor=4, mode='bilinear')
    src_dms = torch.squeeze(src_dms)

    src_dms = src_dms.numpy()
    src_Ks = np.array(Ks)
    src_Ks[:, :2, :] = src_Ks[:, :2, :] * 4
    src_Rs = np.array(Rs)
    src_ts = np.array(ts)
    dms = []
    Ks = []
    Rs = []
    ts = []
    for val in range(src_dms.shape[0]):
        dms.append(src_dms[val])
        Ks.append(src_Ks[val])
        Rs.append(src_Rs[val])
        ts.append(src_ts[val])

    y = src_dms.shape[1]
    x = src_dms.shape[2]
    patch = np.array(
        (0, y, 0, x), dtype=np.int32
    )

    pngs = []
    for i in range(len(dms)):
        image_tmp = Image.open(png_paths[i])
        image_tmp = torch.from_numpy(np.array(image_tmp)).float()
        image_tmp = image_tmp.permute(2, 0, 1)
        pngs.append(image_tmp)
    pngs = torch.stack(pngs, dim=0).float()

    pngs = []
    pngs_gray = []
    pngs_bic_gray = []
    for i in range(len(dms)):
        image_tmp = Image.open(png_paths[i])
        image_tmp_gray = image_tmp.convert('L')
        image_tmp = torch.from_numpy(np.array(image_tmp)).float()
        image_tmp = image_tmp.permute(2, 0, 1)
        pngs.append(image_tmp)
        image_tmp_gray = torch.from_numpy(np.array(image_tmp_gray)).float()
        image_tmp_gray = torch.unsqueeze(image_tmp_gray, dim=0)
        pngs_gray.append(image_tmp_gray)

        image_tmp = Image.open(bic_paths[i])
        image_tmp = image_tmp.convert('L')
        image_tmp = torch.from_numpy(np.array(image_tmp)).float()
        image_tmp = torch.unsqueeze(image_tmp, dim=0)
        pngs_bic_gray.append(image_tmp)

    pngs = torch.stack(pngs, dim=0)
    pngs_bic = torch.stack(pngs_bic_gray, dim=0)
    pngs_gray = torch.stack(pngs_gray, dim=0)

    scatter_src = torch.tensor(1000.0).view(1, 1, 1, 1, 1).repeat(1, 1, 1, pngs.shape[2] * 4, pngs.shape[3] * 4)

    if scene_use_cuda == True:
        pngs = pngs.cuda()
        pngs_bic = pngs_bic.cuda()
        pngs_gray = pngs_gray.cuda()
        scatter_src = scatter_src.cuda()

    for i, dm in enumerate(tqdm(dm_paths, ncols=50)):
        # MVRs:
        save_path_ref = os.path.join(save_path_tmp, str(i).zfill(4))
        if not os.path.exists(save_path_ref):
            os.makedirs(save_path_ref)

        # get align index
        shutil.copy(png_paths[i], os.path.join(save_path_ref, png_paths[i][png_paths[i].rfind('/')+1:]))
        sampling_maps, valid_depth_masks, valid_map_masks = ext.preprocess.get_sampling_map(
            dms[i],
            Ks[i],
            Rs[i],
            ts[i],
            dms,
            Ks,
            Rs,
            ts,
            patch,  # patch,
            0.1,
            True,
        )

        sampling_maps = torch.from_numpy(sampling_maps).float()
        if scene_use_cuda == True:
            sampling_maps = sampling_maps.cuda()

        # align images
        align_pixel = F.grid_sample(
            pngs,
            sampling_maps,
            mode="bilinear",
            padding_mode="zeros",
        )
        align_pixel_clone = align_pixel.clone()

        # align gray images for luminance selection, little effect on the results
        # future work
        align_pixel_gray = F.grid_sample(
            pngs_gray,
            sampling_maps,
            mode="bilinear",
            padding_mode="zeros",
        )

        src_dms_torch = src_dms_clone.float()
        if scene_use_cuda == True:
            src_dms_torch = src_dms_torch.cuda()
        src_dms_torch = torch.unsqueeze(src_dms_torch, dim=1)
        # align depths
        align_depth = F.grid_sample(
            src_dms_torch,
            sampling_maps,
            mode="bilinear",
            padding_mode="zeros",
        )
        bs, _, h_0, w_0 = align_depth.shape
        align_depth = pad(align_depth, pad_width, scene_use_cuda)
        align_depth[align_depth == 0] = 1000
        align_pixel_gray = pad(align_pixel_gray, pad_width, scene_use_cuda)
        bic_image = pad(pngs_bic[i: i + 1].clone(), pad_width, scene_use_cuda)
        bic_image = bic_image.repeat(bs, 1, 1, 1)

        # depth-guided patch-selection strategy
        bs, _, h, w = align_depth.shape
        align_depth = F.unfold(align_depth, kernel_size=pad_width, stride=pad_width, padding=0)
        align_depth = align_depth.permute(0, 2, 1)
        align_depth = torch.mean(align_depth, dim=2, keepdim=True)
        align_depth = align_depth.repeat(1, 1, pad_width * pad_width).permute(0, 2, 1)
        align_depth = F.fold(align_depth, kernel_size=pad_width, padding=0, stride=pad_width, output_size=(h, w))
        align_depth = align_depth[:, :, :h_0, :w_0]


        # remove large luminance differences patch, little effect on the results
        # future work
        align_pixel_gray = F.unfold(align_pixel_gray, kernel_size=pad_width, stride=pad_width, padding=0)
        align_pixel_gray = align_pixel_gray.permute(0, 2, 1)
        align_pixel_gray_mean = torch.mean(align_pixel_gray, dim=2, keepdim=True)
        bic_image = F.unfold(bic_image, kernel_size=pad_width, stride=pad_width, padding=0)
        bic_image = bic_image.permute(0, 2, 1)
        bic_image_mean = torch.mean(bic_image, dim=2, keepdim=True)
        luminance_diff = (2 * align_pixel_gray_mean * bic_image_mean + C1) / (
                    align_pixel_gray_mean ** 2 + bic_image_mean ** 2 + C1)
        luminance_diff = luminance_diff.repeat(1, 1, pad_width * pad_width).permute(0, 2, 1)
        luminance_diff = F.fold(luminance_diff, kernel_size=pad_width, padding=0, stride=pad_width, output_size=(h, w))
        luminance_diff = luminance_diff[:, :, :h_0, :w_0]
        align_depth[luminance_diff < threshold] = 1000

        # get the high-frequency textural patch
        align_depth = torch.unsqueeze(align_depth, dim=0)
        for candidate in range(reference_num):
            num, index = torch.min(align_depth, dim=1)
            index_clone = index.clone()
            index = torch.unsqueeze(index, dim=2).repeat(1, 1, 3, 1, 1)
            multiview_view_reference = torch.unsqueeze(align_pixel_clone, dim=0)
            multiview_view_reference = torch.gather(multiview_view_reference, 1, index)
            multiview_view_reference = torch.squeeze(multiview_view_reference).permute(1, 2, 0)
            multiview_view_reference = PIL.Image.fromarray(multiview_view_reference.cpu().numpy().astype(np.uint8))
            multiview_view_reference.save(os.path.join(save_path_ref, 'refer_' + str(candidate).zfill(2) + '.png'))
            index = torch.unsqueeze(index_clone, dim=2)
            align_depth.scatter_(dim=1, index=index, src=scatter_src)

        # NVRs
        # we set the number of near view as 6
        if i - 3 < 0:
            nbs_min = 0
            nbs_max = i + 4 + abs(i-3)
        else:
            if i + 3 >= len(dms):
                nbs_max = len(dms)
                nbs_min = i - 3 - (i + 3 - (len(dms) - 1))
            else:
                nbs_min = i - 3
                nbs_max = i + 4

        nbs = [abs(j)+2 * min(len(dms) - j, 0)
               for j in range(i-3, i+4, 1)]
        sampling_maps, valid_depth_masks, valid_map_masks = ext.preprocess.get_sampling_map(
            dms[i],
            Ks[i],
            Rs[i],
            ts[i],
            dms[nbs_min:nbs_max],
            Ks[nbs_min:nbs_max],
            Rs[nbs_min:nbs_max],
            ts[nbs_min:nbs_max],
            patch,  # patch,
            0.1,
            True,
        )

        sampling_maps = torch.from_numpy(sampling_maps).permute(0, 3, 1, 2)
        if scene_use_cuda == True:
            sampling_maps = sampling_maps.cuda()
        sampling_maps = sampling_maps.permute(0, 2, 3, 1)

        align_pixel = F.grid_sample(
            pngs[nbs_min:nbs_max],
            sampling_maps.clone(),
            mode="bilinear",
            padding_mode="zeros",
        )
        align_pixel_clone = align_pixel.clone()

        align_pixel_gray = F.grid_sample(
            pngs_gray[nbs_min:nbs_max],
            sampling_maps,
            mode="bilinear",
            padding_mode="zeros",
        )

        src_dms_torch = src_dms_clone.float()
        if scene_use_cuda == True:
            src_dms_torch = src_dms_torch.cuda()
        src_dms_torch = torch.unsqueeze(src_dms_torch, dim=1)

        align_depth = F.grid_sample(
            src_dms_torch[nbs_min:nbs_max],
            sampling_maps.clone(),
            mode="bilinear",
            padding_mode="zeros",
        )

        bs, _, h_0, w_0 = align_depth.shape
        align_depth = pad(align_depth, pad_width, scene_use_cuda)
        align_depth[align_depth == 0] = 1000
        align_pixel_gray = pad(align_pixel_gray, pad_width, scene_use_cuda)
        bic_image = pad(pngs_bic[i: i + 1].clone(), pad_width, scene_use_cuda)
        bic_image = bic_image.repeat(bs, 1, 1, 1)

        bs, _, h, w = align_depth.shape
        align_depth = F.unfold(align_depth, kernel_size=pad_width, stride=pad_width, padding=0)
        align_depth = align_depth.permute(0, 2, 1)
        align_depth = torch.mean(align_depth, dim=2, keepdim=True)
        align_depth = align_depth.repeat(1, 1, pad_width * pad_width).permute(0, 2, 1)
        align_depth = F.fold(align_depth, kernel_size=pad_width, padding=0, stride=pad_width, output_size=(h, w))
        align_depth = align_depth[:, :, :h_0, :w_0]

        align_pixel_gray = F.unfold(align_pixel_gray, kernel_size=pad_width, stride=pad_width, padding=0)
        align_pixel_gray = align_pixel_gray.permute(0, 2, 1)
        align_pixel_gray_mean = torch.mean(align_pixel_gray, dim=2, keepdim=True)
        bic_image = F.unfold(bic_image, kernel_size=pad_width, stride=pad_width, padding=0)
        bic_image = bic_image.permute(0, 2, 1)
        bic_image_mean = torch.mean(bic_image, dim=2, keepdim=True)
        l12 = (2 * align_pixel_gray_mean * bic_image_mean + C1) / (
                align_pixel_gray_mean ** 2 + bic_image_mean ** 2 + C1)
        ssim_result = l12
        ssim_result = ssim_result.repeat(1, 1, pad_width * pad_width).permute(0, 2, 1)
        ssim_result = F.fold(ssim_result, kernel_size=pad_width, padding=0, stride=pad_width, output_size=(h, w))
        ssim_result = ssim_result[:, :, :h_0, :w_0]
        align_depth[ssim_result < threshold] = 1000

        align_depth = torch.unsqueeze(align_depth, dim=0)
        for candidate in range(6):
            num, index = torch.min(align_depth, dim=1)
            index_clone = index.clone()
            index = torch.unsqueeze(index, dim=2).repeat(1, 1, 3, 1, 1)
            multiview_view_reference = torch.unsqueeze(align_pixel_clone, dim=0)
            multiview_view_reference = torch.gather(multiview_view_reference, 1, index)
            multiview_view_reference = torch.squeeze(multiview_view_reference).permute(1, 2, 0)
            multiview_view_reference = PIL.Image.fromarray(multiview_view_reference.cpu().numpy().astype(np.uint8))
            multiview_view_reference.save(os.path.join(save_path_ref, 'refer_near_' + str(candidate).zfill(2) + '.png'))
            index = torch.unsqueeze(index_clone, dim=2)
            align_depth.scatter_(dim=1, index=index, src=scatter_src)


