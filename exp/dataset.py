from random import randrange
import os
import math
import random
import torch
import torch.nn.functional as F
import numpy as np
import PIL
import logging
import sys
from pathlib import Path
import cv2

sys.path.append("../")
import ext
import co
import co.utils as co_utils
import config

meshGrids = {}

class Dataset(co.mytorch.BaseDataset):
    def __init__(
        self,
        *,
        name,
        im_size=None,
        pad_width=None,
        patch=None,
        image_dir=None,
        ref_num=None,
        scale = None,
        use_complement=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)


        self.im_size = im_size
        self.pad_width = pad_width
        self.patch = patch
        self.ref_num = ref_num

        self.bic_l = []
        self.hr_l = []
        self.lr_l = []
        self.refer_l = []
        self.refer_near_l = []
        self.image_dir = image_dir
        sub_dirs = os.listdir(self.image_dir)
        sub_dirs.sort()
        for i, sub_dir in enumerate(sub_dirs):
            self.bic_l.append(Path(os.path.join(self.image_dir, sub_dir, 'bic_' + str(i).zfill(8) + '.png')))
            self.hr_l.append(Path(os.path.join(self.image_dir, sub_dir, 'hr_' + str(i).zfill(8) + '.png')))
            self.lr_l.append(Path(os.path.join(self.image_dir, sub_dir, 'im_' + str(i).zfill(8) + '.png')))
            for j in range(self.ref_num):
                self.refer_l.append(Path(os.path.join(self.image_dir, sub_dir, 'refer_'+ str(j).zfill(2) +'.png')))
                self.refer_near_l.append(Path(os.path.join(self.image_dir, sub_dir, 'refer_near_'+ str(j).zfill(2) +'.png')))

        n_tgt_im_paths = len(self.lr_l) if self.lr_l else 0
        shape_tgt_im = (
            self.load_pad(self.lr_l[0]).shape if self.lr_l else None
        )
        logging.info(
            f"    #tgt_im_paths={n_tgt_im_paths}, # tgt_im={shape_tgt_im}"
        )

        self.len = len(self.lr_l)
        self.count = 0
        self.use_complement = use_complement
        self.scale = scale


    def load_data(self, p):
        if p.suffix == ".npy":
            npy = np.load(p)
            return npy
        elif p.suffix in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
            im = PIL.Image.open(p)
            im = np.array(im)

            im = (im.astype(np.float32) / 255) * 2 - 1
            im = im.transpose(2, 0, 1)
            return im
        else:
            raise Exception("invalid suffix")

    def pad(self, im, hr=False, h=0, w=0):
        if self.im_size is not None or h != 0:
            shape = [s for s in im.shape]
            if hr == False:
                shape[-2] = self.im_size[0]
                shape[-1] = self.im_size[1]
            elif hr == True:
                shape[-2] = h
                shape[-1] = w
            im_p = np.zeros(shape, dtype=im.dtype)
            sh = min(im_p.shape[-2], im.shape[-2])
            sw = min(im_p.shape[-1], im.shape[-1])
            im_p[..., :sh, :sw] = im[..., :sh, :sw]
            im = im_p
        if self.pad_width is not None:
            h, w = im.shape[-2:]
            mh = h % self.pad_width
            ph = 0 if mh == 0 else self.pad_width - mh
            mw = w % self.pad_width
            pw = 0 if mw == 0 else self.pad_width - mw
            shape = [s for s in im.shape]
            shape[-2] += ph
            shape[-1] += pw
            im_p = np.zeros(shape, dtype=im.dtype)
            im_p[..., :h, :w] = im
            im = im_p
        return im

    def load_pad(self, p, hr=False, h=0, w=0):
        im = self.load_data(p)
        return self.pad(im, hr=hr, h=h, w=w)

    def base_len(self):
        return len(self.lr_l)

    def base_getitem(self, idx, rng):
        ret = {}

        img_lr = self.load_data(self.lr_l[idx])
        if self.train == False:
            ret["HR_size"] = np.array((img_lr.shape[1] * 4, img_lr.shape[2] * 4), dtype=np.int32)
        img_lr = self.pad(img_lr, hr=False)
        y = img_lr.shape[1]
        x = img_lr.shape[2]
        del img_lr

        if self.patch:
            patch_h_from = rng.randint(0, y - self.patch[0])
            patch_w_from = rng.randint(0, x - self.patch[1])
            patch_h_to = patch_h_from + self.patch[0]
            patch_w_to = patch_w_from + self.patch[1]
            patch = np.array(
                (patch_h_from, patch_h_to, patch_w_from, patch_w_to),
                dtype=np.int32,
            )
        else:
            patch = np.array(
                (0, y, 0, x), dtype=np.int32
            )

        img_lr = self.load_pad(self.lr_l[idx], hr=False)
        img_hr = self.load_pad(self.hr_l[idx], hr=True, h=y * self.scale, w=x * self.scale)
        img_bic = self.load_pad(self.bic_l[idx], hr=True, h=y * self.scale, w=x * self.scale)
        img_lr = img_lr[:, patch[0]: patch[1], patch[2]: patch[3]]
        img_hr = img_hr[:, patch[0] * self.scale: patch[1] * self.scale, patch[2] * self.scale: patch[3] * self.scale]
        img_bic = img_bic[:, patch[0] * self.scale: patch[1] * self.scale, patch[2] * self.scale: patch[3] * self.scale]
        img_ref_multiview = []
        img_ref_near = []
        # print(self.lr_l[idx])
        for i in range(idx * self.ref_num, idx * self.ref_num + self.ref_num):
            ref_tmp = self.load_pad(self.refer_l[i], hr=True, h=y * self.scale, w=x * self.scale)
            ref_tmp = ref_tmp[:, patch[0] * self.scale: patch[1] * self.scale,
                     patch[2] * self.scale: patch[3] * self.scale]
            img_ref_multiview.append(ref_tmp)

        for i in range(idx * self.ref_num, idx * self.ref_num + self.ref_num):
            ref_tmp = self.load_pad(self.refer_near_l[i], hr=True, h=y * self.scale, w=x * self.scale)
            ref_tmp = ref_tmp[:, patch[0] * self.scale: patch[1] * self.scale,
                     patch[2] * self.scale: patch[3] * self.scale]
            img_ref_near.append(ref_tmp)

        if self.use_complement == True:
            valid_map = np.zeros([1, img_bic.shape[1], img_bic.shape[2]])
            valid_map[(img_ref_multiview[-1][0:1, :, :] == -1) & (img_ref_multiview[-1][1:2, :, :] == -1) & (img_ref_multiview[-1][2:3, :, :] == -1)] = 1
            img_ref_multiview[-1] = img_ref_multiview[-1] + valid_map + img_bic * valid_map
            if self.ref_num - 2 >= 0:
                for i in range(self.ref_num-2, -1 , -1):
                    valid_map = np.zeros([1, img_bic.shape[1], img_bic.shape[2]])
                    valid_map[(img_ref_multiview[i][0:1, :, :] == -1) & (img_ref_multiview[i][1:2, :, :] == -1) & (img_ref_multiview[i][2:3, :, :] == -1)] = 1
                    img_ref_multiview[i] = img_ref_multiview[i] + valid_map + img_ref_multiview[i+1] * valid_map

            valid_map = np.zeros([1, img_bic.shape[1], img_bic.shape[2]])
            valid_map[(img_ref_near[-1][0:1, :, :] == -1) & (img_ref_near[-1][1:2, :, :] == -1) & (img_ref_near[-1][2:3, :, :] == -1)] = 1
            img_ref_near[-1] = img_ref_near[-1] + valid_map + img_bic * valid_map
            if self.ref_num - 2 >= 0:
                for i in range(self.ref_num-2, -1 , -1):
                    valid_map = np.zeros([1, img_bic.shape[1], img_bic.shape[2]])
                    valid_map[(img_ref_near[i][0:1, :, :] == -1) & (img_ref_near[i][1:2, :, :] == -1) & (img_ref_near[i][2:3, :, :] == -1)] = 1
                    img_ref_near[i] = img_ref_near[i] + valid_map + img_ref_near[i+1] * valid_map


        ret["img_lr"] = img_lr
        ret["img_hr"] = img_hr
        ret["img_bic"] = img_bic
        ret["img_ref_multiview"] = np.array(img_ref_multiview)
        ret["img_ref_near"] = np.array(img_ref_near)

        return ret


