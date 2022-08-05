import numpy as np
import time
from collections import OrderedDict
import argparse
import subprocess
import string
import random
import logging
import sys
import torch
import math

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

def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
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
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

# rgb_to_ycbcr = np.array([[65.481, 128.553, 24.966],
#                          [-37.797, -74.203, 112.0],
#                          [112.0, -93.786, -18.214]])
#
# ycbcr_to_rgb = np.linalg.inv(rgb_to_ycbcr)
#
# def rgb2ycbcr(img, only_y = True):
#     """ img value must be between 0 and 255"""
#     img = np.float64(img)
#     img = np.dot(img, rgb_to_ycbcr.T) / 255.0
#     img = img + np.array([16, 128, 128])
#     if only_y == True:
#         return img[...,0]
#     return img
#
# def ycbcr2rgb(img):
#     """ img value must be between 0 and 255"""
#     img = np.float64(img)
#     img = img - np.array([16, 128, 128])
#     img = np.dot(img, ycbcr_to_rgb.T) * 255.0
#     return img

def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((
            (absx > 1) * (absx <= 2)).type_as(absx))

def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)

def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC BGR [0,1]
    # output: HWC BGR [0,1] w/o round
    img = torch.from_numpy(img)

    in_H, in_W, in_C = img.size()
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[i, :, 0] = img_aug[idx:idx + kernel_width, :, 0].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 1] = img_aug[idx:idx + kernel_width, :, 1].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 2] = img_aug[idx:idx + kernel_width, :, 2].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, i, 0] = out_1_aug[:, idx:idx + kernel_width, 0].mv(weights_W[i])
        out_2[:, i, 1] = out_1_aug[:, idx:idx + kernel_width, 1].mv(weights_W[i])
        out_2[:, i, 2] = out_1_aug[:, idx:idx + kernel_width, 2].mv(weights_W[i])

    return out_2.numpy()

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def logging_setup(out_path=None):
    if logging.root:
        del logging.root.handlers[:]
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(str(out_path)),
            logging.StreamHandler(stream=sys.stdout),
        ],
        # format="[%(asctime)s:%(levelname)s:%(module)s:%(funcName)s] %(message)s",
        format="[%(asctime)s/%(levelname)s/%(module)s] %(message)s",
        datefmt="%Y-%m-%d/%H:%M",
    )


def random_string(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def format_seconds(secs_in, millis=True):
    s = []
    days, secs = divmod(secs_in, 24 * 60 * 60)
    if days > 0:
        s.append(f"{int(days)}d")
    hours, secs = divmod(secs, 60 * 60)
    if hours > 0:
        s.append(f"{int(hours):02d}h")
    mins, secs = divmod(secs, 60)
    if mins > 0:
        s.append(f"{int(mins):02d}m")
    if millis:
        s.append(f"{secs:06.3f}s")
    else:
        s.append(f"{int(secs):02d}s")
    s = "".join(s)
    return s


class Timer(object):
    def __init__(self):
        self.tic = time.time()

    def done(self):
        diff = time.time() - self.tic
        return diff

    def __call__(self):
        return self.done()

    def __str__(self):
        diff = self.done()
        return format_seconds(diff)


class StopWatch(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.timings = OrderedDict()
        self.starts = {}

    def toogle(self, name):
        if name in self.starts:
            self.stop(name)
        else:
            self.start(name)

    def start(self, name):
        self.starts[name] = time.time()

    def stop(self, name):
        tic = time.time()
        if name not in self.timings:
            self.timings[name] = []
        diff = tic - self.starts.pop(name, tic)
        self.timings[name].append(diff)
        return diff

    def get(self, name=None, reduce=np.sum):
        if name is not None:
            return reduce(self.timings[name])
        else:
            ret = {}
            for k in self.timings:
                ret[k] = reduce(self.timings[k])
            return ret

    def format_str(self, reduce=np.sum):
        return ", ".join(
            [
                f"{k}: {format_seconds(v)}"
                for k, v in self.get(reduce=reduce).items()
            ]
        )

    def __repr__(self):
        return self.format_str()

    def __str__(self):
        return self.format_str()


class ETA(object):
    def __init__(self, length, current_idx=0):
        self.reset(length, current_idx=current_idx)

    def reset(self, length=None, current_idx=0):
        if length is not None:
            self.length = length
        self.current_idx = current_idx
        self.start_time = time.time()
        self.current_time = time.time()

    def update(self, idx):
        self.current_idx = idx
        self.current_time = time.time()

    def inc(self):
        self.current_idx += 1
        self.current_time = time.time()

    def get_elapsed_time(self):
        return self.current_time - self.start_time

    def get_item_time(self):
        return self.get_elapsed_time() / (self.current_idx + 1)

    def get_remaining_time(self):
        return self.get_item_time() * (self.length - self.current_idx + 1)

    def get_total_time(self):
        return self.get_item_time() * self.length

    def get_elapsed_time_str(self, millis=True):
        return format_seconds(self.get_elapsed_time(), millis=millis)

    def get_remaining_time_str(self, millis=True):
        return format_seconds(self.get_remaining_time(), millis=millis)

    def get_percentage_str(self):
        perc = self.get_elapsed_time() / self.get_total_time() * 100
        return f"{int(perc):02d}%"

    def get_str(
        self, percentage=True, elapsed=True, remaining=True, millis=False
    ):
        s = []
        if percentage:
            s.append(self.get_percentage_str())
        if elapsed:
            s.append(self.get_elapsed_time_str(millis=millis))
        if remaining:
            s.append(self.get_remaining_time_str(millis=millis))
        return "/".join(s)


def flatten(vals):
    if isinstance(vals, dict):
        ret = []
        for v in vals.values():
            ret.extend(flatten(v))
        return ret
    elif isinstance(vals, (list, np.ndarray)):
        if isinstance(vals, np.ndarray):
            vals = vals.ravel()
        ret = []
        for v in vals:
            ret.extend(flatten(v))
        return ret
    else:
        return [vals]


class CumulativeMovingAverage(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.vals = None

    def append(self, x):
        if isinstance(x, dict):
            if self.n == 0:
                self.vals = {}
                for k, v in x.items():
                    self.vals[k] = np.array(v)
            else:
                for k, v in x.items():
                    self.vals[k] = (np.array(v) + self.n * self.vals[k]) / (
                        self.n + 1
                    )
        else:
            x = np.asarray(x)
            if self.n == 0:
                self.vals = x
            else:
                self.vals = (x + self.n * self.vals) / (self.n + 1)
        self.n += 1
        return self.vals

    def vals_list(self):
        return flatten(self.vals)


def git_hash(cwd=None):
    ret = subprocess.run(
        ["git", "describe", "--always"],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    hash = ret.stdout
    if hash is not None and "fatal" not in hash.decode():
        return hash.decode().strip()
    else:
        return None
