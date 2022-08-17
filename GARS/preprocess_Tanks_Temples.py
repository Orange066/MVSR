import os
import sys
import cv2
import numpy as np
import  shutil as shutil
import  math
import  torch

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

def generate_mod_LR_bic(up_scale, sourcedir, savedir):
    # params: upscale factor, input directory, output directory
    saveHRpath = os.path.join(savedir, 'HR', 'x' + str(up_scale))
    saveLRpath = os.path.join(savedir, 'LR', 'x' + str(up_scale))
    saveBicpath = os.path.join(savedir, 'Bic', 'x' + str(up_scale))

    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    if not os.path.isdir(os.path.join(savedir, 'HR')):
        os.makedirs(os.path.join(savedir, 'HR'))
    if not os.path.isdir(os.path.join(savedir, 'LR')):
        os.makedirs(os.path.join(savedir, 'LR'))
    if not os.path.isdir(os.path.join(savedir, 'Bic')):
        os.makedirs(os.path.join(savedir, 'Bic'))

    if not os.path.isdir(saveHRpath):
        os.makedirs(saveHRpath)
    else:
        print('It will cover ' + str(saveHRpath))

    if not os.path.isdir(saveLRpath):
        os.makedirs(saveLRpath)
    else:
        print('It will cover ' + str(saveLRpath))

    if not os.path.isdir(saveBicpath):
        os.makedirs(saveBicpath)
    else:
        print('It will cover ' + str(saveBicpath))

    # filepaths = [f for f in os.listdir(sourcedir) if f.endswith('.png')]
    filepaths = [f for f in os.listdir(sourcedir)]
    num_files = len(filepaths)

    # prepare data with augementation
    for i in range(num_files):
        filename = filepaths[i]
        print('No.{} -- Processing {}'.format(i, filename))
        for root, dirs, files in os.walk(os.path.join(sourcedir, filename)):
            # if len(files) > 1 and 'ibr3d_pw_0.50' in root:
            if len(files) > 1:
                subfile_path = root[root.find('Tanks_and_Temples') + len('Tanks_and_Temples/'):]
                # print('subfile_path', subfile_path)
                files.sort()
                for file in files:
                    if os.path.splitext(file)[1] != '.jpg' or 'dm' in file:
                        continue
                    # read image
                    #print('sourcedir, file', os.path.join(root, file))
                    # print(os.path.join(root, file))
                    image = cv2.imread(os.path.join(root, file))
                    dm_npy = np.load(os.path.join(root, file.replace('im', 'dm').replace('jpg', 'npy')))
                    count_npy = os.path.join(root, file.replace('im', 'count').replace('jpg', 'npy'))

                    # image = imresize_np(image, 0.5, True)
                    # image = image[(image.shape[0]-H_dst) // 2:(image.shape[0]-H_dst) // 2 + H_dst, (image.shape[1]-W_dst) // 2: (image.shape[1]-W_dst) // 2+W_dst, :]
                    # print(image)
                    # print(type(image), image)
                    width = int(np.floor(image.shape[1] / up_scale))
                    height = int(np.floor(image.shape[0] / up_scale))
                    # modcrop
                    if len(image.shape) == 3:
                        image_HR = image[0:up_scale * height, 0:up_scale * width, :]
                    else:
                        image_HR = image[0:up_scale * height, 0:up_scale * width]
                    dm_npy = dm_npy[0:up_scale * height, 0:up_scale * width]

                    # LR
                    image_LR = imresize_np(image_HR, 1 / up_scale, True)
                    # bic
                    image_Bic = imresize_np(image_LR, up_scale, True)

                    # dm_npy = Image.fromarray(dm_npy)
                    # dm_npy = dm_npy.resize((dm_npy.size[0]*4, dm_npy.size[1]*4), Image.BICUBIC)
                    dm_npy = np.repeat(np.expand_dims(dm_npy, axis=2), 3, axis=2)
                    # dm_npy = np.repeat(np.expand_dims(dm_npy, axis=2), 3, axis=2).shape
                    dm_npy = imresize_np(dm_npy, 1 / up_scale, True)
                    dm_npy = dm_npy[:,:,0]
                    # dm_npy = np.array(dm_npy)
                    # dm_npy = dm_npy[..., np.newaxis]

                    dm_color = cv2.applyColorMap(cv2.convertScaleAbs(dm_npy , alpha=30), cv2.COLORMAP_WINTER)


                    # print('saveHRpath', saveHRpath)
                    print(os.path.join(saveHRpath, subfile_path, file))

                    if not os.path.isdir(os.path.join(saveHRpath, subfile_path)):
                        os.makedirs(os.path.join(saveHRpath, subfile_path))
                    if not os.path.isdir(os.path.join(saveLRpath, subfile_path)):
                        os.makedirs(os.path.join(saveLRpath, subfile_path))
                    if not os.path.isdir(os.path.join(saveBicpath, subfile_path)):
                        os.makedirs(os.path.join(saveBicpath, subfile_path))

                    cv2.imwrite(os.path.join(saveHRpath, subfile_path, file).replace('jpg', 'png'), image_HR)
                    cv2.imwrite(os.path.join(saveLRpath, subfile_path, file).replace('jpg', 'png'), image_LR)
                    cv2.imwrite(os.path.join(saveBicpath, subfile_path, file).replace('jpg', 'png'), image_Bic)
                    cv2.imwrite(os.path.join(saveLRpath, subfile_path, file.replace('im', 'dm')), dm_color)

                    np.save(os.path.join(saveLRpath, subfile_path, file.replace('im', 'dm').replace('jpg', 'npy')), dm_npy)
                    shutil.copy(count_npy, os.path.join(saveLRpath, subfile_path, file.replace('im', 'count').replace('jpg', 'npy')))

                ks_npy = np.load(os.path.join(root, 'Ks.npy'))
                rs_npy = os.path.join(root, 'Rs.npy')
                ts_npy = os.path.join(root, 'ts.npy')
                shutil.copy(rs_npy ,os.path.join(saveLRpath, subfile_path, 'Rs.npy'))
                shutil.copy(ts_npy, os.path.join(saveLRpath, subfile_path, 'ts.npy'))
                ks_npy[:,:2,:] = ks_npy[:,:2,:] / 4
                np.save(os.path.join(saveLRpath, subfile_path, 'Ks.npy'), ks_npy)



if __name__ == "__main__":
    generate_mod_LR_bic(4, './Tanks_and_Temples/', './Tanks_and_Temples_HLBIC_FR/')
