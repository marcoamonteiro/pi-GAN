# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Equivariance metrics (EQ-T, EQ-T_frac, and EQ-R) from the paper
"Alias-Free Generative Adversarial Networks"."""

import argparse
import math
import glob
import sys
import os

import copy
import numpy as np
import torch
import torch.fft
from torch_utils.ops import upfirdn2d
from torchvision.utils import save_image
from tqdm import tqdm

import curriculums

#----------------------------------------------------------------------------
# Utilities.

def sinc(x):
    y = (x * np.pi).abs()
    z = torch.sin(y) / y.clamp(1e-30, float('inf'))
    return torch.where(y < 1e-30, torch.ones_like(x), z)

def lanczos_window(x, a):
    x = x.abs() / a
    return torch.where(x < 1, sinc(x), torch.zeros_like(x))

def rotation_matrix(angle):
    angle = torch.as_tensor(angle).to(torch.float32)
    mat = torch.eye(3, device=angle.device)
    mat[0, 0] = angle.cos()
    mat[0, 1] = angle.sin()
    mat[1, 0] = -angle.sin()
    mat[1, 1] = angle.cos()
    return mat

#----------------------------------------------------------------------------
# Apply integer translation to a batch of 2D images. Corresponds to the
# operator T_x in Appendix E.1.

def apply_integer_translation(x, tx, ty):
    _N, _C, H, W = x.shape
    tx = torch.as_tensor(tx * W).to(dtype=torch.float32, device=x.device)
    ty = torch.as_tensor(ty * H).to(dtype=torch.float32, device=x.device)
    ix = tx.round().to(torch.int64)
    iy = ty.round().to(torch.int64)

    z = torch.zeros_like(x)
    m = torch.zeros_like(x)
    if abs(ix) < W and abs(iy) < H:
        y = x[:, :, max(-iy,0) : H+min(-iy,0), max(-ix,0) : W+min(-ix,0)]
        z[:, :, max(iy,0) : H+min(iy,0), max(ix,0) : W+min(ix,0)] = y
        m[:, :, max(iy,0) : H+min(iy,0), max(ix,0) : W+min(ix,0)] = 1
    return z, m

#----------------------------------------------------------------------------
# Apply integer translation to a batch of 2D images. Corresponds to the
# operator T_x in Appendix E.2.

def apply_fractional_translation(x, tx, ty, a=3):
    _N, _C, H, W = x.shape
    tx = torch.as_tensor(tx * W).to(dtype=torch.float32, device=x.device)
    ty = torch.as_tensor(ty * H).to(dtype=torch.float32, device=x.device)
    ix = tx.floor().to(torch.int64)
    iy = ty.floor().to(torch.int64)
    fx = tx - ix
    fy = ty - iy
    b = a - 1

    z = torch.zeros_like(x)
    zx0 = max(ix - b, 0)
    zy0 = max(iy - b, 0)
    zx1 = min(ix + a, 0) + W
    zy1 = min(iy + a, 0) + H
    if zx0 < zx1 and zy0 < zy1:
        taps = torch.arange(a * 2, device=x.device) - b
        filter_x = (sinc(taps - fx) * sinc((taps - fx) / a)).unsqueeze(0)
        filter_y = (sinc(taps - fy) * sinc((taps - fy) / a)).unsqueeze(1)
        y = x
        y = upfirdn2d.filter2d(y, filter_x / filter_x.sum(), padding=[b,a,0,0])
        y = upfirdn2d.filter2d(y, filter_y / filter_y.sum(), padding=[0,0,b,a])
        y = y[:, :, max(b-iy,0) : H+b+a+min(-iy-a,0), max(b-ix,0) : W+b+a+min(-ix-a,0)]
        z[:, :, zy0:zy1, zx0:zx1] = y

    m = torch.zeros_like(x)
    mx0 = max(ix + a, 0)
    my0 = max(iy + a, 0)
    mx1 = min(ix - b, 0) + W
    my1 = min(iy - b, 0) + H
    if mx0 < mx1 and my0 < my1:
        m[:, :, my0:my1, mx0:mx1] = 1
    return z, m

#----------------------------------------------------------------------------
# Construct an oriented low-pass filter that applies the appropriate
# bandlimit with respect to the input and output of the given affine 2D
# image transformation.

def construct_affine_bandlimit_filter(mat, a=3, amax=16, aflt=64, up=4, cutoff_in=1, cutoff_out=1):
    assert a <= amax < aflt
    mat = torch.as_tensor(mat).to(torch.float32)

    # Construct 2D filter taps in input & output coordinate spaces.
    taps = ((torch.arange(aflt * up * 2 - 1, device=mat.device) + 1) / up - aflt).roll(1 - aflt * up)
    yi, xi = torch.meshgrid(taps, taps)
    xo, yo = (torch.stack([xi, yi], dim=2) @ mat[:2, :2].t()).unbind(2)

    # Convolution of two oriented 2D sinc filters.
    fi = sinc(xi * cutoff_in) * sinc(yi * cutoff_in)
    fo = sinc(xo * cutoff_out) * sinc(yo * cutoff_out)
    f = torch.fft.ifftn(torch.fft.fftn(fi) * torch.fft.fftn(fo)).real

    # Convolution of two oriented 2D Lanczos windows.
    wi = lanczos_window(xi, a) * lanczos_window(yi, a)
    wo = lanczos_window(xo, a) * lanczos_window(yo, a)
    w = torch.fft.ifftn(torch.fft.fftn(wi) * torch.fft.fftn(wo)).real

    # Construct windowed FIR filter.
    f = f * w

    # Finalize.
    c = (aflt - amax) * up
    f = f.roll([aflt * up - 1] * 2, dims=[0,1])[c:-c, c:-c]
    f = torch.nn.functional.pad(f, [0, 1, 0, 1]).reshape(amax * 2, up, amax * 2, up)
    f = f / f.sum([0,2], keepdim=True) / (up ** 2)
    f = f.reshape(amax * 2 * up, amax * 2 * up)[:-1, :-1]
    return f

#----------------------------------------------------------------------------
# Apply the given affine transformation to a batch of 2D images.

def apply_affine_transformation(x, mat, up=4, **filter_kwargs):
    _N, _C, H, W = x.shape
    mat = torch.as_tensor(mat).to(dtype=torch.float32, device=x.device)

    # Construct filter.
    f = construct_affine_bandlimit_filter(mat, up=up, **filter_kwargs)
    assert f.ndim == 2 and f.shape[0] == f.shape[1] and f.shape[0] % 2 == 1
    p = f.shape[0] // 2

    # Construct sampling grid.
    theta = mat.inverse()
    theta[:2, 2] *= 2
    theta[0, 2] += 1 / up / W
    theta[1, 2] += 1 / up / H
    theta[0, :] *= W / (W + p / up * 2)
    theta[1, :] *= H / (H + p / up * 2)
    theta = theta[:2, :3].unsqueeze(0).repeat([x.shape[0], 1, 1])
    g = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)

    # Resample image.
    y = upfirdn2d.upsample2d(x=x, f=f, up=up, padding=p)
    z = torch.nn.functional.grid_sample(y, g, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Form mask.
    m = torch.zeros_like(y)
    c = p * 2 + 1
    m[:, :, c:-c, c:-c] = 1
    m = torch.nn.functional.grid_sample(m, g, mode='nearest', padding_mode='zeros', align_corners=False)
    return z, m

#----------------------------------------------------------------------------
# Apply fractional rotation to a batch of 2D images. Corresponds to the
# operator R_\alpha in Appendix E.3.

def apply_fractional_rotation(x, angle, a=3, **filter_kwargs):
    angle = torch.as_tensor(angle).to(dtype=torch.float32, device=x.device)
    mat = rotation_matrix(angle)
    return apply_affine_transformation(x, mat, a=a, amax=a*2, **filter_kwargs)

#----------------------------------------------------------------------------
# Modify the frequency content of a batch of 2D images as if they had undergo
# fractional rotation -- but without actually rotating them. Corresponds to
# the operator R^*_\alpha in Appendix E.3.

def apply_fractional_pseudo_rotation(x, angle, a=3, **filter_kwargs):
    angle = torch.as_tensor(angle).to(dtype=torch.float32, device=x.device)
    mat = rotation_matrix(-angle)
    f = construct_affine_bandlimit_filter(mat, a=a, amax=a*2, up=1, **filter_kwargs)
    y = upfirdn2d.filter2d(x=x, f=f)
    m = torch.zeros_like(y)
    c = f.shape[0] // 2
    m[:, :, c:-c, c:-c] = 1
    return y, m


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show(tensor_img):
    if len(tensor_img.shape) > 3:
        tensor_img = tensor_img.squeeze(0)
    tensor_img = tensor_img.permute(1, 2, 0).squeeze().cpu().numpy()
    plt.imshow(tensor_img)
    plt.show()
    
def generate_img(gen, z, **kwargs):
    
    with torch.no_grad():
        img, depth_map = generator.staged_forward(z, **kwargs)
        tensor_img = img.detach()
        
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min)/(img_max-img_min)
        img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    return img, tensor_img, depth_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--seeds', nargs='+', default=[0, 1, 2])
    parser.add_argument('--output_dir', type=str, default='imgs')
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--ray_step_multiplier', type=int, default=2)
    parser.add_argument('--curriculum', type=str, default='CelebA')
    opt = parser.parse_args()

    curriculum = getattr(curriculums, opt.curriculum)
    curriculum['num_steps'] = curriculum[0]['num_steps'] * opt.ray_step_multiplier
    curriculum['img_size'] = opt.image_size
    curriculum['psi'] = 0.7
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['lock_view_dependence'] = opt.lock_view_dependence
    curriculum['last_back'] = curriculum.get('eval_last_back', False)
    curriculum['nerf_noise'] = 0
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}
    
    os.makedirs(opt.output_dir, exist_ok=True)

    generator = torch.load(opt.path, map_location=torch.device(device))
    ema_file = opt.path.split('generator')[0] + 'ema.pth'
    ema = torch.load(ema_file)
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()
    
    face_angles = [-0.5, -0.25, 0.25, 0.5]

    face_angles = [a + curriculum['h_mean'] for a in face_angles]
    curriculum['v_mean'] -= 0.5
    sums = None
    for seed in tqdm(opt.seeds):
        s = []
        images = []
        torch.manual_seed(seed)
        curriculum['h_mean'] = 0
        z = torch.randn((1, 256), device=device)
        _, orig, _ = generate_img(generator, z, **curriculum)

        for i, yaw in enumerate(face_angles):

            curriculum['h_mean'] = yaw
            angle = yaw * np.pi

            _, img, _ = generate_img(generator, z, **curriculum)
            ref, ref_mask = apply_fractional_rotation(orig, angle)
            pseudo, pseudo_mask = apply_fractional_pseudo_rotation(img, angle)

            mask = ref_mask * pseudo_mask
            s += [(ref - pseudo).square() * mask, mask]
        s = torch.stack([x.to(torch.float64).sum() for x in s])
        print(s)
        sums = sums + s if sums is not None else s

    sums = sums.cpu()
    mses = sums[0::2] / sums[1::2]
    psnrs = np.log10(2) * 20 - mses.log10() * 10
    psnrs = tuple(psnrs.numpy())
    
    print(psnrs[0] if len(psnrs) == 1 else psnrs)