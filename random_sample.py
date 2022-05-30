import argparse
import math
import glob
import numpy as np
import sys
import os
from random import *
import random

import torch
from torchvision.utils import save_image
from tqdm import tqdm

import curriculums

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

    a = []
    for cnt in tqdm(range(10000)):
        yaw = random.gauss(math.pi*0.5,0.3)
        pitch = random.gauss(math.pi*0.5,0.155)
        curriculum['h_mean'] = yaw
        curriculum['v_mean'] = pitch

        z = torch.randn((1, 256), device=device)
        img, tensor_img, depth_map = generate_img(generator, z, **curriculum)

        save_image(tensor_img.detach().cpu(), os.path.join(opt.output_dir, f'{cnt:05d}.png'), normalize=True)
        a.append([yaw, pitch])
    np.save(os.path.join(opt.output_dir, 'pose.npy'),np.array(a))