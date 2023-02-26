#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*
import cv2
import os
import json
import numpy
from PIL import Image
import torch
import torch.nn as nn
import copy
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import *
from tensorboardX import SummaryWriter
from torchvision import transforms

def main_loss(A, B, s, cnt=-1, onehot=0, layer_num=20):
    m = s * torch.ones(A.size()).cuda()
    C = (A - B).abs() - m.abs() # m: mean/std
    zero = torch.zeros(C.size()).cuda()
    D = torch.max(C, zero)
    if cnt != -1 and onehot != 0:
        for i in range(min(int(D.size(0)), layer_num)):
            if i % layer_num == cnt:
                D[i] = D[i] * 2
    if cnt == 0:
        return D.norm()**2 / B.size(0)
    else:
        return D.norm()**2 / B.size(0)

class output_hook(object):
    """
	Forward_hook used to get the output of the intermediate layer.
	"""
    def __init__(self):
        super(output_hook, self).__init__()
        self.outputs = None

    def hook(self, module, input, output):
        self.outputs = output

    def clear(self):
        self.outputs = None

def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
                         diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


def compute_sci(phi_fake, phi_real): 
    def compute_diversity(phi):
        phi = phi.view(phi.size(0), -1)
        phi = F.normalize(phi, p=2, dim=1)
        S_B = torch.mm(phi, phi.t())
        eig_vals, eig_vecs = torch.eig(S_B, eigenvectors=True)
        return Variable(eig_vals[:, 0]), Variable(eig_vecs)

    def normalize_min_max(eig_vals):
        min_v, max_v = torch.min(eig_vals), torch.max(eig_vals)
        return (eig_vals - min_v) / (max_v - min_v)

    fake_eig_vals, fake_eig_vecs = compute_diversity(phi_fake)
    real_eig_vals, real_eig_vecs = compute_diversity(phi_real)
    # Scaling factor to make the two losses operating in comparable ranges.
    magnitude_loss = 0.0001 * F.mse_loss(target=real_eig_vals, input=fake_eig_vals).abs()
    structure_loss = -torch.sum(torch.mul(fake_eig_vecs, real_eig_vecs), 0)
    normalized_real_eig_vals = normalize_min_max(real_eig_vals)
    weighted_structure_loss = torch.sum(torch.mul(normalized_real_eig_vals, structure_loss)).abs()
    return magnitude_loss + weighted_structure_loss


def getDistilData(args,
                  teacher_model,
                  dataset,
                  batch_size,
                  num_batch=1,
                  for_inception=False):
    """
	Generate distilled data according to the BatchNorm statistics in the pretrained single-precision model.
	Currently only support a single GPU.

	teacher_model: pretrained single-precision model
	dataset: the name of the dataset
	batch_size: the batch size of generated distilled data
	num_batch: the number of batch of generated distilled data
	for_inception: whether the data is for Inception because inception has input size 299 rather than 224
	"""
    
    high_labels = 10 if dataset == 'cifar10' else 1000
    
    # initialize distilled data with random noise according to the dataset
    dataloader = getRandomData(dataset=dataset,
                               batch_size=batch_size,
                               for_inception=for_inception)
    eps = 1e-6

    # initialize hooks and single-precision model
    hooks, hook_handles, bn_stats, refined_gaussian = [], [], [], []
    teacher_model = teacher_model.cuda()
    teacher_model = teacher_model.eval()

    # get number of BatchNorm layers in the model
    layers = sum([
        1 if isinstance(layer, nn.BatchNorm2d) else 0
        for layer in teacher_model.modules()
    ])

    for n, m in teacher_model.named_modules():
        if isinstance(m, nn.Conv2d) and len(hook_handles) < layers:
            # register hooks on the convolutional layers to get the intermediate output after convolution and before BatchNorm.
            hook = output_hook()
            hooks.append(hook)
            hook_handles.append(m.register_forward_hook(hook.hook))
        if isinstance(m, nn.BatchNorm2d):
            # get the statistics in the BatchNorm layers
            bn_stats.append(
                (m.running_mean.detach().clone().flatten().cuda(),
                 torch.sqrt(m.running_var +
                            eps).detach().clone().flatten().cuda()))
    assert len(hooks) == len(bn_stats)

    label = []
    for i, gaussian_data in enumerate(dataloader):
        gaussian_data = gaussian_data.cuda()
        gaussian_data.requires_grad = True
        labels = Variable(torch.randint(low=0, high=high_labels, size=(batch_size,)))
        label.append(labels)
        labels = labels.cuda()
        crit = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam([gaussian_data], lr=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         min_lr=1e-4,
                                                         verbose=False,
                                                         patience=260)
        input_mean = torch.zeros(1, 3).cuda()
        input_std = torch.ones(1, 3).cuda()
        lim_0 = 30
        it_num = 500
        for it in range(it_num):
            teacher_model.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()
            output = teacher_model(gaussian_data)
            mean_loss = 0
            std_loss = 0

            # Load percentile from local
            m = np.load('./m_n_perc/m.npy', allow_pickle = True)
            n = np.load('./m_n_perc/n.npy', allow_pickle = True)

            # Start optimizing the generated samples
            for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):
                tmp_output = hook.outputs
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                tmp_mean = torch.mean(tmp_output.view(tmp_output.size(0), tmp_output.size(1), -1), dim=2)
                tmp_std = torch.sqrt(torch.var(tmp_output.view(tmp_output.size(0),
                                              tmp_output.size(1), -1),dim=2) + eps)
                
                mean_loss += main_loss(bn_mean, tmp_mean, m[cnt], cnt=cnt, onehot=args.onehot)
                std_loss += main_loss(bn_std, tmp_std, n[cnt], cnt=cnt, onehot=args.onehot)
                    
            tmp_mean = torch.mean(gaussian_data.view(gaussian_data.size(0), 3, -1), dim=2)
            tmp_std = torch.sqrt(torch.var(gaussian_data.view(gaussian_data.size(0), 3, -1), dim=2) + eps)
            mean_loss += main_loss(input_mean, tmp_mean, 0)
            std_loss += main_loss(input_std, tmp_std, 0)
            off1 = torch.randint(-lim_0, lim_0, size = (1,)).item()
            inputs_jit = torch.roll(gaussian_data, shifts=(off1, off1), dims=(2, 3)) 
            off1 = torch.randint(-lim_0, lim_0, size = (1,))
            loss_l2 = torch.norm(inputs_jit.view(batch_size, -1), dim=1).mean()
            loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

            _loss = mean_loss + std_loss

            images = gaussian_data
            gauss_img = torch.randn_like(images).cuda()
            sci_loss = compute_sci(images, gauss_img)
            total_loss = _loss + sci_loss / len(m)

            loss_labels = crit(output, labels)
            total_loss += loss_labels
            total_loss += (loss_labels + 0.01*loss_var_l2 + 0.01*loss_l2)

            if i % 5 == 0 and (it+1) % 500 == 0:
                print('iter:', it)
                print('mean_loss:', mean_loss.item())
                print('std_loss:', std_loss.item())
                print('loss_labels:', loss_labels.item())
                print('loss_l2:', loss_l2.item())
                print('loss_var_l1:', loss_var_l1.item())
                print('loss_var_l2:', loss_var_l2.item())

            # update the distilled data
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())

        refined_gaussian.append(gaussian_data.detach().clone())

        if i == num_batch-1:
            break

    for handle in hook_handles:
        handle.remove()

    return refined_gaussian
