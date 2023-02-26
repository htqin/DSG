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

import argparse
import torch
import numpy as np
import random
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from utils import *
from distill_data import *

import logging
import sys
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format=log_format,
                    datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()

# model settings
def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    
    parser.add_argument('--dataset',
                        type=str,
                        default='imagenet',
                        choices=['imagenet', 'cifar10'],
                        help='type of dataset')
    parser.add_argument('--data_path',
                        type=str,
                        default='path/to/dataset',
                        help='path of dataset')
    parser.add_argument('--model',
                        type=str,
                        default='resnet18',
                        choices=[
                            'resnet18', 'resnet50', 'inceptionv3',
                            'mobilenetv2_w1', 'shufflenet_g1_w1',
                            'resnet20_cifar10', 'sqnxt23_w2', 'bn_vgg16'
                        ],
                        help='model to be quantized')
    parser.add_argument('--batch_size',
                        type=int,
                        default=40,
                        help='batch size of distilled data')
    parser.add_argument('--w_bit',
                        type=int,
                        default=4)
    parser.add_argument('--a_bit',
                        type=int,
                        default=4)
    parser.add_argument('--act_q_method',
                        type=str,
                        default='ema')
    parser.add_argument('--onehot',
                        type=int,
                        default=1)
    parser.add_argument('--seed',
                        type=int,
                        default=666)
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=256,
                        help='batch size of test data')
    parser.add_argument('--resume', type=str)
    args = parser.parse_args()
    return args

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    
if __name__ == '__main__':
    args = arg_parse()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    setup_seed(args.seed)
    
    # Load pretrained model
    if args.model == 'bn_vgg16':
        from utils.vgg import *
        model = vgg16_bn()
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = ptcv_get_model(args.model, pretrained=True)

    logger.info('seed: {}'.format(args.seed))
    logger.info('w_bit: ' + str(args.w_bit))
    logger.info('a_bit: ' + str(args.a_bit))
    logger.info("args: " + str(args))
    logger.info('****** Full precision model loaded ******')

    # Load validation data
    test_loader = getTestData(args.dataset,
                              batch_size=args.test_batch_size,
                              path=args.data_path,
                              for_inception=args.model.startswith('inception'))
    logger.info('****** Test Data loaded ******')

    # Generate distilled data
    dataloader = getDistilData(
        args,
        model.cuda(),
        args.dataset,
        batch_size=args.batch_size,
        for_inception=args.model.startswith('inception'))
    logger.info('****** Distill Data loaded ******')

    # Quantize single-precision model to 8-bit model
    quantized_model = quantize_model(model, act_q_method=args.act_q_method, w_bit=args.w_bit, a_bit=args.a_bit)

    # Freeze BatchNorm statistics
    quantized_model.eval()
    quantized_model = quantized_model.cuda()

    # Update activation range according to distilled data
    update(quantized_model, dataloader)
    logger.info('****** Zero Shot Quantization Finished ******')

    # Freeze activation range during test
    freeze_model(quantized_model)
    quantized_model = nn.DataParallel(quantized_model).cuda()

    # Test the final quantized model
    test(quantized_model, test_loader)
