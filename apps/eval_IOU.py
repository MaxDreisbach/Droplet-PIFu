import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import cv2
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.data import *
from lib.model import *
from lib.geometry import index

# get options
opt = BaseOptions().parse()

def train(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    train_dataset = TrainDataset(opt, phase='train')
    test_dataset = TrainDataset(opt, phase='test')

    projection_mode = train_dataset.projection_mode

    # create data loader
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)

    print('train data size: ', len(train_data_loader))

    # NOTE: batch size should be 1 and use all the points for evaluation
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1, shuffle =True,
                                  num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('validation data size: ', len(test_data_loader))

    # create net
    netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
    print('Using Network: ', netG.name)

    def set_eval():
        netG.eval()

    # load checkpoints
    if opt.load_netG_checkpoint_path is not None:
        print('loading for net G ...', opt.load_netG_checkpoint_path)
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

    if opt.continue_train:
        if opt.resume_epoch < 0:
            model_path = '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name)
        else:
            model_path = '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
        print('Resuming from ', model_path)
        netG.load_state_dict(torch.load(model_path, map_location=cuda))

    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    IOU_log = os.path.join(opt.checkpoints_path, opt.name, str(opt.resume_epoch) + '_IOU.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))

        #### test
        with torch.no_grad():
            set_eval()

            if not opt.no_num_eval:
                test_losses = {}
                print('calc error (validation) ...')
                test_errors = calc_error(opt, netG, cuda, test_dataset, opt.batch_size)
                str_err_test = 'eval val MSE: {0:06f} IOU: {1:06f} prec: {2:06f} recall: {3:06f}\n'.format(*test_errors)
                print(str_err_test)
                with open(IOU_log, 'a') as outfile:
                    outfile.write(str_err_test)
                MSE, IOU, prec, recall = test_errors
                test_losses['MSE(val)'] = MSE
                test_losses['IOU(val)'] = IOU
                test_losses['prec(val)'] = prec
                test_losses['recall(val)'] = recall

                print('calc error (train) ...')
                train_dataset.is_train = False
                train_errors = calc_error(opt, netG, cuda, train_dataset, opt.batch_size)
                train_dataset.is_train = True
                str_err_train = 'eval train MSE: {0:06f} IOU: {1:06f} prec: {2:06f} recall: {3:06f}\n'.format(*train_errors)
                print(str_err_train)
                with open(IOU_log, 'a') as outfile:
                    outfile.write(str_err_train)
                MSE, IOU, prec, recall = train_errors
                test_losses['MSE(train)'] = MSE
                test_losses['IOU(train)'] = IOU
                test_losses['prec(train)'] = prec
                test_losses['recall(train)'] = recall



            if not opt.no_gen_mesh:
                print('generate mesh (test) ...')
                for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                    save_path = '%s/%s/test_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, 0, test_dataset[gen_idx]['name'])
                    print(save_path)
                    #evaluator.eval(test_data, True)
                    gen_mesh(opt, netG, cuda, test_dataset[gen_idx], save_path, use_octree=True)

                print('generate mesh (train) ...')
                train_dataset.is_train = False
                for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                    save_path = '%s/%s/train_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, 0, train_dataset[gen_idx]['name'])
                    gen_mesh(opt, netG, cuda, train_dataset[gen_idx], save_path, use_octree=True)
                train_dataset.is_train = True


if __name__ == '__main__':
    train(opt)