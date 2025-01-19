# Code adapted from https://github.com/utkuozbulak/pytorch-cnn-visualizations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from torch.autograd import grad

from lib.options import BaseOptions
from lib.train_util import *
from lib.data.TrainDataset_vis import TrainDataset_vis
from lib.model import *
from lib.plotting import *

# get options
opt = BaseOptions().parse()
PLOT_TIMESTEP = False
n_plot = 100
n_res = opt.num_sample_inout
#n_res = 50

def train(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    train_dataset = TrainDataset_vis(opt, phase='train')
    test_dataset = TrainDataset_vis(opt, phase='test')

    projection_mode = train_dataset.projection_mode

    # create data loader
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=1, shuffle=False,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)

    print('train data size: ', len(train_data_loader))

    # NOTE: batch size should be 1 and use all the points for evaluation
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1, shuffle=False,
                                  num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('test data size: ', len(test_data_loader))

    # create net
    netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
    print('Using Network: ', netG.name)
    
    def set_train():
        netG.train()

    def set_eval():
        netG.eval()

    # load checkpoints
    if opt.load_netG_checkpoint_path is not None:
        print('loading for net G ...', opt.load_netG_checkpoint_path)
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    PLOT_TIMESTEP = False
    if PLOT_TIMESTEP:
        num_data = len(train_dataset) // 36
        angle_iterator = np.arange(0, 36, 1) * num_data
        print(angle_iterator)
        plot_timesteps = angle_iterator + 75
    else:
        plot_timesteps = np.arange(0, n_plot, 1)

    dataset = []
    for idx in plot_timesteps:
        train_data = train_dataset[idx]
        print('timestep: ', str(train_data['name']), 'rotation angle: ', str(train_data['yid'] * 10))
        dataset.append(train_data)


    # training
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch,0)
    for epoch in range(start_epoch, opt.num_epoch):
        set_train()
        for train_data in dataset:

            # retrieve the data
            image_tensor = torch.unsqueeze(train_data['img'].to(device=cuda), 0)
            image_tensor.requires_grad = True
            calib_tensor = torch.unsqueeze(train_data['calib'].to(device=cuda), 0)
            sample_tensor = torch.unsqueeze(train_data['samples'].to(device=cuda), 0)

            image_tensor, calib_tensor = reshape_multiview_tensors(image_tensor, calib_tensor)

            label_tensor = train_data['labels'].to(device=cuda)

            res, error = netG.forward(image_tensor, sample_tensor, calib_tensor, labels=label_tensor)
            netG.zero_grad()

            grads = torch.zeros((1, 3, 512, 512)).to(device=cuda)

            for i in range(n_res):
                print(i)
                if res[0, 0, i] > 0.5:
                    grads = grad(res[0, 0, i], image_tensor, grad_outputs=torch.ones_like(res[0, 0, i]),
                                 retain_graph=True, allow_unused=True)[0]
                    grads += torch.abs(grads)

            grads = grads[0, :, :, :].detach().cpu().numpy()

            # Tranpose into image shape (512, 512, 3) and average
            grads = grads / n_res

            # Plotting
            plot_saliency_map(grads, train_data, opt)


if __name__ == '__main__':
    train(opt)