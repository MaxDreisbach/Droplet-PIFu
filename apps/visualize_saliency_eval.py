# Code adapted from https://github.com/utkuozbulak/pytorch-cnn-visualizations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import torch
from torch.autograd import grad
from torch.utils.data import DataLoader

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.model import *

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm

# get options
opt = BaseOptions().parse()

class Evaluator:
    def __init__(self, opt, projection_mode='orthogonal'):
        self.opt = opt
        self.load_size = self.opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # set cuda
        cuda = torch.device('cuda:%d' % opt.gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        # create net
        netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
        print('Using Network: ', netG.name)

        if opt.load_netG_checkpoint_path:
            print('loading for net G ...', opt.load_netG_checkpoint_path)
            netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

        if opt.load_netC_checkpoint_path is not None:
            print('loading for net C ...', opt.load_netC_checkpoint_path)
            netC = ResBlkPIFuNet(opt).to(device=cuda)
            netC.load_state_dict(torch.load(opt.load_netC_checkpoint_path, map_location=cuda))
        else:
            netC = None

        os.makedirs(opt.results_path, exist_ok=True)
        os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

        opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
        with open(opt_log, 'w') as outfile:
            outfile.write(json.dumps(vars(opt), indent=2))

        self.cuda = cuda
        self.netG = netG
        self.netC = netC

    def load_image(self, image_path, mask_path):
        # Name
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        # Calib
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        yid = 0
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib = torch.Tensor(projection_matrix).float()
        # Mask
        mask = Image.open(mask_path).convert('L')
        mask = transforms.Resize(self.load_size)(mask)
        mask = transforms.ToTensor()(mask).float()
        # image
        image = Image.open(image_path).convert('RGB')
        image = self.to_tensor(image)        
        raw_image = image
        image = mask.expand_as(image) * image
        return {
            'name': img_name,
            'img': image.unsqueeze(0),
            'raw_img': raw_image.unsqueeze(0),
            'calib': calib.unsqueeze(0),
            'mask': mask.unsqueeze(0),
            'b_min': B_MIN,
            'b_max': B_MAX,
            'yid' : yid
        }


    def eval(self, data, resolution=512, num_samples=10000, use_octree=False, transform=None):
        '''
        Evaluate a data point
        :param data: a dict containing at least ['name'], ['image'], ['calib'], ['b_min'] and ['b_max'] tensors.
        :return:
        '''
        opt = self.opt
        self.netG.train()
        save_path = '%s/%s/result_%s.obj' % (opt.results_path, opt.name, data['name'])

        image_tensor = data['img'].to(device=self.cuda)
        image_tensor.requires_grad = True
        calib_tensor = data['calib'].to(device=self.cuda)

        net = self.netG

        net.filter(image_tensor)

        b_min = data['b_min']
        b_max = data['b_max']

        coords, mat = create_grid(resolution, resolution, resolution,
                                  b_min, b_max, transform=transform)

        # Then we define the lambda function for cell evaluation
        def eval_func(points):
            points = np.expand_dims(points, axis=0)
            points = np.repeat(points, net.num_views, axis=0)
            samples = torch.from_numpy(points).to(device=self.cuda).float()
            net.query(samples, calib_tensor)
            pred = net.get_preds()[0][0]
            return pred

        # Then we evaluate the grid
        if use_octree:
            sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
        else:
            coords = coords.reshape([3, -1])
            num_pts = coords.shape[1]
            sdf = torch.zeros(num_pts)

            num_batches = num_pts // num_samples
            for i in range(num_batches):
                sdf[i * num_samples:i * num_samples + num_samples] = eval_func(
                    coords[:, i * num_samples:i * num_samples + num_samples])
            if num_pts % num_samples:
                sdf[num_batches * num_samples:] = eval_func(coords[:, num_batches * num_samples:])

        # Calculate gradients
        grads = torch.zeros((1, 3, 512, 512)).to(device=self.cuda)
        n_res = resolution**3
        n_res = 10000
        
        # random sample of prediction grid
        perm = torch.randperm(sdf.size(0))
        idx = perm[:n_res]
        sdf = sdf[idx]

        for i in range(n_res):
            print(i)
            if sdf[i] > 0.5:
                grads = grad(sdf[i], image_tensor, grad_outputs=torch.ones_like(sdf[i]),
                             retain_graph=True, allow_unused=True)[0]
                grads += torch.abs(grads)

        grads = grads[0, :, :, :].detach().cpu().numpy()

        # Tranpose into image shape (512, 512, 3) and average
        grads = grads / n_res

        # Plotting
        plot_saliency_map(grads, data, opt)


if __name__ == '__main__':
    evaluator = Evaluator(opt)

    test_images = glob.glob(os.path.join(opt.test_folder_path, '*'))
    test_images = [f for f in test_images if ('png' in f or 'jpg' in f) and (not 'mask' in f)]
    #NEW: sort and cut list
    test_images.sort()
    #test_images = test_images[4:]
    test_masks = [f[:-4]+'_mask.png' for f in test_images]

    print("num; ", len(test_masks))

    for image_path, mask_path in tqdm.tqdm(zip(test_images, test_masks)):
        print(image_path, mask_path)
        data = evaluator.load_image(image_path, mask_path)
        evaluator.eval(data, resolution=opt.resolution, use_octree=False)

