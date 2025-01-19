# Code adapted from https://github.com/utkuozbulak/pytorch-cnn-visualizations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image, ImageFilter

from lib.options import BaseOptions
from lib.train_util import *
from lib.data.TrainDataset_vis import TrainDataset_vis
from lib.model import *
from lib.plotting import *

# get options
opt = BaseOptions().parse()
PLOT_TIMESTEP = True
n_plot = 2
n_res = opt.num_sample_inout
#n_res = 500


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.image_filter._modules.items():
            #print('module_pos: ', module_pos, 'module: ', module)
            x = module(x)  # Forward
            # 'conv_last0', 'conv_last1', 'conv_last2', 'conv_last3'
            if module_pos == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
                #print('DEBUG: Got conv output')
        #exit()
        return conv_output, x

    def forward_pass(self, image_tensor, sample_tensor, calib_tensor, labels=None):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(image_tensor)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        res, error = self.model.forward(image_tensor, sample_tensor, calib_tensor, labels=labels)
        #print('res shape: ', res.shape)
        return conv_output, res


class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, image_tensor, sample_tensor, calib_tensor, labels=None, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(image_tensor, sample_tensor, calib_tensor, labels)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Get convolution outputs
        target = conv_output[0]
        # Create empty numpy array for cam
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i in range(len(target)):
            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :],0),0)
            # Upsampling to input size
            saliency_map = F.interpolate(saliency_map, size=(512, 512), mode='bilinear', align_corners=False)

            #plot_map = np.uint8(np.transpose(saliency_map[0,:,:,:].detach().cpu().numpy(), (1, 2, 0)))
            #plt.imshow(plot_map)
            #plt.show()

            if saliency_map.max() == saliency_map.min():
                continue
            # Scale between 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            # Get the target score
            image_tensorX = image_tensor * norm_saliency_map
            conv_output, masked_output = self.extractor.forward_pass(image_tensorX, sample_tensor, calib_tensor, labels=labels)

            #print('image_tensorX: ', image_tensorX.shape)
            #print('masked_output: ', masked_output.shape)

            # get difference of masked to baseline prediction? -> according to original implementation this is not required
            #res = (masked_output - model_output)[0, 0, :]
            res = masked_output[0, 0, :]
            # get binary result for softmax
            inverse_res = torch.ones_like(res) - res
            binary_res = torch.stack((res, inverse_res), dim=0)
            # multiply by labels to only get positive predictions
            w_point = F.softmax(binary_res, dim=0)[1] * labels[0, 0, :]
            #print('w_point: ', w_point.shape)
            idx = torch.nonzero(w_point)
            w = torch.mean(w_point[idx])
            #print('w_point[idx]: ', w_point[idx].shape)

            #w = torch.mean(F.softmax(res[0, 0, :], dim=0))

            cam += w.detach().cpu().numpy() * target[i, :, :].detach().cpu().numpy()

        cam = np.maximum(cam, 0)

        # percentiles for plotting -> better visibility of CAM
        cam_max = np.percentile(cam, 99.5)
        cam_min = np.min(cam)
        cam = (np.clip((cam - cam_min) / (cam_max - cam_min), 0, 1))


        #cam = np.minimum(cam, 0)
        #cam = np.abs(cam)
        #cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        #cam = np.uint8(255-cam)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((image_tensor.shape[2],
                       image_tensor.shape[3]), Image.ANTIALIAS))/255
        return cam

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


    if PLOT_TIMESTEP:
        num_data = len(train_dataset) // 36
        angle_iterator = np.arange(0, 36, 1) * num_data
        angle_iterator = np.arange(0, 2, 1) * num_data
        print(angle_iterator)
        plot_timesteps = angle_iterator + 56
        #plot_timesteps = angle_iterator + 25
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
        netG.eval()
        for train_data in dataset:

            # retrieve the data
            image_tensor = torch.unsqueeze(train_data['img'].to(device=cuda), 0)
            image_tensor.requires_grad = True
            calib_tensor = torch.unsqueeze(train_data['calib'].to(device=cuda), 0)
            sample_tensor = torch.unsqueeze(train_data['samples'].to(device=cuda), 0)

            image_tensor, calib_tensor = reshape_multiview_tensors(image_tensor, calib_tensor)

            label_tensor = torch.unsqueeze(train_data['labels'].to(device=cuda), 0)

            # Score cam
            PLOT_ALL_HG_LAYERS = True
            if PLOT_ALL_HG_LAYERS:
                target_layers = ['conv_last0', 'conv_last1', 'conv_last2', 'conv_last3', 'l0', 'l1', 'l2', 'l3']
            else:
                target_layers = ['l3']

            for layer in target_layers:
                score_cam = ScoreCam(netG, target_layer=layer)
                # Generate cam mask
                cam = score_cam.generate_cam(image_tensor, sample_tensor, calib_tensor, labels=label_tensor, target_class=1)

                # Plotting
                plot_Score_CAM(cam, train_data, layer, opt)




if __name__ == '__main__':
    train(opt)