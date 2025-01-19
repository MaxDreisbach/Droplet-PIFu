# Code adapted from https://github.com/utkuozbulak/pytorch-cnn-visualizations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import torch
import torch.nn.functional as F

from lib.options import BaseOptions
from lib.train_util import *
from lib.model import *

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm

# get options
opt = BaseOptions().parse()

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
            x = module(x)  # Forward
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
        # Forward pass on the classifier
        res, error = self.model.forward(image_tensor, sample_tensor, calib_tensor, labels=labels)
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

            # get difference of masked to baseline prediction? -> according to original implementation this is not required
            res = masked_output[0, 0, :]
            # get binary result for softmax
            inverse_res = torch.ones_like(res) - res
            binary_res = torch.stack((res, inverse_res), dim=0)

            #Construct labels tensor from binarised res, ie. if res>0.5 labels=1, else labels=0
            labels = (res > 0.5).float()

            # multiply by labels to only get positive predictions
            #w_point = F.softmax(binary_res, dim=0)[1]
            w_point = F.softmax(binary_res, dim=0)[1] * labels
            idx = torch.nonzero(w_point)
            w = torch.mean(w_point[idx])

            cam += w.detach().cpu().numpy() * target[i, :, :].detach().cpu().numpy()

        #cam = -cam
        #cam = np.maximum(cam, 0)
        #cam = np.minimum(cam, 0)
        cam = np.abs(cam)
        #cam = -cam

        # percentiles for plotting -> better visibility of CAM
        cam_max = np.percentile(cam, 99.5)
        cam_min = np.min(cam)
        cam = (np.clip((cam - cam_min) / (cam_max - cam_min), 0, 1))
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((image_tensor.shape[2],
                       image_tensor.shape[3]), Image.ANTIALIAS))/255
        return cam


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
        self.netG.eval()

        image_tensor = data['img'].to(device=self.cuda)
        #image_tensor.requires_grad = True
        calib_tensor = data['calib'].to(device=self.cuda)

        net = self.netG

        b_min = data['b_min']
        b_max = data['b_max']

        coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max, transform=transform)

        # Evaluate the grid
        coords = coords.reshape([3, -1])
        num_pts = coords.shape[1]
        label_tensor = torch.unsqueeze(torch.unsqueeze(torch.ones(num_pts), 0), 0).to(device=self.cuda)
        sample_tensor = torch.unsqueeze(torch.tensor(coords.astype(np.float32)), 0).to(device=self.cuda)

        # Score cam
        PLOT_ALL_HG_LAYERS = True
        if PLOT_ALL_HG_LAYERS:
            target_layers = ['conv_last0', 'conv_last1', 'conv_last2', 'conv_last3', 'l0', 'l1', 'l2', 'l3']
        else:
            target_layers = ['l3']

        for layer in target_layers:
            score_cam = ScoreCam(net, target_layer=layer)
            # Generate cam mask
            cam = score_cam.generate_cam(image_tensor, sample_tensor, calib_tensor, labels=label_tensor,
                                         target_class=1)

            # Plotting
            plot_Score_CAM(cam, data, layer, opt)


if __name__ == '__main__':
    evaluator = Evaluator(opt)

    test_images = glob.glob(os.path.join(opt.test_folder_path, '*'))
    test_images = [f for f in test_images if ('png' in f or 'jpg' in f) and (not 'mask' in f)]
    test_images.sort()
    test_images = test_images[10:]
    #test_images = test_images[4:]
    test_masks = [f[:-4]+'_mask.png' for f in test_images]

    print("num; ", len(test_masks))

    for image_path, mask_path in tqdm.tqdm(zip(test_images, test_masks)):
        print(image_path, mask_path)
        data = evaluator.load_image(image_path, mask_path)
        evaluator.eval(data, resolution=opt.resolution, use_octree=False)

