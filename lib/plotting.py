import copy
import os
from PIL import Image, ImageFilter, ImageOps
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
from scipy.interpolate import interpn
from scipy.interpolate import griddata
import pyvista

def plot_contour(opt, samples, preds, labels, plane_dim, name, type, sample_name, dataset_type):
    font_size = 28.0
    sample_x = samples[0, 0, :].detach().cpu().numpy()
    sample_y = samples[0, 1, :].detach().cpu().numpy()
    sample_z = samples[0, 2, :].detach().cpu().numpy()
    sample = np.vstack((sample_x, sample_y, sample_z)).T
    label = labels.detach().cpu().numpy()
    pred = preds.detach().cpu().numpy()

    # interpolate point cloud to 2D-plane
    grid_res = complex(0, opt.resolution)
    if plane_dim == 'x':
        X, Y, Z = np.mgrid[-0.1:0.1:3j, -28:228:grid_res, -128:128:grid_res]
    if plane_dim == 'y':
        X, Y, Z = np.mgrid[-128:128:grid_res, 7.9:8.1:1j, -128:128:grid_res]
    if plane_dim == 'z':
        X, Y, Z = np.mgrid[-128:128:grid_res, -28:228:grid_res, -0.1:0.1:1j]

    pred_interpn = griddata(sample, pred, (X,Y,Z), method='linear')
    pred_nearest = griddata(sample, pred, (X,Y,Z), method='nearest')
    pred_interpn[np.isnan(pred_interpn)] = pred_nearest[np.isnan(pred_interpn)]
    pred_interpn = griddata(sample, pred, (X,Y,Z), method='nearest')

    label_interpn = griddata(sample, label, (X,Y,Z), method='linear')
    label_nearest = griddata(sample, label, (X,Y,Z), method='nearest')
    label_interpn[np.isnan(label_interpn)] = label_nearest[np.isnan(label_interpn)]
    label_interpn = griddata(sample, label, (X, Y, Z), method='nearest')

    if plane_dim == 'x':
        var_plot = np.squeeze(pred_interpn[0, :, :])
        gt_plot = np.squeeze(label_interpn[0, :, :])
    if plane_dim == 'y':
        var_plot = np.squeeze(pred_interpn[:, 0, :].T)
        gt_plot = np.squeeze(label_interpn[:, 0, :].T)
    if plane_dim == 'z':
        var_plot = np.squeeze(pred_interpn[:, :, 0].T)
        gt_plot = np.squeeze(label_interpn[:, :, 0].T)

    err_plot = np.absolute(gt_plot - var_plot)
    x, y = np.meshgrid(np.arange(opt.resolution) / opt.resolution, np.arange(opt.resolution) / opt.resolution)

    if type == 'alpha':
        levels = np.linspace(0, 1.0, 9)
        colormap = 'RdBu_r'
    if type == 'vel':
        levels = np.linspace(-1.5, 1.5, 10)
        colormap = 'RdBu_r'
    if type == 'pres':
        levels = np.linspace(-0.6, 3.0, 10)
        colormap = 'viridis'

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    plt.rcParams['figure.constrained_layout.use'] = True


    #fig, axs = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(15, 6.85))
    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(15, 7.0))

    p1 = axs[0].contourf(x, y, var_plot, levels=levels, cmap=colormap)
    p2 = axs[1].contourf(x, y, gt_plot, levels=levels, cmap=colormap)
    p3 = axs[2].contourf(x, y, err_plot, levels=levels, cmap=colormap)
    axs[0].contourf(x, y, var_plot, levels=levels, cmap=colormap)
    axs[1].contourf(x, y, gt_plot, levels=levels, cmap=colormap)
    axs[2].contourf(x, y, err_plot, levels=levels, cmap=colormap)
    axs[0].set_xbound(lower=0.0, upper=1.0)
    axs[1].set_xbound(lower=0.0, upper=1.0)
    axs[2].set_xbound(lower=0.0, upper=1.0)

    if plane_dim == 'x':
        axs[0].set_ylabel('$y$', fontsize=font_size)
        axs[0].set_xlabel('$z$', fontsize=font_size)
        axs[1].set_xlabel('$z$', fontsize=font_size)
        axs[2].set_xlabel('$z$', fontsize=font_size)
    if plane_dim == 'y':
        axs[0].set_ylabel('$z$', fontsize=font_size)
        axs[0].set_xlabel('$x$', fontsize=font_size)
        axs[1].set_xlabel('$x$', fontsize=font_size)
        axs[2].set_xlabel('$x$', fontsize=font_size)
    if plane_dim == 'z':
        axs[0].set_ylabel('$y$', fontsize=font_size)
        axs[0].set_xlabel('$x$', fontsize=font_size)
        axs[1].set_xlabel('$x$', fontsize=font_size)
        axs[2].set_xlabel('$x$', fontsize=font_size)

    x = np.arange(0.2, 0.8 + 0.001, 0.2)
    y = np.arange(0.0, 1.0 + 0.001, 0.2)

    axs[0].set_xticks(x)
    axs[0].set_yticks(y)
    axs[1].set_xticks(x)
    axs[1].set_yticks(y)
    axs[2].set_xticks(x)
    axs[2].set_yticks(y)


    axs[0].tick_params(axis ='both', which ='major', labelsize = font_size)
    axs[1].tick_params(axis ='both', which ='major', labelsize = font_size)
    axs[2].tick_params(axis ='both', which ='major', labelsize = font_size)

    axs[0].set_title(r'$\alpha_{\mathrm{R}}$', fontsize=32, y=1, pad=20)
    axs[1].set_title(r'$\alpha_{\mathrm{GT}}$', fontsize=32, y=1, pad=20)
    axs[2].set_title(r'$\alpha_{\mathrm{err}}$', fontsize=32, y=1, pad=20)

    axs[0].set_aspect('equal', adjustable="datalim")
    axs[1].set_aspect('equal', adjustable="datalim")
    axs[2].set_aspect('equal', adjustable="datalim")
    
    cbar_label = r'$\alpha$ [-]'
    cbar_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]


    cbar1 = fig.colorbar(p2, ax=axs[0], location='bottom', fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=font_size, which='major')
    cbar1.set_ticks(cbar_ticks)
    
    cbar2 = fig.colorbar(p2, ax=axs[1], location='bottom', fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=font_size, which='major')
    cbar2.set_ticks(cbar_ticks)
    
    cbar3 = fig.colorbar(p2, ax=axs[2], location='bottom', fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=font_size, which='major')
    cbar3.set_ticks(cbar_ticks)
    cbar1.set_label(cbar_label, fontsize=font_size, labelpad=0.1)
    cbar2.set_label(cbar_label, fontsize=font_size, labelpad=0.1)
    cbar3.set_label(cbar_label, fontsize=font_size, labelpad=0.1)

    filename = 'results/' + opt.name + '/pred_fields/'  + dataset_type + '_' + sample_name + '_' + name + '_' + plane_dim + '_pred.pdf'
    plt.savefig(filename)
    #plt.show()


def plot_iso_surface(opt, samples, preds, name, sample_name, dataset_type):
    # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
    # otherwise create interactive plot
    OFF_SCREEN = False
    if OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)

    sample_x = samples[0, 0, :].detach().cpu().numpy()
    sample_y = samples[0, 1, :].detach().cpu().numpy()
    sample_z = samples[0, 2, :].detach().cpu().numpy()
    sample = np.vstack((sample_x, sample_y, sample_z)).T
    pred = preds.detach().cpu().numpy()

    # interpolate point cloud to 2D-plane
    grid_res = complex(0, opt.resolution)
    X, Y, Z = np.mgrid[-128:128:grid_res, -28:228:grid_res, -128:128:grid_res]

    pred_interpn = griddata(sample, pred, (X,Y,Z), method='linear')
    pred_linear = griddata(sample, pred, (X,Y,Z), method='nearest')
    pred_interpn[np.isnan(pred_interpn)] = pred_linear[np.isnan(pred_interpn)]

    mesh = pyvista.StructuredGrid(X, Y, Z)
    mesh.point_data['values'] = pred_interpn.ravel(order='F')

    vmin = pred.min()
    vmax = pred.max()
    labels = dict(zlabel='Z', xlabel='X', ylabel='Y')
    contours = mesh.contour(np.linspace(vmin, vmax, 10))

    camera = pyvista.Camera()
    camera.position = (700.0, 100.0, 700.0)
    camera.focal_point = (5.0, 20.0, 5.0)

    if OFF_SCREEN:
        p = pyvista.Plotter(off_screen=True)
    else:
        p = pyvista.Plotter()

    p.add_mesh(mesh.outline(), color="k")
    p.add_mesh(contours, opacity=0.25, clim=[vmin, vmax])
    p.show_grid(**labels)
    p.add_axes(**labels)

    p.camera = camera

    if OFF_SCREEN:
        filename = 'results/' + dataset_type + '_' + sample_name + '_' + name + '_pred_3d.svg'
        p.save_graphic(filename)
        # p.screenshot(filename, transparent_background=True)
        p.close()
    else:
        p.show()


def plot_im_feat(im_feat):
    print(im_feat.shape)
    feature_map = im_feat.detach().cpu().numpy()


    # Reshape the tensor to (4, 4, 128, 128)
    tensor_reshaped = feature_map[0, :, :, :].reshape(16, 16, 128, 128)

    # Concatenate along the rows (vertical direction) first
    vert_concatenate = np.concatenate(np.split(tensor_reshaped, 16, axis=0), axis=2)

    # Concatenate along the columns (horizontal direction)
    image_2d = np.concatenate(np.split(vert_concatenate, 16, axis=1), axis=3)

    image_2d = image_2d[0, 0, :, :]

    # Display the result
    plt.imshow(image_2d)
    plt.title('Combined 2D Image')
    plt.show()


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,H)
    """
    #grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    grayscale_im = np.max(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    #heatmap = heatmap.filter(ImageFilter.GaussianBlur(radius=2))
    #no_trans_heatmap = no_trans_heatmap.filter(ImageFilter.GaussianBlur(radius=2))

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def plot_saliency_map(grads, train_data, opt):
    img_int = 1.25
    dir_path = './results/' + opt.name + '/saliency_map/'

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    yid = str(train_data['yid'] * 10)
    name = str(train_data['name'])
    print('timestep: ', name, 'rotation angle: ', yid)


    # Normalize gradients
    grads = grads - grads.min()
    grads /= grads.max()
    grayscale_grads = convert_to_grayscale(grads)

    raw_img = np.uint8((np.transpose(train_data['raw_img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    raw_img = Image.fromarray(np.uint8(raw_img)).convert('RGB')

    img = np.uint8((np.transpose(train_data['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(np.uint8(img)).convert('RGB')

    cam = (grayscale_grads - np.min(grayscale_grads)) / (
                np.max(grayscale_grads) - np.min(grayscale_grads))  # Normalize between 0-1
    cam = np.uint8(cam[0, :, :] * 255)  # Scale between 0-255 to visualize

    # Grayscale activation map
    # cmap = 'seismic'
    # cmap = 'bwr'
    cmap = 'coolwarm'
    heatmap, heatmap_on_image = apply_colormap_on_image(img, cam, cmap)

    heatmap_on_image = np.array(heatmap_on_image)
    heatmap = np.array(heatmap)
    img = np.array(img)
    raw_img = np.array(raw_img)
    grads = np.transpose(grads, (1, 2, 0))
    grayscale_grads = np.transpose(grayscale_grads, (1, 2, 0))
    
    img =  np.uint8(np.clip(img * img_int, a_min=0, a_max=255))
    raw_img =  np.uint8(np.clip(raw_img * img_int, a_min=0, a_max=255))

    # Plotting

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    plt.rcParams['figure.constrained_layout.use'] = True

    # fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(15, 6.5))
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(10, 4.25))
    ax1, ax2, ax3 = axes.flatten()
    # Note: grads are in BGR format
    p1 = ax1.imshow(heatmap / 255,
                    vmin=0, vmax=1, cmap=cmap)
    p2 = ax2.imshow(raw_img)
    px = ax3.imshow(heatmap / 255,
                    vmin=0, vmax=1)

    # ax1.set_title(r'saliency map', fontsize=16, y=1, pad=20)
    # ax2.set_title(r'input image', fontsize=16, y=1, pad=20)
    # ax3.set_title(r'image and map', fontsize=16, y=1, pad=20)

    cbar1 = fig.colorbar(p1, ax=ax1, location='bottom')
    cbar2 = fig.colorbar(p2, ax=ax2, location='bottom')
    cbar3 = fig.colorbar(px, ax=ax3, location='bottom')
    cbar1.ax.tick_params(labelsize=16, which='major')
    cbar2.ax.tick_params(labelsize=16, which='major')
    cbar3.ax.tick_params(labelsize=16, which='major')
    px.set_cmap(cmap)
    cbar2.remove()

    p3 = ax3.imshow(heatmap_on_image)
    cbar_ticks = [0.0, 0.5, 1.0]
    cbar1.set_ticks(cbar_ticks)
    cbar3.set_ticks(cbar_ticks)
    # cbar1.ax.locator_params(nbins=5)
    # cbar3.ax.locator_params(nbins=5)
    fig.execute_constrained_layout()

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    filename = dir_path + name + '_' + yid + '_smap.pdf'
    plt.savefig(filename, format='pdf', dpi=1200)

    # fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
    # ax1, ax2, ax3 = axes.flatten()
    # Note: grads are in BGR format
    # ax1.imshow(grads[:, :, 2])
    # ax2.imshow(grads[:, :, 1])
    # ax3.imshow(grads[:, :, 0])
    # fig.suptitle('Gradients of prediction wrt input image, red (left), green (middle), blue channel (right)')
    # plt.show()

    # filename = dir_path + name + '_' + yid + '_grads.pdf'
    # plt.savefig(filename, format='pdf', dpi=1200)


def calculate_image_gradients(img):
    blue_channel = img[:, :, 2]
    _, binary_blue = cv2.threshold(blue_channel, 100, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # Create a mask and fill the holes
    mask = np.zeros_like(binary_blue)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Calculate gradients using Sobel operator on the filled blue channel
    grad_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=5)  # Gradient in x direction
    grad_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=5)  # Gradient in y direction
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    magnitude_max = np.percentile(magnitude, 99.5)
    magnitude_min = np.min(magnitude)
    magnitude = (np.clip((magnitude - magnitude_min) / (magnitude_max - magnitude_min), 0.4, 1))
    magnitude = np.uint8(magnitude * 255)  # Scale between 0-255 to visualize

    contour = cv2.bitwise_not(magnitude)

    return contour


def plot_Score_CAM(cam, train_data, layer, opt):
    img_int = 1.5
    eps = 10 ** -8
    I_cut_off = 0.03 # lower cut off value for logarithmic plotting
    dir_path = './results/' + opt.name + '/Score_CAM/'
    # Define font size variables
    font_size = 16
    font_size_ticks = 14

    # overlayed activation map
    cmap = 'plasma'
    # stand-alone activation map ['viridis', jet', coolwarm']
    cmap2 = 'plasma'

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    yid = str(train_data['yid'] * 10)
    name = str(train_data['name'])
    print('timestep: ', name, 'rotation angle: ', yid)

    raw_img = np.uint8((np.transpose(train_data['raw_img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    raw_img = Image.fromarray(np.uint8(raw_img)).convert('RGB')

    img = np.uint8((np.transpose(train_data['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    contour = calculate_image_gradients(img)
    img = Image.fromarray(np.uint8(img)).convert('RGB')

    # calculate logarithmic values of activation intensity
    cam = np.where((cam < I_cut_off) | (cam > 1), I_cut_off, cam)
    cam_log = np.log10(cam + eps)
    cam_log = (cam_log - np.min(cam_log)) / (np.max(cam_log) - np.min(cam_log)) * 255
    cam_log = cam_log.astype(np.uint8)

    heatmap, heatmap_on_image = apply_colormap_on_image(img, cam_log, cmap2)

    # Overlay the contour onto the heatmap
    contour = Image.fromarray(np.uint8(contour))
    alpha_mask = contour.point(lambda p: 255 if p < 28 else 0)
    contour = ImageOps.invert(contour)
    contour_rgba = contour.convert('RGBA')
    contour_rgba.putalpha(alpha_mask)
    contour_on_heatmap = Image.alpha_composite(heatmap, contour_rgba)


    # Cropping
    box = (50, 240, 512-50, 512)
    raw_img = raw_img.crop(box)
    heatmap = heatmap.crop(box)
    contour_on_heatmap = contour_on_heatmap.crop(box)
    cam = cam[240:512, 50:512 - 50]

    heatmap_on_image = np.array(heatmap_on_image)
    heatmap = np.array(heatmap)/255.0
    img = np.array(img)
    raw_img = np.array(raw_img)

    img = np.uint8(np.clip(img * img_int, a_min=0, a_max=255))
    raw_img = np.uint8(np.clip(raw_img * img_int, a_min=0, a_max=255))


    # Plotting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    plt.rcParams['figure.constrained_layout.use'] = True

    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(10, 4.25))
    ax1, ax2, ax3 = axes.flatten()
    px1 = ax1.matshow(cam, cmap=cmap, norm=LogNorm(vmin=I_cut_off, vmax=1))
    p2 = ax2.matshow(cam, cmap=cmap, norm=LogNorm(vmin=I_cut_off, vmax=1))
    px3 = ax3.matshow(cam, cmap=cmap, norm=LogNorm(vmin=I_cut_off, vmax=1))

    cbar1 = fig.colorbar(px1, ax=ax1, location='bottom', shrink=0.9, pad=0.01)
    cbar2 = fig.colorbar(p2, ax=[ax2, ax3], location='bottom', shrink=0.5, aspect=20, pad=0.01)
    cbar2.ax.tick_params(labelsize=font_size_ticks)
    cbar2.set_label('Activation intensity $I_A$ [-]', fontsize=font_size, labelpad=0.1)

    p1 = ax1.matshow(raw_img)
    p3 = ax3.matshow(contour_on_heatmap)
    cbar_ticks = [0.01, 0.1, 1.0]
    cbar2.set_ticks(cbar_ticks)
    cbar1.remove()

    # Axis labels
    ax1.set_xlabel('$x$ [px]', fontsize=font_size, labelpad=5)
    ax1.set_ylabel('$y$ [px]', fontsize=font_size, labelpad=5)
    ax2.set_xlabel('$x$ [px]', fontsize=font_size, labelpad=5)
    ax3.set_xlabel('$x$ [px]', fontsize=font_size, labelpad=5)

    # Set tick parameters for axes
    ax1.tick_params(axis='both', labelsize=font_size_ticks, which='major')
    ax2.tick_params(axis='both', labelsize=font_size_ticks, which='major')
    ax3.tick_params(axis='both', labelsize=font_size_ticks, which='major')

    filename = dir_path + name + '_' + yid + '_' + layer + '_score_cam.svg'
    plt.savefig(filename, format='svg', dpi=1200)

    ''' separate plotting '''
    # input image
    fig, ax = plt.subplots(figsize=(3, 4.25))
    p = ax.imshow(raw_img)
    fig.execute_constrained_layout()
    # Axis labels
    ax.set_xlabel('$x$ [px]', fontsize=font_size, labelpad=5)  # Set your label
    ax.set_ylabel('$y$ [px]', fontsize=font_size, labelpad=5)  # Set your label
    ax.tick_params(axis='both', labelsize=font_size_ticks, which='major')
    filename = dir_path + name + '_' + yid + '_' + layer + '_raw_img_score_cam.svg'
    plt.savefig(filename, format='svg', dpi=1200)

    # heatmap
    fig, ax = plt.subplots(figsize=(3, 4.25))
    p = ax.matshow(cam, cmap=cmap, norm=LogNorm(vmin=I_cut_off, vmax=1))
    cbar = fig.colorbar(p, ax=ax, location='bottom')
    cbar.ax.tick_params(labelsize=font_size_ticks, which='major')
    # cbar.set_ticks(cbar_ticks)
    fig.execute_constrained_layout()
    # Axis labels
    ax.set_xlabel('$x$ [px]', fontsize=font_size, labelpad=5)  # Set your label
    ax.set_ylabel('$y$ [px]', fontsize=font_size, labelpad=5)  # Set your label
    ax.tick_params(axis='both', labelsize=font_size_ticks, which='major')
    filename = dir_path + name + '_' + yid + '_' + layer + '_heatmap_score_cam.svg'
    plt.savefig(filename, format='svg', dpi=1200)

    # contour on heatmap
    fig, ax = plt.subplots(figsize=(3, 4.25))
    px = ax.matshow(cam, cmap=cmap, norm=LogNorm(vmin=I_cut_off, vmax=1))
    cbar = fig.colorbar(px, ax=ax, location='bottom')
    cbar.ax.tick_params(labelsize=font_size_ticks, which='major')
    cbar.set_ticks(cbar_ticks)
    p = ax.matshow(contour_on_heatmap)
    fig.execute_constrained_layout()
    # Axis labels
    ax.set_xlabel('$x$ [px]', fontsize=font_size, labelpad=5)  # Set your label
    ax.set_ylabel('$y$ [px]', fontsize=font_size, labelpad=5)  # Set your label
    ax.tick_params(axis='both', labelsize=font_size_ticks, which='major')
    filename = dir_path + name + '_' + yid + '_' + layer + '_heat_on_img_score_cam.svg'
    plt.savefig(filename, format='svg', dpi=1200)
