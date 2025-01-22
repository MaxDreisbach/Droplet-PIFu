# Droplet-PIFu: Interface reconstruction of adhering droplets for distortion correction using glare points and deep learning
by Maximilian Dreisbach (Institute of Fluid Mechanics (ISTM) - Karlsruhe Institute of Technology (KIT))

[![Preprint](https://img.shields.io/badge/arxiv-preprint-blue)](https://arxiv.org/abs/2501.03453)


This code allows for the evalution and training of neural networks for spatio-temporal gas-liquid interface reconstruction in two-phase flows as presented 
in the research article "Interface reconstruction of adhering droplets for distortion correction using glare points and deep learning". \
The datasets used in this work, as well as the weights of the neural networks trained for interface reconstruction on these datasets are available here: https://doi.org/10.35097/egqrfznmr9yp2s7f

If you have any questions regarding this code, please feel free to contact Maximilian Dreisbach (maximilian.dreisbach@kit.edu).

## Requirements
- Python 3 (required packages below)
- PyTorch
- json
- PIL
- skimage
- tqdm
- numpy
- cv2
- matplotlib

for training and data generation
- trimesh with pyembree
- pyexr
- PyOpenGL
- freeglut

## Tested for: 
(see requirements.txt)

## Getting Started
- Create conda environment from requirements.txt (conda create --name <env> --file requirements.txt)
- Download pre-processed glare-point shadowgraphy images from https://doi.org/10.35097/egqrfznmr9yp2s7f
- OR use processing scripts on own data (see https://github.com/MaxDreisbach/GPS-Processing)
- Download network weights and move it to `./Droplet-PIFu/checkpoints/`
- OR train the network on new data (see below)
- Run eval.py forvolumetric reconstruction (see below)
- Open .obj file of reconstructed interface in Meshlab, Blender, or any 3D visualization software 

## Evaluation
This script reconstructs each image into an `.obj` file under `./PIFu/results/name_of_experiment` represnting the 3D gas-liquid interface.

`python -m apps.eval --name {name_of_experiment} --test_folder_path {path_to_processed_image_data} --load_netG_checkpoint_path {path_to_network_weights}`


## Data Generation (Linux Only)
The data generation uses codes adapted from PIFu by Saito et al. (2019), see https://github.com/shunsukesaito/PIFu for further reference.
The following code should be run with [pyembree](https://github.com/scopatz/pyembree), as it is otherwise very slow. \
The data generation requires `.obj` files of ground truth 3D gas-liquid interfaces, e.g. obtained by numerical simulation. 
First, binary masks, placeholder renderings, and calibration matrices are computed from the specified `.obj` files.
Then, physically-based rendering in Blender is used to generate realistic synethtic images resembling the recordings from the experiments.

1. The following script precomputes spherical harmonics coefficients for rendering. Adjust the path to the `.obj` files in `prt_util_batch.py`.
```
python -m apps.prt_util_batch
```
2. The following script creates renderings, masks, and calibration matrices representing the relative orientation of the renderings and 3D geometries. The files are saved in newly created folders named `GEO`, `RENDER`, `MASK`, `PARAM`, `UV_RENDER`, `UV_MASK`, `UV_NORMAL`, and `UV_POS` under the specified training data path. Adjust the path to the `.obj` files in `prt_util_batch.py`.
```
python -m apps.render_data_batch -i {path_to_OBJ} -o {path_to_training_data}
```
3. Run the synhetic training data generation in Blender https://github.com/MaxDreisbach/RenderGPS
4. Copy the renderings from blender into the `RENDER` folder

## Training (Linux Only)
The following code should be run with [pyembree](https://github.com/scopatz/pyembree), as it is otherwise very slow. \
1. Run the following script to train the reconstruction network. The intermediate checkpoints are saved under `./checkpoints`. You can add `--batch_size` and `--num_sample_input` flags to adjust the batch size and the number of sampled points based on available GPU memory. The flags `--random_scale`, and `--random_trans` enable data augmentation and perform random scaling and random cropping and translation of the images and associated 3D geometries.
```
python -m apps.train_shape --dataroot {path_to_training_data} --random_scale --random_trans
```

## Related Research
This code is an extension of "PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization" by Saito et al. (2019) \
(see https://github.com/shunsukesaito/PIFu)
