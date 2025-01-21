# Droplet-PIFu
Interface reconstruction of adhering droplets for distortion correction using glare points and deep learning (code) \
by Maximilian Dreisbach (Institute of Fluid Mechanics (ISTM) - Karlsruhe Institute of Technology (KIT))

This code allows for the evalution and training of neural networks for spatio-temporal gas-liquid interface reconstruction in two-phase flows as presented 
in the research article "Interface reconstruction of adhering droplets for distortion correction using glare points and deep learning". \
The datasets used in this work, as well as the weights of the neural networks trained for interface reconstruction on these datasets are available here: https://doi.org/10.35097/egqrfznmr9yp2s7f

If you have any questions regarding this code, please feel free to contact Maximilian Dreisbach (maximilian.dreisbach@kit.edu).

## Requirements
- Python 3.7 (required packages below)
- PyTorch

## Tested for: 
(see requirements.txt)

## Getting Started
### Evaluation

python ./apps/eval.py 

## Related Research
This code is an extension of "PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization" by Saito et al. (2019) \
(see https://github.com/shunsukesaito/PIFu)
