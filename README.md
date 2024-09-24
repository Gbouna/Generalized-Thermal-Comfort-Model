# Generalized Thermal Comfort Model using Thermographic Images and Compact Convolutional Transformers

This is the official repository for **A Generalized Thermal Comfort Model using Thermographic Images and Compact Convolutional Transformers: Towards Scalable and Adaptive Occupant Comfort Optimization**, our paper published in the Journal of Building and Environment. 

# Prerequisites
Create a conda environment with Python 3.8

Install tensorflow_addons (check here: https://anaconda.org/esri/tensorflow-addons)

If the latest tensorflow_addons is not compatible with your Cuda version, check for a compatible version here: https://anaconda.org/Esri/tensorflow-addons/files

Install matplotlib (check here: https://anaconda.org/conda-forge/matplotlib)

Install scikit-learn (check here: https://anaconda.org/conda-forge/scikit-learn)

Alternatively, you can use the dependency file of our environment to create an environment with all dependencies

```
Create the environment using the following command

conda env create -f environment.yml

Then, activate it using the following command

conda activate tf

```
`
# Data Preparation

Your dataset should have the following structure: 
```
- all_data/
  - participant 1/
  - participant 2/
  ...
    - data/
      - Cool
      - Neutral
      - Warm
```

# Training

### LOSO Training:

Run the following script for LOSO training `python main.py --train_type loso --base_dir /path/to/base_dir --checkpoint_dir /path/to/checkpoint_dir`

### LOGO Training:

Run the following script for LOGO training `python main.py --train_type logo --base_dir /path/to/base_dir --checkpoint_dir /path/to/checkpoint_dir`
