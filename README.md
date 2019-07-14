# WEGAN: Unsupervised Meta-learning of Figure-Ground Segmentation via Imitating Visual Effects
A clean and readable Pytorch implementation of WEGAN (https://arxiv.org/abs/1812.08442)

## Prerequisites
Code is intended to work with ```Python 3.6.x```, it hasn't been tested with previous versions

### [PyTorch & torchvision](http://pytorch.org/)
Follow the instructions in [pytorch.org](http://pytorch.org) for your current setup

## Training
### 1. Setup the dataset
First, you will need to download and setup a dataset. Recommended using [MSRA10K](http://mftp.mmcheng.net/Data/MSRA10K_Imgs_GT.zip) dataset. Unzip the file and put it in the datasets folder. 

```
mkdir datasets
```
### 2. Train!
```
python train.py --cuda
```
This command will start a training session using the images  MSRA10K under the *./datasets/* directory.  You are free to change those hyperparameters. 

If you don't own a GPU remove the --cuda option, although I advise you to get one!

There are three types of visual effect to choose. (black-background, color-selectivo, defocus).

```
python train.py --visual_effect color-selectivo --cuda
```

Examples of the generated outputs (default params, MSRA10K dataset):

Input Image  -->  Output mask  -->  Output Image  -->  Ground truth Image

![Example_1](https://github.com/timy90022/WEGAN/result/191_3.jpg)

<img src="result/191_3.jpg" width="700px"/>

![Real zebra](https://github.com/ai-tor/PyTorch-CycleGAN/raw/master/output/real_B.jpg)
![Fake horse](https://github.com/ai-tor/PyTorch-CycleGAN/raw/master/output/fake_A.png)