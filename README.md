# OCFD-Net_pytorch
This is the official repo for our work 'Learning Occlusion-aware Coarse-to-Fine Depth Map for Self-supervised Monocular Depth Estimation' (ACM-MM' 2022).  
[Paper](https://dl.acm.org/doi/10.1145/3503161.3548381)  
Citation information:
```
@inproceedings{zhou2022learning,
title = {Learning Occlusion-Aware Coarse-to-Fine Depth Map for Self-Supervised Monocular Depth Estimation},
author = {Zhou, Zhengming and Dong, Qiulei},
booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
pages = {6386–6395},
year = {2022},
publisher = {Association for Computing Machinery},
doi = {10.1145/3503161.3548381},
}
```

## Setup
We built and ran the repo with CUDA 11.0, Python 3.7.11, and Pytorch 1.7.0. For using this repo, we recommend creating a virtual environment by [Anaconda](https://www.anaconda.com/products/individual). Please open a terminal in the root of the repo folder for running the following commands and scripts.
```
conda env create -f environment.yml
conda activate pytorch170cu11
```

## Pre-trained models
|Model Name|Dataset(s)|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|
|----------|----------|--------|-------|----|-------|--|
|[OCFD-Net_384](https://pan.baidu.com/s/1yaLFyzKeWiPNlJotvP9mRg)|K|0.091|0.576|4.036|0.174|0.901|
|[OCFD-Net_CS+K_384](https://pan.baidu.com/s/1-rzjbp_mk1qQ7nnEHLfsgg)|CS+K|0.088|0.554|3.944|0.171|0.909|

* **code for all the download links is `ocfd`**
## Prediction
To predict depth maps for your images, please firstly download the pretrained model from the column named `Model Name` in the above table. After unzipping the downloaded model, you could predict the depth maps for your images by
```
python predict.py\
 --image_path <path to your image or folder name for your images>\
 --exp_opts options/_base/networks/ocfd_net.yaml\
 --model_path <path to the downloaded or trained model (.pth)>
```
You also could set `--input_size` to decide the size that the images are reshaped before they are input to the model. If you want to predict on CPU, please set `--cpu`. The depth results `<image name>_pred.npy` and the visualization results `<image name>_visual.png` will be saved in the same folder as the input images.  

## Data preparation
#### Set Data Path
We give an example `path_example.py` for setting the path in the repository.
Please create a python file named `path_my.py` and copy the code in `path_example.py` to the `path_my.py`. Then you can replace the used paths to your folder in the `path_my.py`.
the folder for each dataset should be organized like:
```
<root of kitti>
|---2011_09_26
|   |---2011_09_26_drive_0001_sync
|   |   |---image_02
|   |   |---image_03
|   |   |---velodyne_points
|   |   |---...
|   |---2011_09_26_drive_0002_sync
|   |   |---image_02
|   |   |---image_03
|   |   |---velodyne_points
|   |   |---...
|   '''
|---2011_09_28
|   |--- ...
|---gt_depths_raw.npz (for raw Eigen test set)
|---gt_depths_improved.npz (for improved Eigen test set)
```
```
<root of cityscapes>
|---leftImg8bit
|   |---train
|   |   |---aachen
|   |   |   |---aachen_000000_000019_leftImg8bit.png
|   |   |   |---aachen_000001_000019_leftImg8bit.png
|   |   |   |---...
|   |   |---bochum
|   |   |---...
|   |---train_extra
|   |   |---augsburg
|   |   |---...
|   |---test
|   |   |---...
|   |---val
|   |   |---...
|---rightImg8bit
|   |--- ...
|---camera
|   |--- ...
|---disparity
|   |--- ...
|---gt_depths (for evaluation)
|   |---000_depth.npy
|   |---001_depth.npy
|   |--- ...
```
```
<root of Make3D>
|---Gridlaserdata
|   |---depth_sph_corr-10.21op2-p-015t000.mat
|   |---depth_sph_corr-10.21op2-p-139t000.mat
|   |---...
|---Test134
|   |---img-10.21op2-p-015t000.jpg
|   |---img-10.21op2-p-139t000.jpg
|   |---...
```
#### KITTI
For training the methods on the KITTI dataset (the Eigen split), you should download the entire KITTI dataset (about 175GB) by:
```
wget -i ./datasets/kitti_archives_to_download.txt -P <save path>
```
And you could unzip them with:
```
cd <save path>
unzip "*.zip"
```

For evaluating the methods on the KITTI (Eigen raw test set), you should further generate the ground-truth depth file by (as done in the [Monodepth2](https://github.com/nianticlabs/monodepth2)):

```
python datasets/utils/export_kitti_gt_depth.py --data_path <root of KITTI> --split raw
```
If you want to evaluate the method on the KITTI improved test set, you should download the `annotated depth maps` (about 15GB) at [Here](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) and unzip it in the root of the KITTI dataset. Then you could generate the imporved ground-truth depth file by:
```
python datasets/utils/export_kitti_gt_depth.py --data_path <root of KITTI> --split improved
```
As an alternative, we provide the Eigen test subset and the generated `gt_depth` files [Here](https://pan.baidu.com/s/1NejtxajjJt6pQ-VIRJDcUg) (about 2GB) for the people who just want to do the evaluation.
##### Cityscapes (Optional)
Cityscapes could be used to jointly train the model with KITTI, which is helpful to improve the performance of the model. If you want to use the Cityscapes, please download the following parts of the dataset at [Here](https://www.cityscapes-dataset.com/downloads/) and unzip them to your `<root of cityscapes>` (Note: For some files, you should apply for download permission by email.):
```
leftImg8bit_trainvaltest.zip (11GB)
leftImg8bit_trainextra.zip (44GB)
rightImg8bit_trainvaltest.zip (11GB)
rightImg8bit_trainextra.zip (44GB)
camera_trainvaltest.zip (2MB)
camera_trainextra.zip (8MB)
```
Then, please generate the camera parameter matrices by:
```
python datasets/utils/export_cityscapes_matrix.py
```
##### Make3D (Optional)
Make3D could be used to evaluate the OCFD-Net for testing the cross-dataset generalization ability. If you want to evaluate on the Make3D, please download the test set (named `Test 134 images` and `Test 134 depths` ) of it at [Here](http://make3d.cs.cornell.edu/data.html) and unzip them to your `<root of make3d>`.
## Evaluation
To evaluate the methods on the prepared dataset, you could simply use 
```
python evaluate.py\
 --exp_opts <path to the method EVALUATION option>\
 --model_path <path to the downloaded or trained model>\
```
We provide the EVALUATION option files in `options/OCFD-Net/eval/*`. Here we introduce some important arguments.
|Argument|Information|
|--------|-----------|
|`--visual_list`|The samples which you want to save the output (path to a `.txt` file)|
|`--save_pred`|Save the predicted depths of the samples which are in `--visual_list`|
|`--save_visual`|Save the visualization results of the samples which are in `--visual_list`|
|`-fpp`|Adopt the post-processing step.|
|`--metric_name`|Adopt different metrics for different dataset (please use 'depth_m3d' for Make3D). Default: 'depth_kitti'.|

The output files are saved in `eval_res\` by default. Please check `evaluate.py` for more information about arguments.

## Training
OCFD-Net could be trained by simply using the commands provided in `options/OCFD-Net/train/train_scripts.sh`.
For example, you could use the following commands for training the OCFD-Net on KITTI:
```
# train OCFD-Net with 192x640 patches
CUDA_VISIBLE_DEVICES=0 python\
 train_dist.py\
 --name OCFD-Net_192Crop_KITTI_S_B8\
 --exp_opts options/OCFD-Net/train/ocfd-net_192crop_kitti_stereo.yaml\
 --batch_size 8\
 --save_freq 10\
 --visual_freq 2000

```
Or use the following command to train the OCFD-Net on both KITTI and Cityscapes:
```
# train OCFD-Net with 192x640 patches
# on both kitti and cityscapes dataset
CUDA_VISIBLE_DEVICES=0 python\
 train_dist.py\
 --name OCFD-Net_192Crop_KITTI-Cityscapes_S_B8\
 --exp_opts options/OCFD-Net/train/ocfd-net_192crop_cityscapes-kitti_stereo.yaml\
 --batch_size 8\
 --save_freq 10\
 --visual_freq 2000
 ```

## Acknowledgment
Some of this repo come from [Monodepth2](https://github.com/nianticlabs/monodepth2).
