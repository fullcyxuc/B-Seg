# B-Seg
Code for our SIGGRAPH'2023 paper: "UrbanBIS: a Large-scale Benchmark for Fine-grained Urban Building Instance Segmentation"

![Pipeline Image](https://github.com/fullcyxuc/B-Seg/blob/main/docs/methodPipeline.jpg)

## Installation

### Requirements
* Python 3.6.0 or above
* Pytorch 1.2.0 or above
* CUDA 10.0 or above 

### Virtual Environment
```
conda create -n bseg python==3.6
source activate bseg
```

### Install B-Seg

(1) Clone from the repository.
```
git clone https://github.com/fullcyxuc/B-Seg.git
cd B-Seg
```

(2) Install the dependent libraries.
```
pip install -r requirements.txt
conda install -c bioconda google-sparsehash 
```

(3) For the SparseConv, we apply the implementation of [spconv](https://github.com/traveller59/spconv) as [Pointgroup](https://github.com/dvlab-research/PointGroup) did. The repository is recursively downloaded at step (1). We use the version 1.0 of spconv. 

**Note:** it was modify `spconv\spconv\functional.py` to make `grad_output` contiguous. Make sure you use the modified `spconv`.

* First please download the [spconv](https://github.com/traveller59/spconv), and put it into lib directory

* To compile `spconv`, firstly install the dependent libraries. 
```
conda install libboost
conda install -c daleydeng gcc-5 # need gcc-5.4 for sparseconv
```
Add the `$INCLUDE_PATH$` that contains `boost` in `lib/spconv/CMakeLists.txt`. (Not necessary if it could be found.)
```
include_directories($INCLUDE_PATH$)
```

* Compile the `spconv` library.
```
cd lib/spconv
python setup.py bdist_wheel
```

* Run `cd dist` and use pip to install the generated `.whl` file.



(4) We also use other cuda and cpp extension([pointgroup_ops](https://github.com/dvlab-research/PointGroup/tree/master/lib/pointgroup_ops),[pcdet_ops](https://github.com/yifanzhang713/IA-SSD/tree/main/pcdet/ops)), and put them into the lib, to compile them:
```
cd lib/**  # (** refer to a specific extension)
python setup.py develop
```


## Data Preparation

(1) Download the [UranBIS](https://vcc.tech/urbanbis) training set and test set for the building instance segmentation

(2) Put the data in the corresponding folders, which are organized as follows.
```
B-Seg
├── dataset
│   ├── UrbanBIS
│   │   ├── original
│   │   │   ├── Qingdao
│   │   │   │   ├── train
│   │   │   │   │   ├── Areax.txt 
│   │   │   │   ├── test
│   │   │   │   │   ├── Areax.txt 
│   │   │   │   ├── val
│   │   │   │   │   ├── Areax.txt 
│   │   │   ├── Wuhu
│   │   │   │   ├── train
│   │   │   │   │   ├── Areax.txt 
│   │   │   │   ├── test
│   │   │   │   │   ├── Areax.txt 
│   │   │   │   ├── val
│   │   │   │   │   ├── Areax.txt 
...
```
(3) Preprocess and generate the block files `_inst_nostuff.pth` for building instance segmentation. 
```
cd dataset/UrbanBIS
python prepare_data_inst_instance_UrbanBIS.py
```
then, it will create a `processed` folder under the `UrbanBIS` folder, which contains the files for training and testing. That will be:
```
B-Seg
├── dataset
│   ├── UrbanBIS
│   │   ├── original
│   │   ├── processed
│   │   │   ├── Qingdao
│   │   │   │   ├── train
│   │   │   │   │   ├── X.pth or X.txt 
│   │   │   │   ├── test_w_label
│   │   │   │   │   ├── X.pth or X.txt 
│   │   │   │   ├── test_w_label_gt
│   │   │   │   │   ├── X.txt 
│   │   │   │   ├── val
│   │   │   │   │   ├── X.pth or X.txt 
│   │   │   │   ├── val_gt
│   │   │   │   │   ├── X.txt 
│   │   │   ├── Wuhu
│   │   │   │   ├── train
│   │   │   │   │   ├── X.pth or X.txt 
│   │   │   │   ├── test_w_label
│   │   │   │   │   ├── X.pth or X.txt 
│   │   │   │   ├── test_w_label_gt
│   │   │   │   │   ├── X.txt 
│   │   │   │   ├── val
│   │   │   │   │   ├── X.pth or X.txt 
│   │   │   │   ├── val_gt
│   │   │   │   │   ├── X.txt 
...
```


By default, it only processes the `Qingdao` city scene, and this can be changed at the `line 177` in the `prepare_data_inst_instance_UrbanBIS.py` file.


## Training
```
CUDA_VISIBLE_DEVICES=0 python train.py --config config/BSeg_default_urbanbis.yaml
```

## Inference and Evaluation
For evaluation, please set `eval` as `True` in the config file, and set `split` as `val` for validation set or `test_w_label` for testing set with labels
```
CUDA_VISIBLE_DEVICES=0 python test.py --config config/BSeg_default_urbanbis.yaml
```

## Acknowledgement
This repo is built upon several repos, e.g., [Pointgroup](https://github.com/dvlab-research/PointGroup), [HAIS](https://github.com/hustvl/HAIS), [DyCo3D](https://github.com/aim-uofa/DyCo3D), [SoftGroup](https://github.com/thangvubk/SoftGroup), [DKNet](https://github.com/W1zheng/DKNet), [SparseConvNet](https://github.com/facebookresearch/SparseConvNet), [spconv](https://github.com/traveller59/spconv), [IA-SSD](https://github.com/yifanzhang713/IA-SSD/tree/main/pcdet/ops) and [STPLS3D](https://github.com/meidachen/STPLS3D.git). 

