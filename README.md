# `Skimming-Perusal' Tracking: A Framework for Real-Time and Robust Long-term Tracking

![splt](https://github.com/iiau-tracker/SPLT/blob/master/results/SPLT.png)


This is the **python 3.6** version code for the ICCV 2019 paper SPLT[[arxiv]](https://arxiv.org/abs/1909.01840). This code has been tested on 
- RTX 2080Ti
- CUDA 10.0 + cuDNN 7.6 / CUDA 9.0 + cuDNN 7.1.2
- Python 3.6
- Ubuntu 18.04.2 LTS

The F-score on VOT18-LT35 of this code is 0.610, which is slightly lower than that of origin branch(0.616). 
However, performance on some videos is actually better than original version. So feel free to try this code :) 

Please cite our paper if you find it useful for your research.
```
@inproceedings{ iccv19_SPLT,
    title={`Skimming-Perusal' Tracking: A Framework for Real-Time and Robust Long-term Tracking},
    author={Yan, Bin and Zhao, Haojie and Wang, Dong and Lu, Huchuan and Yang, Xiaoyun},
    booktitle={IEEE International Conference on Computer Vision (ICCV)},
    year={2019}
}
```
## Raw Results
The raw experimental results on **VOT2018-LT**, **VOT2019LT** and **LaSOT** benchmarks can be found in [Google Drive](https://drive.google.com/drive/folders/1ZsavuH8LwUD4zG-Y7WEfpEmDadNX__mt)
 or [Baidu Drive](https://pan.baidu.com/s/1PUkqNmVwczQAvrw8ZqiQOg) (extraction code: gc9g).
 
The raw experimental results on **[TLP](https://amoudgl.github.io/tlp/)** benchmark can be found in [Baidu Drive](https://pan.baidu.com/s/1ZI-hOBLFXrBbhDnpTlSllQ) (extraction code: 2qz2).


## Installation

- Create anaconda environment:
```bash
conda create -n SPLT36 python=3.6
conda activate SPLT36
```

- Clone the repo and install requirements:
```bash
git clone https://github.com/iiau-tracker/SPLT.git
cd <path/to/SPLT>
pip install -r requirements.txt
```

- CUDA and cuDNN:
```bash
conda install cudatoolkit=10.0
conda install cudnn=7.6.0

# or CUDA 9.0 + cuDNN 7.1.2 for TensorFlow  < 1.13.0
conda install cudatoolkit=9.0
conda install cudnn=7.1.2
```
- Add object_detection to environment variable
```
sudo gedit ~/.bashrc
# go to the end of the file, add the following command.
export PYTHONPATH=<SPLT_PATH>/lib/object_detection:$PYTHONPATH
# Replace <SPLT_PATH> with your real path
```
## Models
| Model | Size | Google Drive  | Baidu |
|:-----:|:----:|:-------------:|:---------:|
| SiamRPN | 215 MB | [model.ckpt-470277](https://drive.google.com/open?id=1t-rJSHWGgm_9VfqzZaLfhN5XZ8dotXSb)  | [Mirror](https://pan.baidu.com/s/1Ft-OorgWQIh7rvWvdGodUA) |
| Verifier | 178 MB | [V_resnet50_VID_N-65624](https://drive.google.com/open?id=1jsGkEUinQwvotwWJzsMzXNaHOYkJrPeh)  | [Mirror](https://pan.baidu.com/s/1gHAaFAwgX5ROfaucaaGafQ) |

- extract `model.ckpt-470277` to `./RPN`
- extract `V_resnet50_VID_N-65624` to `./Verifier`

## Demo
```bash
# modify 'PROJECT_PATH' in 'demo.py' 
python demo.py
```

## Evaluation on VOT
start from `RPN_Verifier_Skim_top3.py`

- modify `PROJECT_PATH` in `RPN_Verifier_Skim_top3.py`
- add `set_global_variable('python', 'env -i <path/to/anaconda/envs/SPLT/bin/python>');` to `configuration.m`

[raw resluts (vot-toolkt version 6.0.3)](https://github.com/iiau-tracker/SPLT/tree/master/results)

## Train
### Train the Verifier(optional)
Download [ResNet50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz) model pretrained on IMAGENET.Then put extracted ckpt file in train_Verifier/lib
```bash
cd train_Verifier/experiments
# modify paths in classify.py
python classify.py
# modify paths in triplet_pairs.py
python triplet_pairs.py
# modify paths in train_multi_gpu.py
python train_multi_gpu.py
```

