# `Skimming-Perusal' Tracking: A Framework for Real-Time and Robust Long-term Tracking

This code has been tested on 
- RTX 2080Ti
- CUDA 10.0 + cuDNN 7.6 / CUDA 9.0 + cuDNN 7.1.2
- Python 2.7
- Ubuntu 18.04.2 LTS

# Installation
- Create anaconda environment:
```bash
conda create -n SPLT python=2.7
conda activate SPLT
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

# Models
| Model | Size | Google Drive  | Baidu |
|:-----:|:----:|:-------------:|:---------:|
| SiamRPN | 215 MB | [model.ckpt-470277](https://drive.google.com/open?id=1t-rJSHWGgm_9VfqzZaLfhN5XZ8dotXSb)  | [Mirror](https://pan.baidu.com/s/1Ft-OorgWQIh7rvWvdGodUA) |
| Verifier | 178 MB | [V_resnet50_VID_N-65624](https://drive.google.com/open?id=1jsGkEUinQwvotwWJzsMzXNaHOYkJrPeh)  | [Mirror](https://pan.baidu.com/s/1gHAaFAwgX5ROfaucaaGafQ) |
| Skimming | 24 MB | [Skim](https://drive.google.com/open?id=10kqcAPw19fBLnoW4O0qQMUOAm7YgpWsg)  | [Mirror](https://pan.baidu.com/s/1XRAbBfiQ32Ey52LYTJzErw) |

- extract `model.ckpt-470277` to `./RPN`
- extract `V_resnet50_VID_N-65624` to `./Verifier`
- extract `Skim` to `./Skim`

# Demo
```bash
# modify 'PROJECT_PATH' in 'demo.py' 
python demo.py
```

# Evaluation on VOT
[raw resluts (vot-toolkt version 6.0.3)](https://github.com/iiau-tracker/SPLT/tree/master/results), 
- modify 'PROJECT_PATH' in 'RPN_Verifier_Skim_top3.py'
- add `set_global_variable('python', 'env -i <path/to/anaconda/envs/SPLT/bin/python>');` to `configuration.m`

# Training
...

# Citation
If you use SPLT or this code base in your work, please cite
```
@inproceedings{ iccv19_SPLT,
    title={`Skimming-Perusal' Tracking: A Framework for Real-Time and Robust Long-term Tracking},
    author={Yan, Bin and Zhao, Haojie and Wang, Dong and Lu, Huchuan and Yang, Xiaoyun},
    booktitle={IEEE International Conference on Computer Vision (ICCV)},
    year={2019}
}
```