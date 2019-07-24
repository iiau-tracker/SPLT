# `Skimming-Perusal' Tracking: A Framework for Real-Time and Robust Long-term Tracking

This code has been tested on 
- RTX 2080Ti
- CUDA 10.0 + cudnn 7.6 / CUDA 9.0 + cudnn 7.1.2
- Python 2.7
- Ubuntu 18.04.2 LTS

# Installation
```bash
git clone https://github.com/iiau-tracker/SPLT.git
cd <path/to/SPLT>

conda create -n SPLT python=3.6
conda activate SPLT
pip install -r requirements.txt
```

```bash
conda install cudatoolkit=10.0
conda install cudnn=7.6.0
or
conda install cudatoolkit=9.0
conda install cudnn=7.1.2
```

# Models
| Model | Size | Google Drive  | Baidu Yun |
|:-----:|:----:|:-------------:|:---------:|
| SiamRPN | 215 MB | [model.ckpt-470277](https://drive.google.com/open?id=1t-rJSHWGgm_9VfqzZaLfhN5XZ8dotXSb)  | [Mirror]() |
| Verifier | 178 MB | [V_resnet50_VID_N-65624](https://drive.google.com/open?id=1jsGkEUinQwvotwWJzsMzXNaHOYkJrPeh)  | [Mirror]() |
| Skimming | 24 MB | [Skim](https://drive.google.com/open?id=10kqcAPw19fBLnoW4O0qQMUOAm7YgpWsg)  | [Mirror]() |

- extract `model.ckpt-470277` to `./RPN`
- extract `V_resnet50_VID_N-65624` to `./Verifier`
- extract `model.ckpt-470277` to `./Skim`

# Demo
```
# modify 'PROJECT_PATH' in 'demo.py' 
python demo.py
```

# Evaluation
...
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