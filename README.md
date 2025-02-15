# [DEVIANT: Depth EquiVarIAnt NeTwork for Monocular 3D Object Detection](https://arxiv.org/pdf/2207.10758.pdf)

### [KITTI Demo](https://www.youtube.com/watch?v=2D73ZBrU-PA) | [KITTI Eqv Error Demo](https://www.youtube.com/watch?v=70DIjQkuZvw) | [Waymo Demo](https://www.youtube.com/watch?v=46S_OGxYFOM) | [Project](http://cvlab.cse.msu.edu/project-deviant.html) | [Talk](https://www.youtube.com/watch?v=yDrLYjmER4M) | [Slides](https://docs.google.com/presentation/d/1sSH1ArzqWvyswgmqEJCXlxAJBaLY1qxzZX0w4NojUAA/edit?usp=sharing) | [Poster](https://docs.google.com/presentation/d/1NB5YuPNDhctkA2EHYGJ1NZYe6ttPsumIll4ef9BDji4/edit?usp=sharing)

[Abhinav Kumar](https://sites.google.com/view/abhinavkumar/)<sup>1</sup>, 
[Garrick Brazil](https://garrickbrazil.com/)<sup>2</sup>, 
[Enrique Corona](https://www.linkedin.com/in/enrique-corona-0752b84)<sup>3</sup>, 
[Armin Parchami](https://www.linkedin.com/in/parchami/)<sup>3</sup>, 
[Xiaoming Liu](http://www.cse.msu.edu/~liuxm/index2.html)<sup>1</sup> <br>
<sup>1</sup>Michigan State University, <sup>2</sup>Meta AI, <sup>3</sup>Ford Motor Company

in [ECCV 2022](https://eccv2022.ecva.net/)

<img src="code/images/waymo_detection_demo.gif" width="512">
<img src="code/images/idea_overview.png">

> Modern neural networks use building blocks such as convolutions that are equivariant to arbitrary 2D translations $(t_u, t_v)$. However, these vanilla blocks are not equivariant to arbitrary 3D translations $(t_x, t_y, t_z)$ in the projective manifold. Even then, all monocular 3D detectors use vanilla blocks to obtain the 3D coordinates, a task for which the vanilla blocks are not designed for. This paper takes the first step towards convolutions equivariant to arbitrary 3D translations in the projective manifold. Since the depth is the hardest to estimate for monocular detection, this paper proposes Depth EquiVarIAnt NeTwork (DEVIANT) built with existing scale equivariant steerable blocks. As a result, DEVIANT is equivariant to the depth translations $(t_z)$ in the projective manifold whereas vanilla networks are not. The additional depth equivariance forces the DEVIANT to learn consistent depth estimates, and therefore, DEVIANT achieves state-of-the-art monocular 3D detection results on KITTI and Waymo datasets in the image-only category and performs competitively to methods using extra information. Moreover, DEVIANT works better than vanilla networks in cross-dataset evaluation.


Much of the codebase is based on [GUP Net](https://github.com/SuperMHP/GUPNet). Some implementations are from [GrooMeD-NMS](https://github.com/abhi1kumar/groomed_nms) and [PCT](https://github.com/amazon-research/progressive-coordinate-transforms). Scale Equivariant Steerable (SES) implementations are from [SiamSE](https://github.com/ISosnovik/SiamSE).

[![arXiv](http://img.shields.io/badge/arXiv-2207.10758-B31B1B.svg)](https://arxiv.org/abs/2207.10758)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Citation

If you find our work useful in your research, please consider starring the repo and citing:

```Bibtex
@inproceedings{kumar2022deviant,
   title={{DEVIANT: Depth EquiVarIAnt NeTwork for Monocular $3$D Object Detection}},
   author={Kumar, Abhinav and Brazil, Garrick and Corona, Enrique and Parchami, Armin and Liu, Xiaoming},
   booktitle={ECCV},
   year={2022}
}
```

## Setup

- **Requirements**

    1. Python 3.7
    2. [PyTorch](http://pytorch.org) 1.10
    3. Torchvision 0.11
    4. Cuda 11.3
    5. Ubuntu 18.04/Debian 8.9

This is tested with NVIDIA A100 GPU. Other platforms have not been tested. Clone the repo first. Unless otherwise stated, the below scripts and instructions assume the working directory is the directory `code`:

```bash
git clone https://github.com/abhi1kumar/DEVIANT.git
cd DEVIANT/code
```

- **Cuda & Python**

Build the DEVIANT environment by installing the requirements:

```bash
conda create --name DEVIANT --file conda_GUP_environment_a100.txt
conda activate DEVIANT
pip install opencv-python pandas
```

- **KITTI, nuScenes and Waymo Data**

Follow instructions of [data_setup_README.md](code/data/data_setup_README.md) to setup KITTI, nuScenes and Waymo as follows:

```bash
./code
├── data
│      ├── KITTI
│      │      ├── ImageSets
│      │      ├── kitti_split1
│      │      ├── training
│      │      │     ├── calib
│      │      │     ├── image_2
│      │      │     └── label_2
│      │      │
│      │      └── testing
│      │            ├── calib
│      │            └── image_2
│      │
│      ├── nusc_kitti
│      │      ├── ImageSets
│      │      ├── training
│      │      │     ├── calib
│      │      │     ├── image
│      │      │     └── label
│      │      │
│      │      └── validation
│      │            ├── calib
│      │            ├── image
│      │            └── label
│      │
│      └── waymo
│             ├── ImageSets
│             ├── training
│             │     ├── calib
│             │     ├── image
│             │     └── label
│             │
│             └── validation
│                   ├── calib
│                   ├── image
│                   └── label
│
├── experiments
├── images
├── lib
├── nuscenes-devkit        
│ ...
```


- **AP Evaluation**

Run the following to generate the KITTI binaries corresponding to `R40`:

```bash
sudo apt-get install libopenblas-dev libboost-dev libboost-all-dev
sudo apt install gfortran
sh data/KITTI/kitti_split1/devkit/cpp/build.sh
```

We finally setup the Waymo evaluation. The Waymo evaluation is setup in a different environment `py36_waymo_tf` to avoid package conflicts with our DEVIANT environment:

```bash
# Set up environment
conda create -n py36_waymo_tf python=3.7
conda activate py36_waymo_tf
conda install cudatoolkit=11.3 -c pytorch

# Newer versions of tf are not in conda. tf>=2.4.0 is compatible with conda.
pip install tensorflow-gpu==2.4
conda install pandas
pip3 install waymo-open-dataset-tf-2-4-0 --user
```

To verify that your Waymo evaluation is working correctly, pass the ground truth labels as predictions for a sanity check. Type the following:

```bash
/mnt/home/kumarab6/anaconda3/envs/py36_waymo_tf/bin/python -u data/waymo/waymo_eval.py --sanity
```

You should see AP numbers as 100 in every entry after running this sanity check.


## Training

Train the model:

```bash
chmod +x scripts_training.sh
./scripts_training.sh
```

The current Waymo config files use the full val set in training. For Waymo models, we had subsampled Waymo validation set by a factor of 10 (4k images) to save training time as in [DD3D](https://github.com/TRI-ML/dd3d#models). Change `val_split_name` from `'val'` to `'val_small'` in waymo configs to use subsampled Waymo val set.

## Testing Pre-trained Models

### Model Zoo

We provide logs/models/predictions for the main experiments on KITTI Val /KITTI Test/Waymo Val data splits available to download here.

| Data Split | Method  | Run Name/Config Yaml                                                       | Weights  |
|------------|---------|----------------------------------------------------------------------------|----------|
| KITTI Val  | GUP Net | [config_run_201_a100_v0_1](code/experiments/config_run_201_a100_v0_1.yaml) | [gdrive](https://drive.google.com/file/d/17qezmIjckRSAva1fNnYBmgR9LaY-dPnp/view?usp=sharing)   |     
| KITTI Val  | DEVIANT | [run_221](code/experiments/run_221.yaml)                                   | [gdrive](https://drive.google.com/file/d/1CBJf8keOutXVSAiu9Fj7XQPQftNYC1qv/view?usp=sharing)   |
| KITTI Test | DEVIANT | [run_250](code/experiments/run_250.yaml)                                   | [gdrive](https://drive.google.com/file/d/1_79GfHcpAQR3wdvhj9GDHc7_c_ndf1Al/view?usp=sharing)   |
| Waymo Val  | GUP Net | [run_1050](code/experiments/run_1050.yaml)                                 | [gdrive](https://drive.google.com/file/d/1wuTTuZrFVsEv4ttQ0r3X_s8D3OjYE84E/view?usp=sharing)   |
| Waymo Val  | DEVIANT | [run_1051](code/experiments/run_1051.yaml)                                 | [gdrive](https://drive.google.com/file/d/1ixCVS85yVU9k6kuHrcYw_qJoy9Z4d0FD/view?usp=sharing)   |

### Testing

Make `output` folder in the `code` directory:

```bash
mkdir output
```
Place models in the `output` folder as follows:

```bash
./code
├── output
│      ├── config_run_201_a100_v0_1
│      ├── run_221
│      ├── run_250
│      ├── run_1050
│      └── run_1051
│
│ ...
```

Then, to test, run the file as:

```bash
chmod +x scripts_inference.sh
./scripts_inference.sh
```

### Cross-Dataset Evaluation of KITTI on nuScenes Frontal Val

See [scripts_inference.sh](code/scripts_inference.sh)

### Qualitative Plots

To get qualitative plots, type the following:

```bash
python plot/plot_qualitative_output.py --dataset waymo --folder output/run_1051/results_test/data
```

Type the following to reproduce our other plots:

```bash
python plot/plot_sesn_basis.py
python plot/visualize_output_of_cnn_and_sesn.py
```

## FAQ

- **Inference on older cuda version** For inference on older cuda version, type the following before running inference:

```bash
source cuda_9.0_env
```

- **Correct Waymo version** You should see a 16th column in each ground truth file inside `data/waymo/validation/label/`. This corresponds to the `num_lidar_points_per_box`. If you do not see this column, run:

```bash
cd data/waymo
python waymo_check.py 
```

to see if `num_lidar_points_per_box` is printed. If nothing is printed, you are using the wrong Waymo dataset version and you should download the correct dataset version.

- **Cannot convert a symbolic Tensor (strided_slice:0) to a numpy array**  This error indicates that you're trying to pass a Tensor to a NumPy call". This means you have a wrong numpy version. Install the correct numpy as:

```bash
pip install numpy==1.19.5
```

## Acknowledgements
We thank the authors of [GUP Net](https://github.com/SuperMHP/GUPNet), [GrooMeD-NMS](https://github.com/abhi1kumar/groomed_nms), [SiamSE](https://github.com/ISosnovik/SiamSE), [PCT](https://github.com/amazon-research/progressive-coordinate-transforms) and [patched nuscenes-devkit](https://github.com/abhi1kumar/nuscenes-devkit) for their awesome codebases. Please also consider citing them.

## Contact
For questions, feel free to post here or drop an email to this address- ```abhinav3663@gmail.com```

# Other Monocular 3D Object Detection Papers

Source: https://github.com/BigTeacher-777/Awesome-Monocular-3D-detection/blob/main/README.md

## Contents
- [Paper List](#Paper-List)
   	- [2023](#2023)
    - [2022](#2022)
    - [2021](#2021)
    - [2020](#2020)
    - [2019](#2019)
    - [2018](#2018)
    - [2017](#2017)
    - [2016](#2016)
- [KITTI Results](#KITTI-Results)

# Paper List
## 2023
- <a id="ADD"></a>**[ADD]** Attention-based Depth Distillation with 3D-Aware Positional Encoding for Monocular 3D Object Detection[[AAAI2023](https://arxiv.org/pdf/2211.16779.pdf)]
## 2022
- <a id="LPCG"></a>**[LPCG]** Lidar Point Cloud Guided Monocular 3D Object Detection [[ECCV2022](https://arxiv.org/abs/2104.09035)][[Pytorch](https://github.com/SPengLiang/LPCG)]
- <a id="MVC-MonoDet"></a>**[MVC-MonoDet]** Semi-Supervised Monocular 3D Object Detection by Multi-View Consistency [[ECCV2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680702.pdf)][[Pytorch](https://github.com/lianqing11/mvc_monodet)]
- <a id="CMKD"></a>**[CMKD]** Cross-Modality Knowledge Distillation Network for Monocular 3D Object Detection [[ECCV2022](https://arxiv.org/abs/2211.07171)][[Pytorch](https://github.com/Cc-Hy/CMKD)]
- <a id="DfM"></a>**[DfM]**  Monocular 3D Object Detection with Depth from Motion [[ECCV2022](https://arxiv.org/pdf/2207.12988.pdf)][[Pytorch](https://github.com/Tai-Wang/Depth-from-Motion)]
- <a id="DEVIANT"></a>**[DEVIANT]** DEVIANT: Depth EquiVarIAnt NeTwork for Monocular 3D Object Detection [[ECCV2022](https://arxiv.org/pdf/2207.10758.pdf)][[Pytorch](https://github.com/abhi1kumar/DEVIANT)]
- <a id="DCD"></a>**[DCD]** Densely Constrained Depth Estimator for Monocular 3D Object Detection [[ECCV2022](https://arxiv.org/pdf/2207.10047.pdf)][[Pytorch](https://github.com/BraveGroup/DCD)]
- <a id="STMono3D"></a>**[STMono3D]** Unsupervised Domain Adaptation for Monocular 3D Object Detection via Self-Training [[ECCV2022](https://arxiv.org/pdf/2204.11590.pdf)]
- <a id="DID-M3D"></a>**[DID-M3D]** DID-M3D: Decoupling Instance Depth for Monocular 3D Object Detection [[ECCV2022](https://arxiv.org/pdf/2207.08531.pdf)][[Pytorch](https://github.com/SPengLiang/DID-M3D)]
- <a id="SGM3D"></a>**[SGM3D]** SGM3D: Stereo Guided Monocular 3D Object Detection [[RA-L2022](https://arxiv.org/pdf/2112.01914.pdf)][[Pytorch](https://github.com/zhouzheyuan/sgm3d)]
- <a id="PRT"></a>**[PRT]** Depth Estimation Matters Most: Improving Per-Object Depth Estimation for Monocular 3D Detection and Tracking [[ICRA2022](https://arxiv.org/pdf/2206.03666.pdf)]
- <a id="Time3D"></a>**[Time3D]** Time3D: End-to-End Joint Monocular 3D Object Detection and Tracking for Autonomous Driving [[CVPR2022](https://arxiv.org/pdf/2205.14882.pdf)]
- <a id="MonoGround"></a>**[MonoGround]** MonoGround: Detecting Monocular 3D Objects from the Ground [[CVPR2022](https://arxiv.org/pdf/2206.07372.pdf)][[Pytorch](https://github.com/cfzd/MonoGround)]
- <a id="DimEmbedding"></a>**[DimEmbedding]** Dimension Embeddings for Monocular 3D Object Detection [[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Dimension_Embeddings_for_Monocular_3D_Object_Detection_CVPR_2022_paper.pdf)]
- <a id="GeoAug"></a>**[GeoAug]** Exploring Geometric Consistency for Monocular 3D Object Detection [[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Lian_Exploring_Geometric_Consistency_for_Monocular_3D_Object_Detection_CVPR_2022_paper.pdf)]
- <a id='MonoDDE'></a>**[MonoDDE]** Diversity Matters: Fully Exploiting Depth Clues for Reliable Monocular 3D Object Detection [[CVPR2022](https://arxiv.org/pdf/2205.09373.pdf)]
- <a id='Homography'></a>**[Homography]** Homography Loss for Monocular 3D Object Detection [[CVPR2022](https://arxiv.org/pdf/2204.00754.pdf)]
- <a id='Rope3D'></a>**[Rope3D]** Rope3D: TheRoadside Perception Dataset for Autonomous Driving and Monocular 3D Object Detection Task [[CVPR2022](https://arxiv.org/pdf/2203.13608.pdf)][[Pytorch](https://github.com/liyingying0113/rope3d-dataset-tools)]
- <a id='MonoDTR'></a>**[MonoDTR]** MonoDTR: Monocular 3D Object Detection with Depth-Aware Transformer [[CVPR2022](https://arxiv.org/pdf/2203.10981.pdf)][[Pytorch](https://github.com/kuanchihhuang/MonoDTR)]
- <a id='MonoJSG'></a>**[MonoJSG]** MonoJSG: Joint Semantic and Geometric Cost Volume for Monocular 3D Object Detection [[CVPR2022](https://arxiv.org/pdf/2203.08563.pdf)][[Pytorch](https://github.com/lianqing11/MonoJSG)]
- <a id='Pseudo-Stereo'></a>**[Pseudo-Stereo]** Pseudo-Stereo for Monocular 3D Object Detection in Autonomous Driving [[CVPR2022](https://arxiv.org/pdf/2203.02112.pdf)][[Pytorch](https://github.com/revisitq/Pseudo-Stereo-3D)]
- <a id='MonoDistill'></a>**[MonoDistill]** MonoDistill: Learning Spatial Features for Monocular 3D Object Detection [[ICLR2022](https://arxiv.org/pdf/2201.10830.pdf)][[Pytorch](https://github.com/monster-ghost/MonoDistill)]
- <a id='WeakM3D'></a>**[WeakM3D]** WeakM3D: Towards Weakly Supervised Monocular 3D Object Detection [[ICLR2022](https://openreview.net/pdf?id=ahi2XSHpAUZ)][[Pytorch](https://github.com/SPengLiang/WeakM3D)]
- <a id='MonoCon'></a>**[MonoCon]** Learning Auxiliary Monocular Contexts Helps Monocular 3D Object Detection [[AAAI2022](https://arxiv.org/pdf/2112.04628.pdf)][[Pytorch](https://github.com/Xianpeng919/MonoCon)]
- <a id='ImVoxelNet'></a>**[ImVoxelNet]** ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection [[WACV2022](https://arxiv.org/pdf/2106.01178.pdf)][[Pytorch](https://github.com/saic-vul/imvoxelnet)]


## 2021
- <a id="PCT"></a>**[PCT]** Progressive Coordinate Transforms for Monocular 3D Object Detection [[NeurIPS2021](https://arxiv.org/pdf/2108.05793.pdf)][[Pytorch](https://github.com/amazon-research/progressive-coordinate-transforms)]
- <a id="DeepLineEncoding"></a>**[DeepLineEncoding]** Deep Line Encoding for Monocular 3D Object Detection and Depth Prediction [[BMVC2021](https://www.bmvc2021-virtualconference.com/assets/papers/0299.pdf)][[Pytorch](https://github.com/cnexah/DeepLineEncoding)]
- <a id="DFR-Net"></a>**[DFR-Net]** The Devil Is in the Task: Exploiting Reciprocal Appearance-Localization Features for Monocular 3D Object Detection [[ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zou_The_Devil_Is_in_the_Task_Exploiting_Reciprocal_Appearance-Localization_Features_ICCV_2021_paper.html)]
- <a id="AutoShape"></a>**[AutoShape]** AutoShape: Real-Time Shape-Aware Monocular 3D Object Detection [[ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_AutoShape_Real-Time_Shape-Aware_Monocular_3D_Object_Detection_ICCV_2021_paper.pdf)][[Pytorch](https://github.com/zongdai/AutoShape)][[Paddle](https://github.com/zongdai/AutoShape)]
- <a id="analysis"></a>**[pseudo-analysis]** Are we Missing Confidence in Pseudo-LiDAR Methods for Monocular 3D Object Detection? [[ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Simonelli_Are_We_Missing_Confidence_in_Pseudo-LiDAR_Methods_for_Monocular_3D_ICCV_2021_paper.pdf)]
- <a id="Gated3D"></a>**[Gated3D]** Gated3D: Monocular 3D Object Detection From Temporal Illumination Cues [[ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Julca-Aguilar_Gated3D_Monocular_3D_Object_Detection_From_Temporal_Illumination_Cues_ICCV_2021_paper.pdf)]
- <a id="MonoRCNN"></a>**[MonoRCNN]** Geometry-based Distance Decomposition for Monocular 3D Object Detection [[ICCV2021](https://arxiv.org/abs/2104.03775)][[Pytorch](https://github.com/Rock-100/MonoDet)]
- <a id="DD3D"></a>**[DD3D]** Is Pseudo-Lidar needed for Monocular 3D Object detection [[ICCV2021](https://arxiv.org/pdf/2108.06417.pdf)][[Pytorch](https://github.com/tri-ml/dd3d)]
- <a id="GUPNet"></a>**[GUPNet]** Geometry Uncertainty Projection Network for Monocular 3D Object Detection [[ICCV2021](https://arxiv.org/pdf/2107.13774.pdf)][[Pytorch](https://github.com/SuperMHP/GUPNet)]
- <a id="neighbor-voting"></a>**[Neighbor-Vote]** Neighbor-Vote: Improving Monocular 3D Object Detection through Neighbor Distance Voting [[ACMMM2021](https://arxiv.org/pdf/2107.02493.pdf)][[Pytorch](https://github.com/cxmomo/Neighbor-Vote)]
- <a id="MonoEF"></a>**[MonoEF]** Monocular 3D Object Detection: An Extrinsic Parameter Free Approach [[CVPR2021](https://arxiv.org/abs/2106.15796?context=cs)][[Pytorch](https://github.com/ZhouYunsong-SJTU/MonoEF)]
- <a id="monodle"></a>**[monodle]** Delving into Localization Errors for Monocular 3D Object Detection [[CVPR2021](https://arxiv.org/abs/2103.16237)][[Pytorch](https://github.com/xinzhuma/monodle)]
- <a id="Monoflex"></a>**[Monoflex]** Objects are Different: Flexible Monocular 3D Object Detection [[CVPR2021](https://arxiv.org/abs/2104.02323)][[Pytorch](https://github.com/zhangyp15/MonoFlex)]
- <a id="GrooMeD-NMS"></a>**[GrooMeD-NMS]** GrooMeD-NMS: Grouped Mathematically Differentiable NMS for Monocular 3D Object Detection [[CVPR2021](https://arxiv.org/abs/2103.17202)][[Pytorch](https://github.com/abhi1kumar/groomed_nms)]
- <a id="DDMP-3D"></a>**[DDMP-3D]** Depth-conditioned Dynamic Message Propagation for Monocular 3D Object Detection [[CVPR2021](https://arxiv.org/abs/2103.16470)][[Pytorch](https://github.com/Willy0919/DDMP-3D)]
- <a id="MonoRUn"></a>**[MonoRUn]** MonoRUn: Monocular 3D Object Detection by Reconstruction and Uncertainty Propagation [[CVPR2021](https://arxiv.org/abs/2103.12605)][[Pytorch](https://github.com/tjiiv-cprg/MonoRUn)]
- <a id="M3DSSD"></a>**[M3DSSD]** M3DSSD: Monocular 3D Single Stage Object Detector [[CVPR2021](https://arxiv.org/abs/2103.13164)][[Pytorch](https://github.com/mumianyuxin/M3DSSD)]
- <a id="CaDDN"></a>**[CaDDN]** Categorical Depth Distribution Network for Monocular 3D Object Detection [[CVPR2021](https://arxiv.org/abs/2103.01100)][[Pytorch](https://github.com/TRAILab/CaDDN)]
- <a id="visualDet3D"></a>**[visualDet3D]** Ground-aware Monocular 3D Object Detection for Autonomous Driving [[RA-L](https://arxiv.org/abs/2102.00690)][[Pytorch](https://github.com/Owen-Liuyuxuan/visualDet3D)]
 
## 2020
- <a name="UR3D"></a>**[UR3D]** Distance-Normalized Unified Representation for Monocular 3D Object Detection [[ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740086.pdf)]
- <a name="MonoDR"></a>**[MonoDR]** Monocular Differentiable Rendering for Self-Supervised 3D Object Detection [[ECCV2020](https://arxiv.org/abs/2009.14524)]
- <a id="DA-3Ddet"></a>**[DA-3Ddet]** Monocular 3d object detection via feature domain adaptation [[ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540018.pdf)]
- <a id="MoVi-3D"></a>**[MoVi-3D]** Towards generalization across depth for monocular 3d object detection [[ECCV2020](https://arxiv.org/abs/1912.08035)]
- <a id="PatchNet"></a>**[PatchNet]** Rethinking Pseudo-LiDAR Representation [[ECCV2020](https://arxiv.org/abs/2008.04582)][[Pytorch](https://github.com/xinzhuma/patchnet)]
- <a id="RAR-Net"></a>**[RAR-Net]** Reinforced Axial Refinement Network for Monocular 3D Object Detection [[ECCV2020](https://arxiv.org/abs/2008.13748)]
- <a id='kinematic3d'></a>**[kinematic3d]** Kinematic 3D Object Detection in Monocular Video [[ECCV2020](https://arxiv.org/abs/2007.09548)][[Pytorch](https://github.com/garrickbrazil/kinematic3d)]
- <a id="RTM3D"></a>**[RTM3D]** RTM3D: Real-time Monocular 3D Detection from Object Keypoints for Autonomous Driving [[ECCV2020](https://arxiv.org/abs/2001.03343)][[Pytorch](https://github.com/Banconxuan/RTM3D)]
- <a id="SMOKE"></a>**[SMOKE]** SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation [[CVPRW2020](https://arxiv.org/pdf/2002.10111.pdf)][[Pytorch](https://github.com/lzccccc/SMOKE)]
- <a id="D4LCN"></a>**[D4LCN]** Learning Depth-Guided Convolutions for Monocular 3D Object Detection [[CVPRW2020](https://arxiv.org/abs/1912.04799)][[Pytorch](https://github.com/dingmyu/D4LCN)]
- <a id="MonoPair"></a>**[MonoPair]** MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships [[CVPR2020](https://arxiv.org/abs/2003.00504)]
- <a id="pseudo-LiDAR_e2e"></a>**[pseudo-LiDAR_e2e]** End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection [[CVPR2020](https://arxiv.org/abs/2004.03080)][[Pytorch](https://github.com/mileyan/pseudo-LiDAR_e2e)]
- <a id="Pseudo-LiDAR++"></a>**[Pseudo-LiDAR++]** Pseudo-LiDAR++: Accurate Depth for 3D Object Detection in Autonomous Driving [[ICLR2020](https://arxiv.org/abs/1906.06310)][[Pytorch](https://github.com/mileyan/Pseudo_Lidar_V2)]
- <a id="OACV"></a>**[OACV]** Object-Aware Centroid Voting for Monocular 3D Object Detection [[IROS2020](https://arxiv.org/abs/2007.09836)]
- <a id="MonoGRNet_v2"></a>**[MonoGRNet_v2]** Monocular 3D Object Detection via Geometric Reasoning on Keypoints [[VISIGRAPP2020](https://arxiv.org/abs/1905.05618)]
- <a id="ForeSeE"></a>**[ForeSeE]** Task-Aware Monocular Depth Estimation for 3D Object Detection [[AAAI2020(oral)](https://arxiv.org/abs/1909.07701)][[Pytorch](https://github.com/WXinlong/ForeSeE)]
- <a id="Decoupled-3D"></a>**[Decoupled-3D]** Monocular 3D Object Detection with Decoupled Structured Polygon Estimation and Height-Guided Depth Estimation [[AAAI2020](https://arxiv.org/abs/2002.01619)]

## 2019
- <a id="3d-vehicle-tracking"></a>**[3d-vehicle-tracking]** Joint Monocular 3D Vehicle Detection and Tracking [[ICCV2019](https://arxiv.org/pdf/1811.10742.pdf)][[Pytorch](https://github.com/ucbdrive/3d-vehicle-tracking)]
- <a id="MonoDIS"></a>**[MonoDIS]** Disentangling monocular 3d object detection [[ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Simonelli_Disentangling_Monocular_3D_Object_Detection_ICCV_2019_paper.pdf)]
- <a id="AM3D"></a>**[AM3D]** Accurate Monocular Object Detection via Color-Embedded 3D Reconstruction for Autonomous Driving [[ICCV2019](https://arxiv.org/abs/1903.11444)]
- <a id="M3D-RPN"></a>**[M3D-RPN]** M3D-RPN: Monocular 3D Region Proposal Network for Object Detection [[ICCV2019(Oral)](https://arxiv.org/abs/1907.06038)][[Pytorch](https://github.com/garrickbrazil/M3D-RPN)]
- <a id="MVRA"></a>**[MVRA]** Multi-View Reprojection Architecture for Orientation Estimation [[ICCVW2019](https://openaccess.thecvf.com/content_ICCVW_2019/papers/ADW/Choi_Multi-View_Reprojection_Architecture_for_Orientation_Estimation_ICCVW_2019_paper.pdf)]
- <a id="Mono3DPLiDAR"></a>**[Mono3DPLiDAR]** Monocular 3D Object Detection with Pseudo-LiDAR Point Cloud [[ICCVW2019](https://arxiv.org/abs/1903.09847)]
- <a id="MonoPSR"></a>**[MonoPSR]** Monocular 3D Object Detection Leveraging Accurate Proposals and Shape Reconstruction [[CVPR2019](https://arxiv.org/abs/1904.01690)][[Pytorch](https://github.com/kujason/monopsr)]
- <a id="FQNet"></a>**[FQNet]** Deep fitting degree scoring network for monocular 3d object detection [[CVPR2019](https://arxiv.org/abs/1904.12681)]
- <a id="ROI-10D"></a>**[ROI-10D]** ROI-10D: Monocular Lifting of 2D Detection to 6D Pose and Metric Shape [[CVPR2019](https://arxiv.org/abs/1812.02781)]
- <a id="GS3D"></a>**[GS3D]** GS3D: An Efficient 3D Object Detection Framework for Autonomous Driving [[CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_GS3D_An_Efficient_3D_Object_Detection_Framework_for_Autonomous_Driving_CVPR_2019_paper.html)]
- <a id="Pseudo-LiDAR"></a>**[Pseudo-LiDAR]** Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving [[CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Pseudo-LiDAR_From_Visual_Depth_Estimation_Bridging_the_Gap_in_3D_CVPR_2019_paper.pdf)][[Pytorch](https://github.com/mileyan/pseudo_lidar)]
- <a id="BirdGAN"></a>**[BirdGAN]** Learning 2D to 3D Lifting for Object Detection in 3D for Autonomous Vehicles [[IROS2019](https://arxiv.org/pdf/1904.08494.pdf)]
- <a id="MonoGRNet"></a>**[MonoGRNet]** MonoGRNet: A Geometric Reasoning Network for Monocular 3D Object Localization [[AAAI2019(oral)](https://arxiv.org/abs/1811.10247)][[Tensorflow](https://github.com/Zengyi-Qin/MonoGRNet)]
- <a id="OFT-Net"></a>**[OFT-Net]** Orthographic feature transform for monocular 3d object detection [[BMVC2019](https://bmvc2019.org/wp-content/uploads/papers/0328-paper.pdf)][[Pytorch](https://github.com/tom-roddick/oft)]
- <a id="Shift R-CNN"></a>**[Shift R-CNN]** Shift R-CNN: Deep Monocular 3D Object Detection with Closed-Form Geometric Constraints [[TIP2019](https://arxiv.org/abs/1905.09970)]
- <a id="SS3D"></a>**[SS3D]** SS3D: Monocular 3d object detection and box fitting trained end-to-end using intersection-over-union loss [[Arxiv2019](https://arxiv.org/abs/1906.08070)]

## 2018
- <a id="Multi-Fusion"></a>**[Multi-Fusion]** Multi-Level Fusion based 3D Object Detection from Monocular Images [[CVPR2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Multi-Level_Fusion_Based_CVPR_2018_paper.pdf)][[Pytorch](https://github.com/abbyxxn/maskrcnn-benchmark-3d)]
- <a id="Mono3D++"></a>**[Mono3D++]** Mono3D++: Monocular 3D Vehicle Detection with Two-Scale 3D Hypotheses and Task Priors [[AAAI2018](https://arxiv.org/abs/1901.03446)]

## 2017
- <a id="Deep3DBox"></a>**[Deep3DBox]** 3D Bounding Box Estimation Using Deep Learning and Geometry [[CVPR2017](https://arxiv.org/abs/1612.00496)][[Pytorch](https://github.com/skhadem/3D-BoundingBox)][[Tensorflow](https://github.com/smallcorgi/3D-Deepbox)]
- <a id="Deep MANTA"></a>**[Deep MANTA]** Deep MANTA: A Coarse-to-fine Many-Task Network for joint 2D and 3D vehicle analysis from monocular image [[CVPR2017](https://arxiv.org/abs/1703.07570)]

## 2016
- <a id="Mono3D"></a>**[Mono3D]** Monocular 3D object detection for autonomous driving [[CVPR2016](https://www.cs.toronto.edu/~urtasun/publications/chen_etal_cvpr16.pdf)]



# KITTI Results
<!-- <font color=blue, size=4>val/test</font><font color=blue, size=3> (R<sub>11</sub>/R<sub>40</sub>) @ IOU=0.7</font> -->
<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:5px 10px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-nrix{border-color:inherit;text-align:center;vertical-align:middle}
</style> -->
<table class="tg" style="text-align:center">
<thead>
  <tr>
    <th class="tg-9wq8" rowspan="2">Method</th>
    <th class="tg-nrix" rowspan="2">Extra</th>
    <th class="tg-nrix" colspan="3">Test,      
    AP<sub>3D</sub>|<sub>R<sub>40</sub></th>
    <th class="tg-nrix" colspan="3">Val,      
    AP<sub>3D</sub>|<sub>R<sub>40</sub></th>
<!--       <th class="tg-nrix" colspan="3">Val, 
    AP<sub>3D</sub>|<sub>R<sub>11</sub></th> -->
    <th class="tg-nrix" rowspan="2">Reference</th>
  </tr>
  <tr>
    <th class="tg-nrix">Easy</th>
    <th class="tg-nrix">Mod.</th>
    <th class="tg-nrix">Hard</th>
    <th class="tg-nrix">Easy</th>
    <th class="tg-nrix">Mod.</th>
    <th class="tg-nrix">Hard</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8"><a href="#LPCG">LPCG</a></td>     
    <td class="tg-nrix">Lidar+raw</td>  
    <td class="tg-nrix">25.56</td>
    <td class="tg-nrix">17.80</td>
    <td class="tg-nrix">15.38</td>
    <td class="tg-nrix">31.15</td>  
    <td class="tg-nrix">23.42</td>
    <td class="tg-nrix">20.60</td>
    <td class="tg-nrix">ECCV2022</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="#CMKD">CMKD</a></td>      
    <td class="tg-nrix">Lidar+raw</td>  
    <td class="tg-nrix">28.55</td>
    <td class="tg-nrix">18.69</td>
    <td class="tg-nrix">16.77</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">ECCV2022</td>
  </tr>
  <tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
  <tr>
    <td class="tg-9wq8"><a href="#MonoPSR">MonoPSR</a></td>     
    <td class="tg-nrix">Lidar</td>     
    <td class="tg-nrix">10.76</td>
    <td class="tg-nrix">7.25</td>
    <td class="tg-nrix">5.85</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">CVPR2019</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="#MonoRUn">MonoRUn</a></td>     
    <td class="tg-nrix">Lidar</td>     
    <td class="tg-nrix">19.65</td>
    <td class="tg-nrix">12.30</td>
    <td class="tg-nrix">10.58</td>
    <td class="tg-nrix">20.02</td>
    <td class="tg-nrix">14.65</td>
    <td class="tg-nrix">12.61</td>
    <td class="tg-nrix">CVPR2021</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="#CaDDN">CaDDN</a></td>     
    <td class="tg-nrix">Lidar</td>     
    <td class="tg-nrix">19.17</td>
    <td class="tg-nrix">13.41</td>
    <td class="tg-nrix">11.46</td>
    <td class="tg-nrix">23.57</td>
    <td class="tg-nrix">16.31</td>
    <td class="tg-nrix">13.84</td>
    <td class="tg-nrix">CVPR2021</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="#MonoDistill">MonoDistill</a></td>     
    <td class="tg-nrix">Lidar</td>     
    <td class="tg-nrix">22.97</td>
    <td class="tg-nrix">16.03</td>
    <td class="tg-nrix">13.60</td>
    <td class="tg-nrix">24.31</td>
    <td class="tg-nrix">18.47</td>
    <td class="tg-nrix">15.76</td>
    <td class="tg-nrix">ICLR2022</td>
  </tr>
  <tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
  <tr>
    <td class="tg-9wq8"><a href="#AM3D">AM3D</a></td>
    <td class="tg-nrix">Depth</td>     
    <td class="tg-nrix">16.50</td>
    <td class="tg-nrix">10.74</td>
    <td class="tg-nrix">9.52</td>
    <td class="tg-nrix">28.31</td>
    <td class="tg-nrix">15.76</td>
    <td class="tg-nrix">12.24</td>
    <td class="tg-nrix">ICCV2019</td>
    </tr>
  <tr>
    <td class="tg-9wq8"><a href="#PatchNet">PatchNet</a></td>     
    <td class="tg-nrix">Depth</td>     
    <td class="tg-nrix">15.68</td>
    <td class="tg-nrix">11.12</td>
    <td class="tg-nrix">10.17</td>
    <td class="tg-nrix">31.60</td>
    <td class="tg-nrix">16.80</td>
    <td class="tg-nrix">13.80</td>
    <td class="tg-nrix">ECCV2020</td>
  </tr>
    <tr>
    <td class="tg-9wq8"><a href="#D4LCN">D4LCN</a></td>     
    <td class="tg-nrix">Depth</td>     
    <td class="tg-nrix">16.65</td>
    <td class="tg-nrix">11.72</td>
    <td class="tg-nrix">9.51</td>
    <td class="tg-nrix">22.32</td>
    <td class="tg-nrix">16.20</td>
    <td class="tg-nrix">12.30</td>
    <td class="tg-nrix">CVPRW2020</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="#DFR-Net">DFR-Net</a></td>     
    <td class="tg-nrix">Depth</td>     
    <td class="tg-nrix">19.40</td>
    <td class="tg-nrix">13.63</td>
    <td class="tg-nrix">10.35</td>
    <td class="tg-nrix">24.81</td>
    <td class="tg-nrix">17.78</td>
    <td class="tg-nrix">14.41</td>
    <td class="tg-nrix">ICCV2021</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="#Pseudo-Stereo">Pseudo-Stereo</a></td>     
    <td class="tg-nrix">Depth</td>     
    <td class="tg-nrix">23.74</td>
    <td class="tg-nrix">17.74</td>
    <td class="tg-nrix">15.14</td>
    <td class="tg-nrix">35.18</td>
    <td class="tg-nrix">24.15</td>
    <td class="tg-nrix">20.35</td>
    <td class="tg-nrix">CVPR2022</td>
  </tr>
  <tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
  <tr>
    <td class="tg-9wq8" nowrap="nowrap"><a href="#M3D-RPN">M3D-RPN</a></td>
    <td class="tg-nrix">None</td>
    <td class="tg-nrix">14.76</td>
    <td class="tg-nrix">9.71</td>
    <td class="tg-nrix">7.42</td>
    <td class="tg-nrix">14.53</td>
    <td class="tg-nrix">11.07</td>
    <td class="tg-nrix">8.65</td>
    <td class="tg-nrix">ICCV2019</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="#SMOKE">SMOKE</a></td>
    <td class="tg-nrix">None</td>
    <td class="tg-nrix">14.03</td>
    <td class="tg-nrix">9.76</td>
    <td class="tg-nrix">7.84 </td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">CVPRW2020</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="#MonoPair">MonoPair</a></td>
    <td class="tg-nrix">None</td>
    <td class="tg-nrix">13.04</td>
    <td class="tg-nrix">9.99</td>
    <td class="tg-nrix">8.65 </td>
    <td class="tg-nrix">16.28</td>
    <td class="tg-nrix">12.30</td>
    <td class="tg-nrix">10.42</td>
    <td class="tg-nrix">CVPR2020</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="#RTM3D">RTM3D</a></td>
    <td class="tg-nrix">None</td>
    <td class="tg-nrix">14.41</td>
    <td class="tg-nrix">10.34</td>
    <td class="tg-nrix">8.77 </td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">ECCV2020</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="#M3DSSD">M3DSSD</a></td>
    <td class="tg-nrix">None</td>
    <td class="tg-nrix">17.51</td>
    <td class="tg-nrix">11.46</td>
    <td class="tg-nrix">8.98 </td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">CVPR2021</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="#Monoflex">Monoflex</a></td>
    <td class="tg-nrix">None</td>
    <td class="tg-nrix">19.94</td>
    <td class="tg-nrix">13.89</td>
    <td class="tg-nrix">12.07 </td>
    <td class="tg-nrix">23.64</td>
    <td class="tg-nrix">17.51</td>
    <td class="tg-nrix">14.83</td>
    <td class="tg-nrix">CVPR2021</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="#GUPNet">GUPNet</a></td>     
    <td class="tg-nrix">None</td>     
    <td class="tg-nrix">20.11</td>
    <td class="tg-nrix">14.20</td>
    <td class="tg-nrix">11.77</td>
    <td class="tg-nrix">22.76</td>
    <td class="tg-nrix">16.46</td>
    <td class="tg-nrix">13.72</td>
    <td class="tg-nrix">ICCV2021</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="#MonoCon">MonoCon</a></td>     
    <td class="tg-nrix">None</td>     
    <td class="tg-nrix">22.50</td>
    <td class="tg-nrix">16.46</td>
    <td class="tg-nrix">13.95</td>
    <td class="tg-nrix">26.33</td>
    <td class="tg-nrix">19.01</td>
    <td class="tg-nrix">15.98</td>
    <td class="tg-nrix">AAAI2022</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="#MonoDDE">MonoDDE</a></td>     
    <td class="tg-nrix">None</td>     
    <td class="tg-nrix">24.93</td>
    <td class="tg-nrix">17.14</td>
    <td class="tg-nrix">15.10</td>  
    <td class="tg-nrix">26.66</td>
    <td class="tg-nrix">19.75</td>
    <td class="tg-nrix">16.72</td> 
    <td class="tg-nrix">CVPR2022</td>
  </tr>
      
</tbody>
</table>

