import yaml
import numpy as np
import os
import cv2

import gdown
import torch
import torchvision.transforms as transforms

from deviant.helper import download_file, unzip_file
from deviant.lib.models.gupnet import GUPNet
from deviant.test.test_ses_basis_orthogonality import convert_to_tensor
from deviant.lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections
from deviant.lib.datasets.kitti_utils import Calibration
from deviant.lib.helpers.util import project_3d, draw_3d_box, draw_bev
from torch import nn

CACHE_PATH = os.path.expanduser("~/.deviant")
# CONFIG_PATH = os.path.join(CACHE_PATH, 'config_run_201_a100_v0_1.yaml')
CONFIG_PATH = os.path.join(CACHE_PATH, 'run_250.yaml')
# WEIGHTS_ZIP_FOLDER_NAME = 'config_run_201_a100_v0_1'
WEIGHTS_ZIP_FOLDER_NAME = 'run_250'
WEIGHTS_ZIP_FILE_NAME = WEIGHTS_ZIP_FOLDER_NAME + '.zip'
WEIGHTS_ZIP_PATH = os.path.join(CACHE_PATH, WEIGHTS_ZIP_FILE_NAME)
WEIGHTS_FOLDER_PATH = os.path.join(CACHE_PATH, WEIGHTS_ZIP_FOLDER_NAME)

CHECKPOINT_PATH = os.path.join(WEIGHTS_FOLDER_PATH, 'run_250/checkpoints/checkpoint_epoch_140.pth')

# CONFIG_DOWNLOAD = 'https://raw.githubusercontent.com/Cardinal-Robo-Taxi/DEVIANT/main/code/experiments/config_run_201_a100_v0_1.yaml'
# WEIGHTS_DOWNLOAD = 'https://drive.google.com/u/0/uc?id=17qezmIjckRSAva1fNnYBmgR9LaY-dPnp&export=download'

CONFIG_DOWNLOAD = "https://raw.githubusercontent.com/Cardinal-Robo-Taxi/DEVIANT/main/deviant/experiments/run_250.yaml"
WEIGHTS_DOWNLOAD = "https://drive.google.com/u/0/uc?id=1_79GfHcpAQR3wdvhj9GDHc7_c_ndf1Al&export=download"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomCalibration(Calibration):
    def __init__(self, calib):
        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4
        self.C2V = self.inverse_rigid_trans(self.V2C)

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

def plot_boxes_on_image_and_in_bev(predictions_img, img, canvas_bev, plot_color, p2, box_class_list= ["car", "cyclist", "pedestrian"], use_classwise_color= False, show_3d= True, show_bev= True, thickness= 4, bev_scale = 10.0):
    # https://sashamaps.net/docs/resources/20-colors/
    # Some taken from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/color_map.py#L10
    class_color_map = {'car': (255,51,153),
                       'cyclist': (255, 130, 48),  # Orange
                       'bicycle': (255, 130, 48),  # Orange
                       'pedestrian': (138, 43, 226),  # Violet
                       'bus': (0, 0, 0), # Black
                       'construction_vehicle': (0, 130, 200), # Blue
                       'motorcycle': (220, 190, 255),  # Lavender
                       'trailer': (170, 255, 195), # Mint
                       'truck': (128, 128, 99),  # Olive
                       'traffic_cone': (255, 225, 25), # Yellow
                       'barrier': (128, 128, 128),  # Grey
                       }

    if predictions_img is not None and predictions_img.size > 0:
        # Add dimension if there is a single point
        if predictions_img.ndim == 1:
            predictions_img = predictions_img[np.newaxis, :]

        N = predictions_img.shape[0]
        #   0   _    _    1    2    3   4  5  6    7    8    9    10   11   12   13    14      15
        # (cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score/num_lidar_points  )
        # Add projected 3d center information to predictions_img
        class_name = ['Pedestrian', 'Car', 'Cyclist']
        # cls = list(map(lambda class_index: class_name[np.rint(class_index)], predictions_img[:, 0]))
        cls = predictions_img[:, 0]
        x1  = predictions_img[:, 2]
        y1  = predictions_img[:, 3]
        h3d = predictions_img[:, 6]
        w3d = predictions_img[:, 7]
        l3d = predictions_img[:, 8]
        x3d = predictions_img[:, 9]
        y3d = predictions_img[:, 10] - h3d/2
        z3d = predictions_img[:, 11]
        ry3d = predictions_img[:,12]

        if predictions_img.shape[1] > 13:
            score = predictions_img[:, 13]
        else:
            score = np.ones((predictions_img.shape[0],))

        for j in range(N):
            box_class = class_name[int(cls[j])].lower()
            if box_class == "dontcare":
                continue
            if box_class in box_class_list:
                if use_classwise_color:
                    box_plot_color = class_color_map[box_class]
                else:
                    box_plot_color = plot_color

                # if box_class == "car" and score[j] < 100:
                #     continue
                # if box_class != "car" and score[j] < 50:
                #     continue

                box_plot_color = box_plot_color[::-1]
                if show_3d:
                    verts_cur, corners_3d_cur = project_3d(p2, x3d[j], y3d[j], z3d[j], w3d[j], h3d[j], l3d[j], ry3d[j], return_3d=True)
                    draw_3d_box(img, verts_cur, color= box_plot_color, thickness= thickness)
                    # cv2.putText(img, str(int(score[j])), (int(x1[j]), int(y1[j])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))
                if show_bev:
                    draw_bev(canvas_bev, z3d[j], l3d[j], w3d[j], x3d[j], ry3d[j], color= box_plot_color, scale= bev_scale, thickness= thickness, text= None)#str(int(score[j])))

def get_default_cfg():
    return {
        'random_seed': 111,
        'dataset': {
            'type': 'kitti',
            'root_dir': 'data/',
            'batch_size': 12,
            'resolution': [1280, 384],
            'train_split_name': 'trainval',
            'val_split_name': 'test',
            'class_merging': False,
            'use_dontcare': False,
            'use_3d_center': True,
            'writelist': ['Car', 'Pedestrian', 'Cyclist'],
            'random_flip': 0.5,
            'random_crop': 0.5,
            'scale': 0.4,
            'shift': 0.1
        },
        'model': {
            'type': 'gupnet',
            'backbone': 'dla34',
            'neck': 'DLAUp',
            'use_conv': 'sesn',
            'replace_style': 'max_scale_after_dla34_layer',
            'sesn_norm_per_scale': False,
            'sesn_rescale_basis': False,
            'sesn_scales': [0.83, 0.9, 1.0],
            'scale_index_for_init': 0
        },
        'optimizer': {
            'type': 'adam',
            'lr': 0.00125,
            'weight_decay': 1e-05
        },
        'lr_scheduler': {
            'warmup': True,
            'decay_rate': 0.1,
            'decay_list': [90, 120]
        },
        'trainer': {
            'max_epoch': 140,
            'eval_frequency': 140,
            'save_frequency': 20,
            'disp_frequency': 20,
            'log_dir': 'output/'
        },
        'tester': {'threshold': 0.05}
    }

class Deviant(nn.Module):

    def __init__(self,
            cfg = get_default_cfg(),
            P2 = np.array([
                [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01], 
                [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01], 
                [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03],
            ]),
            mean_size = np.array([[1.76255119    ,0.66068622   , 0.84422524   ],
                [1.52563191462 ,1.62856739989, 3.88311640418],
                [1.73698127    ,0.59706367   , 1.76282397   ]]),
            device=device
        ):
        super().__init__()
        
        os.makedirs(CACHE_PATH, exist_ok=True)

        if not os.path.exists(CONFIG_PATH):
            # Download config
            print("Downloading configs")
            download_file(CONFIG_DOWNLOAD, CONFIG_PATH)

        if not os.path.exists(WEIGHTS_ZIP_PATH):
            # Download config
            print("Downloading weights")
            # download_file(WEIGHTS_DOWNLOAD, WEIGHTS_ZIP_PATH)
            gdown.download(WEIGHTS_DOWNLOAD, WEIGHTS_ZIP_PATH, quiet=False)


        if not os.path.exists(WEIGHTS_FOLDER_PATH):
            # Download config
            print("Unzipping weights")
            unzip_file(WEIGHTS_ZIP_PATH, CACHE_PATH, WEIGHTS_ZIP_FOLDER_NAME)

        self.cfg = cfg
        self.mean_size = mean_size
        self.model = GUPNet(
            backbone=self.cfg['model']['backbone'],
            neck=self.cfg['model']['neck'], 
            mean_size=self.mean_size, 
            cfg=self.cfg
        )

        self.device = device
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])

        self.model = self.model.to(device=self.device)

        self.scale = self.cfg['dataset']['scale']
        self.shift = self.cfg['dataset']['shift']
        self.img_size = np.array(list(self.cfg['dataset']['resolution']) + [3,])

        self.center = np.array(self.img_size) / 2
        self.crop_size = self.img_size
        self.coord_range = np.array([self.center-self.crop_size/2,self.center+self.crop_size/2]).astype(np.float32)
        self.coord_range = torch.tensor(self.coord_range)
        self.calib_P2_np = P2 # 3 x 4
        
        self.calib = CustomCalibration({
            'P2': self.calib_P2_np, # 3x4
            'R0': np.eye(3,3), # 3x3
            'Tr_velo2cam': np.eye(3,4), # 3x4
        })

        self.calib_P2 = torch.tensor(self.calib_P2_np)
        
        self.transform = transforms.ToTensor()

        self.color_gt     = (153, 255, 51) # (0, 255 , 0)
        self.box_class_list = ["car", "cyclist", "pedestrian"]
        self.use_classwise_color = True

        self.resolution = np.array(self.cfg['dataset']['resolution'])
        self.downsample = 4
        self.features_size = self.resolution // self.downsample
        self.cls_mean_size = np.array([
            [1.76255119    , 0.66068622   , 0.84422524    ],
            [1.52563191462 , 1.62856739989, 3.88311640418 ],
            [1.73698127    , 0.59706367   , 1.76282397    ]
        ])

        if len(self.coord_range.shape)==2:
            self.coord_range = self.coord_range.unsqueeze(0)
        if len(self.calib_P2.shape)==2:
            self.calib_P2 = self.calib_P2.unsqueeze(0)
        
        self.coord_range = self.coord_range.to(dtype=torch.float32, device=self.device)
        self.calib_P2 = self.calib_P2.to(dtype=torch.float32, device=self.device)

    
    def forward(self, frame):
        frame = cv2.resize(frame, self.img_size[:2])
        img_tensor = self.transform(frame)

        if len(img_tensor.shape)==3:
            img_tensor = img_tensor.unsqueeze(0)
        

        img_tensor = img_tensor.to(dtype=torch.float32, device=self.device)
        
        outputs = self.model(img_tensor, self.coord_range, self.calib_P2, K=50, mode='test')

        dets = extract_dets_from_outputs(outputs=outputs, K=50)
        dets = dets.detach().cpu().numpy()
        
        info = {
            'img_id': list(range(dets.shape[0])),
            'bbox_downsample_ratio': np.array([self.img_size[:2] / self.features_size,]),
        }
        calibs = [self.calib  for index in info['img_id']]
        dets = decode_detections(dets = dets,
                                    info = info,
                                    calibs = calibs,
                                    cls_mean_size=self.cls_mean_size,
                                    threshold = self.cfg['tester']['threshold'])

        
        preds = np.array(dets[0])
        canvas_bev = np.zeros((640,480,3), dtype=np.uint8)

        return dets[0]


#================================================================
# Main starts here
#================================================================

def main():

    dev = Deviant()

    vid_url = "/home/aditya/Videos/Philadelphia.mp4"
    # vid_url = "/home/aditya/Datasets/hadar_car/2023-02-08_15:42:33.822505/rgb_2.mp4"

    cap = cv2.VideoCapture(vid_url)
    ret, frame = cap.read()


    while ret:
        
        det = dev(frame)

        preds = np.array(det)
        canvas_bev = np.zeros((640,480,3), dtype=np.uint8)

        frame_st = cv2.resize(frame, dev.img_size[:2])
        
        
        plot_boxes_on_image_and_in_bev(preds, canvas_bev=canvas_bev, img=frame_st, p2=dev.calib_P2_np, plot_color= dev.color_gt, box_class_list= dev.box_class_list, use_classwise_color= dev.use_classwise_color, show_3d= True)
        
        cv2.imshow('input', frame_st)
        cv2.imshow('bev', canvas_bev)

        if cv2.waitKey(1) == ord('q'):
            break
        for i in range(10):
            ret, frame = cap.read()

        

if __name__ == '__main__':
    main()