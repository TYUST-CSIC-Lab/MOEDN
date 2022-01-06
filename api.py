# -*- coding: utf-8 -*-

"""
    @date: 2019.07.19
    @author: samuel ko
    @func: same function as api.py in original PRNet Repo.
"""
import torch
import numpy as np
from result import ResFCN256
from torch.autograd import Variable
import os
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp

FLAG = {"start_epoch": 0,
        "target_epoch": 200,
        "device": "cuda",
        "mask_path": "G:\Code\python\PRNet_PyTorch-master/utils/uv_data/uv_weight_mask_gdh.png",
        "block_size": 36,
        "alpha": 48,
        "dist_prob": 0.492,
        "nr_steps": 5e3,
        "lr": 0.00005,
        "batch_size": 64,
        "save_interval": 5,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "images": "G:\Code\python\PRNet_PyTorch-master/results",
        "gauss_kernel": "original",
        "summary_path": "G:\Code\python\PRNet_PyTorch-master\prnet_runs",
        "summary_step": 0,
        "resume": 1}
class PRN:
    '''
        <Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network>

        This class serves as the wrapper of PRNet.
    '''

    def __init__(self, model_dir,is_dlib=0,prefix = '.',**kwargs):
        # resolution of input and output image size.
        self.resolution_inp = kwargs.get("resolution_inp") or 256
        self.resolution_op = kwargs.get("resolution_op") or 256
        self.channel = kwargs.get("channel") or 3
        self.size = kwargs.get("size") or 16
        #---- load detectors
        if is_dlib:
            import dlib
            detector_path = os.path.join(prefix, 'model/mmod_human_face_detector.dat')
            self.face_detector = dlib.cnn_face_detection_model_v1(
                    detector_path)
        self.uv_kpt_ind_path = kwargs.get("uv_kpt_path") or "utils/uv_data/uv_kpt_ind.txt"
        self.face_ind_path = kwargs.get("face_ind_path") or "utils/uv_data/face_ind.txt"
        self.triangles_path = kwargs.get("triangles_path") or "utils/uv_data/triangles.txt"

        # 1) load model.
        self.pos_predictor = ResFCN256(FLAG)
        state = torch.load(model_dir)

        self.pos_predictor.eval()  # inference stage only.
        #if torch.cuda.device_count() > 0:
        self.pos_predictor = torch.nn.DataParallel(self.pos_predictor)
        self.pos_predictor=self.pos_predictor.to("cuda")
        self.pos_predictor.load_state_dict(state['prnet'],False)

        # 2) load uv_file.
        self.uv_kpt_ind = np.loadtxt(self.uv_kpt_ind_path).astype(np.int32)  # 2 x 68 get kpt
        self.face_ind = np.loadtxt(self.face_ind_path).astype(np.int32)  # get valid vertices in the pos map
        self.triangles = np.loadtxt(self.triangles_path).astype(np.int32)  # ntri x 3

        self.uv_coords = self.generate_uv_coords()

    def net_forward(self, image):
        ''' The core of out method: regress the position map of a given image.
        Args:
            image: (3, 256, 256) array. value range: 0~1
        Returns:
            pos: the 3D position map. (3, 256, 256) array.
        '''
        # image = torch.tensor(image)
        # # #image = torch.randn(3, 256, 256, 1)
        # image = image.permute(2, 0, 1)
        return self.pos_predictor(image)

    def process(self, input, image_info=None):
        ''' process image with crop operation.
        Args:
            input: (h,w,3) array or str(image path). image value range:1~255.
            image_info(optional): the bounding box information of faces. if None, will use dlib to detect face.

        Returns:
            pos: the 3D position map. (256, 256, 3).
        '''
        if isinstance(input, str):
            try:
                image = imread(input)
            except IOError:
                print("error opening file: ", input)
                return None
        else:
            image = input

        if image.ndim < 3:
            image = np.tile(image[:, :, np.newaxis], [1, 1, 3])

        if image_info is not None:
            if np.max(image_info.shape) > 4:  # key points to get bounding box
                kpt = image_info
                if kpt.shape[0] > 3:
                    kpt = kpt.T
                left = np.min(kpt[0, :]);
                right = np.max(kpt[0, :]);
                top = np.min(kpt[1, :]);
                bottom = np.max(kpt[1, :])
            else:  # bounding box
                bbox = image_info
                left = bbox[0];
                right = bbox[1];
                top = bbox[2];
                bottom = bbox[3]
            old_size = (right - left + bottom - top) / 2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size * 1.6)
        else:
            detected_faces = self.dlib_detect(image)
            if len(detected_faces) == 0:
                print('warning: no detected face')
                return None

            d = detected_faces[
                0].rect  ## only use the first detected face (assume that each input image only contains one face)
            left = d.left();
            right = d.right();
            top = d.top();
            bottom = d.bottom()
            old_size = (right - left + bottom - top) / 2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.14])
            size = int(old_size * 1.58)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image / 255.
        cropped_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))

        # run our net
        # st = time()
        from torchvision import transforms
        transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(FLAG["normalize_mean"], FLAG["normalize_std"])
        ])
        cropped_image = transform_img(cropped_image)
        cropped_image_t = cropped_image.unsqueeze(0)
        cropped_image_t = cropped_image_t.type(torch.FloatTensor)
        cropped_pos = self.net_forward(cropped_image_t.cuda())
        #cropped_pos=cropped_pos.cpu().detach().numpy()
        out = cropped_pos.cpu().detach().numpy()
        cropped_pos = np.squeeze(out)
        cropped_pos = cropped_pos * 255
        cropped_pos = cropped_pos.transpose(1, 2, 0)
        # print 'net time:', time() - st

        # restore
        cropped_vertices =np.reshape(cropped_pos, [-1, 3]).T
        z = cropped_vertices[2, :].copy() / tform.params[0, 0]
        cropped_vertices[2, :] = 1
        vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices)
        vertices = np.vstack((vertices[:2, :], z))
        pos = np.reshape(vertices.T, [self.resolution_op, self.resolution_op, 3])
        # pos = torch.from_numpy(pos)
        # pos = torch.randn(3, 256, 256, 1)
        # pos = pos.permute(3,0, 1, 2)
        #pos= Variable(torch.unsqueeze(pos, dim=0).float(), requires_grad=False)
        return pos

    def generate_uv_coords(self):
        resolution = self.resolution_op
        uv_coords = np.meshgrid(range(resolution), range(resolution))
        uv_coords = np.transpose(np.array(uv_coords), [1, 2, 0])
        uv_coords = np.reshape(uv_coords, [resolution ** 2, -1])
        uv_coords = uv_coords[self.face_ind, :]
        uv_coords = np.hstack((uv_coords[:, :2], np.zeros([uv_coords.shape[0], 1])))
        return uv_coords

    def get_landmarks(self, pos):
        '''
        Notice: original tensorflow version shape is [256, 256, 3] (H, W, C)
                where our pytorch shape is [3, 256, 256] (C, H, W).

        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            kpt: 68 3D landmarks. shape = (68, 3).
        '''
        kpt = pos[self.uv_kpt_ind[1, :], self.uv_kpt_ind[0, :], :]
        return kpt

    def get_vertices(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (3, 256, 256).
        Returns:
            vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
        '''
        all_vertices = np.reshape(pos, [self.resolution_op ** 2, -1])
        vertices = all_vertices[self.face_ind, :]

        return vertices

    def get_colors(self, image, vertices):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        [h, w, _] = image.shape
        vertices[:, 0] = np.minimum(np.maximum(vertices[:, 0], 0), w - 1)  # x
        vertices[:, 1] = np.minimum(np.maximum(vertices[:, 1], 0), h - 1)  # y
        ind = np.round(vertices).astype(np.int32)
        colors = image[ind[:, 1], ind[:, 0], :]  # n x 3

        return colors
    def dlib_detect(self, image):
        return self.face_detector(image, 1)
