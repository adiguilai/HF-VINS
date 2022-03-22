import numpy as np
import cv2
import torch

from hloc import extractors, matchers
from hloc.utils.base_model import dynamic_load


def resize_image(image, size, interp):
    interp = getattr(cv2, 'INTER_'+interp[len('cv2_'):].upper())
    h, w = image.shape[:2]
    if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
        interp = cv2.INTER_LINEAR
    resized = cv2.resize(image, size, interpolation=interp)
    return resized


class SuperPointExtractor:
    def __init__(self, nms_radius=3, max_keypoints=1024):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # load superpoint
        Model = dynamic_load(extractors, 'superpoint')
        self.model = Model({'name': 'superpoint', 'nms_radius': nms_radius, 'max_keypoints': max_keypoints}).eval().to(self.device)

    @torch.no_grad()
    def extract_feature(self, image):
        """ Extract key points and local descriptors with superpoint
        Input
        image: numpy ndarray of grayscale image shaped H x W.
        Output
        spp_info: dict
        {
            'keypoints': array N x 2 dtype=float32
            'scores': array N dtype=float32
            'descriptors': array 256 x N dtype=float32
            'image_size': array 2 (W, H) of image dtype=int
        }
        """
        image = image.astype(np.float32)
        # resize
        size = image.shape[:2][::-1]
        # maxsize = 1024
        if max(size) > 1024:
            scale = 1024 / max(size)
            size_new = tuple(int(round(x*scale)) for x in size)
            image = resize_image(image, size_new, 'cv2_area')
        image = image[None]  # grayscale
        image = image / 255.
        data = {
            'name': "test",
            'image': torch.from_numpy(image.__array__()).float().unsqueeze(0).to(self.device),
            'original_size': np.array(size),
        }
        spp_info = self.model(data)
        spp_info = {k: v[0].cpu().numpy() for k, v in spp_info.items()}
        
        # reduce position of keypoints
        spp_info['image_size'] = original_size = data['original_size']
        size = np.array(data['image'].shape[-2:][::-1])
        scales = (original_size / size).astype(np.float32)
        spp_info['keypoints'] = (spp_info['keypoints'] + .5) * scales[None] - .5
        return spp_info


class NetVLADExtractor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # load superpoint
        Model = dynamic_load(extractors, 'netvlad')
        self.model = Model({'name': 'netvlad'}).eval().to(self.device)

    @torch.no_grad()
    def extract_feature(self, image):
        """ Extract key points and local descriptors with superpoint
        Input
        image: numpy ndarray of RGB image shaped H x W x C.
        Output
        global_desc: dict
        {
            'global_descriptor': array 4096 dtype=float32
        }
        """
        # image numpy.array(HxWxC)
        image = image.astype(np.float32)
        # resize
        size = image.shape[:2][::-1]
        # maxsize = 1024
        if max(size) > 1024:
            scale = 1024 / max(size)
            size_new = tuple(int(round(x*scale)) for x in size)
            image = resize_image(image, size_new, 'cv2_area')
        
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.
        data = {
            'name': "test",
            'image': torch.from_numpy(image.__array__()).float().unsqueeze(0).to(self.device),
            'original_size': np.array(size),
        }
        global_desc = self.model(data)
        global_desc = {k: v[0].cpu().numpy() for k, v in global_desc.items()}
        return global_desc


class SuperGlueMatcher:
    def __init__(self, weights='outdoor', sinkhorn_iterations=50):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # load superpoint
        Model = dynamic_load(matchers, 'superglue')
        self.model = Model({'name': 'superglue', 'weights': weights, 'sinkhorn_iterations': sinkhorn_iterations}).eval().to(self.device)

    @torch.no_grad()
    def match_keypoints(self, keypoint_info1, keypoint_info2):
        """ Extract key points and local descriptors with superpoint
        Input
        keypoint_info: dict result from SuperPoint
        {
            'keypoints': array N x 2 dtype=float32
            'scores': array N dtype=float32
            'descriptors': array 256 x N dtype=float32
            'image_size': array 2 (W, H) of image dtype=int
        }
        Output
        spp_info: dict
        {
            'matches0': array N1 -1 for invalid match,
            'matches1': array N2,
            'matching_scores0': array N1,
            'matching_scores1': array N2
        }
        """
        # load data
        data = {}
        for k, v in keypoint_info1.items():
            data[k+'0'] = torch.from_numpy(v.__array__()).float().to(self.device)[None]
        # some matchers might expect an image but only use its size
        data['image0'] = torch.empty((1,)+tuple(keypoint_info1['image_size'])[::-1])[None]
        for k, v in keypoint_info2.items():
            data[k+'1'] = torch.from_numpy(v.__array__()).float().to(self.device)[None]
        data['image1'] = torch.empty((1,)+tuple(keypoint_info2['image_size'])[::-1])[None]
        matches = self.model(data)
        matches = {k: v[0].cpu().numpy() for k, v in matches.items()}
        return matches
