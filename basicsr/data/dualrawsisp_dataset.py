import os
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from basicsr.data.transforms import augment, paired_random_crop, triple_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from numpy import uint16
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import colour_demosaicing

from .data_util import paired_paths_from_folder, paired_paths_from_lmdb, \
    paired_paths_from_meta_info_file, triple_paths_from_folder


def demosaic(bayer_image, pattern='RGGB'):
    raw_demosaic = colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(bayer_image, pattern=pattern)
    raw_demosaic = np.ascontiguousarray(raw_demosaic.astype(np.float32))
    return raw_demosaic  # 输出为3通道的HxW格式


# @DATASET_REGISTRY.register()
# class DUALRAWSISP(data.Dataset):  # 暂停bit0的使用
#     def __init__(self, opt) -> None:
#         super(DUALRAWSISP, self).__init__()

#         self.opt = opt
#         self.bit = self.opt['bit']

#         # file client (io backend)
#         self.file_client = None
#         self.io_backend_opt = opt['io_backend']
#         self.mean = opt['mean'] if 'mean' in opt else None
#         self.std = opt['std'] if 'std' in opt else None
#         self.input_size = opt['input_size'] if 'input_size' in opt else None

#         self.gt_folder, self.lq_folder, self.masko_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_masko']
#         if 'filename_tmpl' in opt:
#             self.filename_tmpl = opt['filename_tmpl']
#         else:
#             self.filename_tmpl = '{}'

#         self.paths = triple_paths_from_folder([self.lq_folder, self.gt_folder, self.masko_folder], ['lq', 'gt', 'masko'], self.filename_tmpl)

#     def __getitem__(self, index) -> dict:
#         scale = self.opt['scale']
#         if self.file_client is None:
#             self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

#         if self.bit == 0:
#             # Load gt and lq images. Made especially for isp's ZRR data sets.
#             # image range: [0, 1], float32.

#             # 读取 GT 图像并归一化
#             gt_path = self.paths[index]['gt_path']
#             img_gt = np.asarray(imageio.imread(gt_path))
#             img_gt = img_gt.astype(np.float32) / 255

#             # 获取 GT 图像的尺寸
#             height, width = img_gt.shape[:2]

#             # 读取 RAW 图像
#             # lq_path = self.paths[index]['lq_path']
#             # raw_img = np.fromfile(lq_path, dtype=np.uint16)
#             # raw_img = raw_img.reshape((height, width))  # 根据 GT 图像的尺寸重塑 RAW 图像

#             lq_path = self.paths[index]['lq_path']
#             raw_img = np.asarray(imageio.imread(lq_path))

#             img_lq = demosaic(raw_img)
#             # img_lq = np.expand_dims(raw_img, axis=2)
#             img_lq = img_lq.astype(np.float32) / 4095  # 12位图像的最大值为4095

#             masko_path = self.paths[index]['masko_path']
#             img_masko = np.asarray(imageio.imread(masko_path))
#             if len(img_masko.shape) == 2:
#                 img_masko = np.expand_dims(img_masko, axis=2)
#             img_masko = img_masko.astype(np.float32) / 255

#             # augmentation for training
#             if self.opt['phase'] == 'train':
#                 gt_size = self.opt['gt_size']
#                 # random crop
#                 if type(gt_size) == int:
#                     img_gt, img_lq, img_masko = triple_random_crop(img_gt, img_lq, img_masko, gt_size, scale, gt_path)
#                 elif gt_size is None:
#                     pass
#                 else:
#                     raise NotImplementedError(f'gt_size {gt_size} is not supported yet.')
#                 # flip, rotation
#                 img_gt, img_lq, img_masko = augment([img_gt, img_lq, img_masko], self.opt['use_hflip'], self.opt['use_rot'])

#             # color space transform
#             # if 'color' in self.opt and self.opt['color'] == 'y':
#             #     img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
#             #     img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

#             # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
#             # TODO: It is better to update the datasets, rather than force to crop
#             # if self.opt['phase'] != 'train':
#             #     img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

#             # BGR to RGB, HWC to CHW, numpy to tensor
#             img_gt, img_lq, img_masko = img2tensor([img_gt, img_lq, img_masko], bgr2rgb=False, float32=True)

#             # normalize
#             if self.mean is not None or self.std is not None:
#                 normalize(img_lq, self.mean, self.std, inplace=True)
#                 normalize(img_gt, self.mean, self.std, inplace=True)

#         else:
#             pass


#         return {'lq': img_lq, 'gt': img_gt, 'masko': img_masko, 'lq_path': lq_path, 'gt_path': gt_path, 'masko_path': masko_path}

#     def __len__(self) -> int:
#         return len(self.paths)

#     @staticmethod
#     def np2tensor(imgs: list) -> list:
#         def _np2tensor(img):
#             return torch.from_numpy(np.ascontiguousarray(img.transpose((2, 0, 1)))).float()

#         return [_np2tensor(img) for img in imgs]

#     @staticmethod
#     def extract_bayer_channels(raw):
#         # Extract Bayer channels using array slicing
#         ch_B = raw[1::2, 1::2]
#         ch_Gb = raw[0::2, 1::2]
#         ch_R = raw[0::2, 0::2]
#         ch_Gr = raw[1::2, 0::2]

#         # Combine channels into an RGB image
#         RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
#         return RAW_combined
    
#     @staticmethod
#     def bayer2rggb(raw):
#         h, w = raw.shape
#         raw = raw.reshape(h // 2, 2, w // 2, 2)
#         raw = raw.transpose([1, 3, 0, 2]).reshape([-1, h // 2, w // 2])
#         return raw

#     @staticmethod
#     def imfrombytes(content, flag='color', float32=False, dtype=np.uint8):
#         """Read an image from bytes.

#         Args:
#             dtype:
#             content (bytes): Image bytes got from files or other streams.
#             flag (str): Flags specifying the color type of a loaded image,
#                 candidates are `color`, `grayscale` and `unchanged`.
#             float32 (bool): Whether to change to float32., If True, will also norm
#                 to [0, 1]. Default: False.

#         Returns:
#             ndarray: Loaded image array.
#         """
#         img_np = np.frombuffer(content, dtype)
#         imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
#         img = cv2.imdecode(img_np, imread_flags[flag])
#         if float32:
#             img = img.astype(np.float32) / 255.
#         return img

@DATASET_REGISTRY.register()
class DUALRAWSISP(data.Dataset):  # 暂停bit0的使用
    def __init__(self, opt) -> None:
        super(DUALRAWSISP, self).__init__()

        self.opt = opt
        self.bit = self.opt['bit']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.input_size = opt['input_size'] if 'input_size' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            if 'meta_info_file' in self.opt and \
                    self.opt['meta_info_file'] != 'None' and \
                    self.opt['meta_info_file'] is not None:
                self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                    self.opt['meta_info_file'])
            else:
                self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and \
                self.opt['meta_info_file'] != 'None' and \
                self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)



    def __getitem__(self, index) -> dict:
        scale = self.opt['scale']
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        if self.bit == 0:
            # Load gt and lq images. Made especially for isp's ZRR data sets.
            # image range: [0, 1], float32.

            # 读取 GT 图像并归一化
            gt_path = self.paths[index]['gt_path']
            img_gt = np.asarray(imageio.imread(gt_path))
            img_gt = img_gt.astype(np.float32) / 255

            # 获取 GT 图像的尺寸
            height, width = img_gt.shape[:2]

            # 读取 RAW 图像
            # lq_path = self.paths[index]['lq_path']
            # raw_img = np.fromfile(lq_path, dtype=np.uint16)
            # raw_img = raw_img.reshape((height, width))  # 根据 GT 图像的尺寸重塑 RAW 图像
            lq_path = self.paths[index]['lq_path']
            raw_img = np.asarray(imageio.imread(lq_path))
            # 提取 Bayer 通道并归一化
            img_lq = demosaic(raw_img)
            # img_lq = np.expand_dims(raw_img, axis=2)
            img_lq = img_lq.astype(np.float32) / 4095  # 12位图像的最大值为4095



            # augmentation for training
            if self.opt['phase'] == 'train':
                gt_size = self.opt['gt_size']
                # random crop
                if type(gt_size) == int:
                    img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
                elif gt_size is None:
                    pass
                else:
                    raise NotImplementedError(f'gt_size {gt_size} is not supported yet.')
                # flip, rotation
                img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

            # color space transform
            # if 'color' in self.opt and self.opt['color'] == 'y':
            #     img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            #     img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

            # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
            # TODO: It is better to update the datasets, rather than force to crop
            # if self.opt['phase'] != 'train':
            #     img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

            # normalize
            if self.mean is not None or self.std is not None:
                normalize(img_lq, self.mean, self.std, inplace=True)
                normalize(img_gt, self.mean, self.std, inplace=True)

        else:
            pass


        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def np2tensor(imgs: list) -> list:
        def _np2tensor(img):
            return torch.from_numpy(np.ascontiguousarray(img.transpose((2, 0, 1)))).float()

        return [_np2tensor(img) for img in imgs]

    @staticmethod
    def extract_bayer_channels(raw):
        # Extract Bayer channels using array slicing
        ch_B = raw[1::2, 1::2]
        ch_Gb = raw[0::2, 1::2]
        ch_R = raw[0::2, 0::2]
        ch_Gr = raw[1::2, 0::2]

        # Combine channels into an RGB image
        RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
        return RAW_combined
    
    @staticmethod
    def bayer2rggb(raw):
        h, w = raw.shape
        raw = raw.reshape(h // 2, 2, w // 2, 2)
        raw = raw.transpose([1, 3, 0, 2]).reshape([-1, h // 2, w // 2])
        return raw

    @staticmethod
    def imfrombytes(content, flag='color', float32=False, dtype=np.uint8):
        """Read an image from bytes.

        Args:
            dtype:
            content (bytes): Image bytes got from files or other streams.
            flag (str): Flags specifying the color type of a loaded image,
                candidates are `color`, `grayscale` and `unchanged`.
            float32 (bool): Whether to change to float32., If True, will also norm
                to [0, 1]. Default: False.

        Returns:
            ndarray: Loaded image array.
        """
        img_np = np.frombuffer(content, dtype)
        imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
        img = cv2.imdecode(img_np, imread_flags[flag])
        if float32:
            img = img.astype(np.float32) / 255.
        return img