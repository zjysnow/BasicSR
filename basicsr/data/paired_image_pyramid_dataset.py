import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

import cv2

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.data.util import (paired_paths_from_folder,
                               paired_paths_from_lmdb,
                               paired_paths_from_meta_info_file)
from basicsr.utils import FileClient, imfrombytes, img2tensor

class PairedImagePyramidDataset(data.Dataset):
    def __init__(self, opt):
        super(PairedImagePyramidDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])

            # to create pyramid for img_gt
            img_re1 = cv2.resize(cv2.resize(img_gt, (gt_size//2, gt_size//2), interpolation=cv2.INTER_LINEAR), (gt_size, gt_size), interpolation=cv2.INTER_LINEAR)
            img_re2 = cv2.resize(cv2.resize(img_gt, (gt_size//4, gt_size//4), interpolation=cv2.INTER_LINEAR), (gt_size, gt_size), interpolation=cv2.INTER_LINEAR)
            img_re3 = cv2.resize(cv2.resize(img_gt, (gt_size//8, gt_size//8), interpolation=cv2.INTER_LINEAR), (gt_size, gt_size), interpolation=cv2.INTER_LINEAR)
            img_re4 = cv2.resize(cv2.resize(img_gt, (gt_size//16, gt_size//16), interpolation=cv2.INTER_LINEAR), (gt_size, gt_size), interpolation=cv2.INTER_LINEAR)

            # TODO: color space transform
            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt, img_lq, img_re1, img_re2, img_re3, img_re4 = img2tensor([img_gt, img_lq, img_re1, img_re2, img_re3, img_re4],
                                        bgr2rgb=True,
                                        float32=True)

            # normalize
            if self.mean is not None or self.std is not None:
                normalize(img_lq, self.mean, self.std, inplace=True, align_corners=True)
                normalize(img_gt, self.mean, self.std, inplace=True, align_corners=True)

                normalize(img_re1, self.mean, self.std, inplace=True, align_corners=True)
                normalize(img_re2, self.mean, self.std, inplace=True, align_corners=True)
                normalize(img_re3, self.mean, self.std, inplace=True, align_corners=True)
                normalize(img_re4, self.mean, self.std, inplace=True, align_corners=True)

            return {
                'lq': img_lq,
                'gt': torch.cat((img_gt, img_re1, img_re2, img_re3, img_re4), 0),
                'lq_path': lq_path,
                'gt_path': gt_path
            }
        elif self.opt['phase'] == 'val':
            h,w,c = img_lq.shape
            if h % 16 != 0 or w % 16 != 0:
                h = h // 16 * 16
                w = w // 16 * 16
                img_lq = cv2.resize(img_lq, (h, w), interpolation=cv2.INTER_LINEAR)
                img_gt = cv2.resize(img_gt, (2*h, 2*w), interpolation=cv2.INTER_LINEAR)

            # TODO: color space transform
            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=True,
                                        float32=True)

            # normalize
            if self.mean is not None or self.std is not None:
                normalize(img_lq, self.mean, self.std, inplace=True, align_corners=True)
                normalize(img_gt, self.mean, self.std, inplace=True, align_corners=True)

            return {
                'lq': img_lq,
                'gt': img_gt,
                'lq_path': lq_path,
                'gt_path': gt_path
            }

    def __len__(self):
        return len(self.paths)
