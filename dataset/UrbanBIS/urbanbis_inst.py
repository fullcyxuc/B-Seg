import os, sys, glob, math, numpy as np, pandas as pd
import scipy.ndimage
import scipy.interpolate
import torch
from torch.utils.data import DataLoader

sys.path.append('../')

from util.config import cfg
from util.log import logger
from lib.pointgroup_ops.functions import pointgroup_ops

import torch.distributed as dist


class Dataset:
    def __init__(self, test=False):
        self.data_root = cfg.data_root
        self.dataset = cfg.dataset
        self.filename_suffix = cfg.filename_suffix

        self.batch_size = cfg.batch_size
        self.train_workers = cfg.train_workers
        self.val_workers = cfg.train_workers

        self.full_scale = cfg.full_scale
        self.scale = cfg.scale
        self.max_npoint = cfg.max_npoint
        self.mode = cfg.mode

        if test:
            self.test_split = cfg.split  # val or test
            self.test_workers = cfg.test_workers
            cfg.batch_size = 1

    def trainLoader(self):
        train_file_names = []
        for city in self.dataset:
            train_file_names += glob.glob(os.path.join(self.data_root, city, 'train', '*' + self.filename_suffix))
        train_file_names = sorted(train_file_names)

        if self.filename_suffix == '.txt':
            self.train_files = [pd.read_csv(i, delimiter=',', header=None).to_numpy().astype(np.float32) for i in train_file_names]
        elif self.filename_suffix == '_inst_nostuff.pth':
            self.train_files = [torch.load(i) for i in train_file_names]
        else:
            raise ValueError

        logger.info('Training samples: {}'.format(len(self.train_files)))

        train_set = list(range(len(self.train_files)))
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge,
                                            num_workers=self.train_workers,
                                            shuffle=True, sampler=None, drop_last=True, pin_memory=True)

    def valLoader(self):
        self.val_file_names = []
        for city in self.dataset:
            self.val_file_names += glob.glob(os.path.join(self.data_root, city, 'val', '*' + self.filename_suffix))
        self.val_file_names = sorted(self.val_file_names)

        if self.filename_suffix == '.txt':
            self.val_files = [pd.read_csv(i, delimiter=',', header=None).to_numpy().astype(np.float32) for i in self.val_file_names]
        elif self.filename_suffix == '_inst_nostuff.pth':
            self.val_files = [torch.load(i) for i in self.val_file_names]

        logger.info('Validation samples: {}'.format(len(self.val_files)))

        val_set = list(range(len(self.val_files)))
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.valMerge,
                                          num_workers=self.val_workers,
                                          shuffle=False, drop_last=True, pin_memory=True)

    def testLoader(self):
        self.test_file_names = []
        for city in self.dataset:
            self.test_file_names += glob.glob(os.path.join(self.data_root, city, cfg.split, '*' + self.filename_suffix))
        self.test_file_names = sorted(self.test_file_names)

        if self.filename_suffix == '.txt':
            self.test_files = [pd.read_csv(i, delimiter=',', header=None).to_numpy().astype(np.float32) for i in self.test_file_names]
        elif self.filename_suffix == '_inst_nostuff.pth':
            self.test_files = [torch.load(i) for i in self.test_file_names]
        else:
            raise ValueError

        logger.info('Testing samples ({}): {}'.format(self.test_split, len(self.test_files)))

        test_set = list(np.arange(len(self.test_files)))
        self.test_data_loader = DataLoader(test_set, batch_size=1, collate_fn=self.testMerge,
                                           num_workers=self.test_workers,
                                           shuffle=False, drop_last=False, pin_memory=True)

    # Elastic distortion
    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

        def g(x_):
            return np.hstack([i(x_)[:, None] for i in interp])

        return x + g(x) * mag

    def getInstanceInfo(self, xyz, instance_label):
        '''
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        '''
        instance_info = np.ones((xyz.shape[0], 9),
                                dtype=np.float32) * -100.0  # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []  # (nInst), int
        instance_center = []
        instance_center_num = 0
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)

            # instance_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            # center_i = np.argmin(np.abs(xyz_i - mean_xyz_i).sum(1))
            # instance_center.append(xyz_i[center_i].reshape(1, -1))
            center_num = max(10, min(100, len(xyz_i) // 1000))
            random_idx = np.random.choice(xyz_i.shape[0], center_num, replace=True)
            instance_center.append(xyz_i[random_idx])

            instance_info_i = instance_info[inst_idx_i]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = min_xyz_i
            instance_info_i[:, 6:9] = max_xyz_i
            instance_info[inst_idx_i] = instance_info_i

            # instance_pointnum
            instance_pointnum.append(inst_idx_i[0].size)
            instance_center_num += center_num

        instance_center = np.concatenate(instance_center, axis=0)
        return instance_num, {"instance_info": instance_info, "instance_pointnum": instance_pointnum,
                              "instance_center": instance_center, "instance_center_num": instance_center_num}

    def dataAugment(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0],
                              [0, 0, 1]])  # rotation
        return np.matmul(xyz, m)


    def crop(self, xyz):
        valid_idxs = (xyz.min(1) >= 0)
        if xyz.shape[0] > self.max_npoint:
            valid_idxs = np.random.choice(xyz.shape[0], self.max_npoint, replace=True)
        return xyz, valid_idxs

    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def trainMerge(self, id):
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int
        instance_center = []  # (total_nInst), int

        batch_offsets = [0]
        inst_batch_offsets = [0]
        inst_center_batch_offsets = [0]

        total_inst_num = 0
        for i, idx in enumerate(id):
            if self.filename_suffix == '.txt':
                data = self.train_files[idx]
                xyz_origin, rgb, label, instance_label = data[:, :3], data[:, 3:6], data[:, 6], data[:, 7]
            else:
                xyz_origin, rgb, label, instance_label, _ = self.train_files[idx]

            # jitter / flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, True, True, True)

            # scale
            xyz = xyz_middle * self.scale

            # # elastic
            # ### 3 here need to be changed to same as scale (STPLS3D)
            # xyz = self.elastic(xyz, 6 * self.scale // 3, 40 * self.scale / 3)
            # xyz = self.elastic(xyz, 20 * self.scale // 3, 160 * self.scale / 3)

            # offset
            xyz -= xyz.min(0)

            # crop
            xyz, valid_idxs = self.crop(xyz)

            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

            # get instance information
            inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32))
            inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]  # (nInst), list
            inst_center = inst_infos["instance_center"]
            inst_center_num = inst_infos["instance_center_num"]


            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

            # merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])
            inst_batch_offsets.append(inst_batch_offsets[-1] + len(inst_pointnum))
            inst_center_batch_offsets.append(inst_center_batch_offsets[-1] + inst_center_num)

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb) + torch.randn(3) * 0.1)
            labels.append(torch.from_numpy(label))
            instance_labels.append(torch.from_numpy(instance_label))

            instance_infos.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)
            # instance_center.extend(inst_center)
            instance_center.append(torch.from_numpy(inst_center))

        # merge all the scenes in the batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)
        inst_batch_offsets = torch.tensor(inst_batch_offsets, dtype=torch.int)  # int (B+1)
        inst_center_batch_offsets = torch.tensor(inst_center_batch_offsets, dtype=torch.int)

        locs = torch.cat(locs, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)
        labels = torch.cat(labels, 0).long()  # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()  # long (N)

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)  # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)
        # instance_center = torch.tensor(instance_center, dtype=torch.int)  # int (total_nInst)
        instance_center = torch.cat(instance_center, 0).to(torch.float32)  # int (total_nInst)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        # voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
                'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
                'instance_center': instance_center, 'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape,
                'inst_batch_offsets': inst_batch_offsets, 'inst_center_batch_offsets': inst_center_batch_offsets}

    def valMerge(self, id):
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int
        instance_center = []  # (total_nInst), int

        batch_offsets = [0]
        inst_batch_offsets = [0]
        inst_center_batch_offsets = [0]

        total_inst_num = 0
        scene_names = []
        for i, idx in enumerate(id):
            scene_names.append(os.path.basename(self.val_file_names[idx].split('/')[-1]).strip(self.filename_suffix))
            if self.filename_suffix == '.txt':
                data = self.val_files[idx]
                xyz_origin, rgb, label, instance_label = data[:, :3], data[:, 3:6], data[:, 6], data[:, 7]
            else:
                xyz_origin, rgb, label, instance_label, _ = self.val_files[idx]

            # flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, False, False, True)

            # scale
            xyz = xyz_middle * self.scale

            # offset
            xyz -= xyz.min(0)

            # crop
            xyz, valid_idxs = self.crop(xyz)

            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

            # get instance information
            inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32))
            inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]  # (nInst), list
            inst_center = inst_infos["instance_center"]  # (nInst, 3), list
            inst_center_num = inst_infos["instance_center_num"]  # (nInst), list

            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

            # merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])
            inst_batch_offsets.append(inst_batch_offsets[-1] + len(inst_pointnum))
            inst_center_batch_offsets.append(inst_center_batch_offsets[-1] + inst_center_num)

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb))
            labels.append(torch.from_numpy(label))
            instance_labels.append(torch.from_numpy(instance_label))

            instance_infos.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)
            # instance_center.extend(inst_center)
            instance_center.append(torch.from_numpy(inst_center))

        # merge all the scenes in the batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)
        inst_batch_offsets = torch.tensor(inst_batch_offsets, dtype=torch.int)  # int (B+1)
        inst_center_batch_offsets = torch.tensor(inst_center_batch_offsets, dtype=torch.int)

        locs = torch.cat(locs, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)
        labels = torch.cat(labels, 0).long()  # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()  # long (N)

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)  # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)
        # instance_center = torch.tensor(instance_center, dtype=torch.int)  # int (total_nInst)
        instance_center = torch.cat(instance_center, 0).to(torch.float32)  # int (total_nInst)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        # voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
                'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
                'instance_center': instance_center, 'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape,
                'inst_batch_offsets': inst_batch_offsets, 'inst_center_batch_offsets': inst_center_batch_offsets,
                'scene_names': scene_names}


    def testMerge(self, id):
        locs = []
        locs_float = []
        feats = []

        labels = []  #
        instance_labels = []

        building_types = []

        batch_offsets = [0]
        scene_names = []
        for i, idx in enumerate(id):
            scene_names.append(os.path.basename(self.test_file_names[idx].split('/')[-1]).strip(self.filename_suffix))
            if self.test_split == 'val' or self.test_split == 'test_w_label':
                if self.filename_suffix == '.txt':
                    data = self.test_files[idx]
                    xyz_origin, rgb, label, instance_label, building_type = data[:, :3], data[:, 3:6], data[:, 6], data[:, 7], data[:, 8]
                else:
                    xyz_origin, rgb, label, instance_label, building_type = self.test_files[idx]
            elif self.test_split == 'test':
                if self.filename_suffix == '.txt':
                    data = self.test_files[idx]
                    xyz_origin, rgb = data[:, :3], data[:, 3:6]
                else:
                    xyz_origin, rgb = self.test_files[idx]
            else:
                print("Wrong test split: {}!".format(self.test_split))
                exit(0)

            # flip x / rotation
            # xyz_middle = self.dataAugment(xyz_origin, False, False, True)
            xyz_middle = xyz_origin

            # scale
            xyz = xyz_middle * self.scale

            # offset
            xyz -= xyz.min(0)

            # merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb))

            if self.test_split == 'val' or self.test_split == 'test_w_label':
                labels.append(torch.from_numpy(label))
                instance_labels.append(torch.from_numpy(instance_label))
                building_types.append(torch.from_numpy(building_type))

        if self.test_split == 'val' or self.test_split == 'test_w_label':
            labels = torch.cat(labels, 0).long()  # long (N)
            instance_labels = torch.cat(instance_labels, 0).long()  # long (N)
            building_types = torch.cat(building_types, 0).long()

        # merge all the scenes in the batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        # voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        if self.test_split == 'val' or self.test_split == 'test_w_label':
            return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                    'locs_float': locs_float, 'feats': feats,
                    'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape, 'labels': labels,
                    'instance_labels': instance_labels, 'building_types': building_types, 'scene_names': scene_names}

        elif self.test_split == 'test':
            return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                    'locs_float': locs_float, 'feats': feats,
                    'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape, "scene_names": scene_names}
        else:
            assert Exception
