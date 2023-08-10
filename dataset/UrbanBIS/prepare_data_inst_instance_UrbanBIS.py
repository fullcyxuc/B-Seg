import glob, numpy as np, torch
import pandas as pd
import os
import json
import random
import math


def split_pointcloud(cloud, size=50.0, stride=50, split='train'):
    limit_max = np.amax(cloud[:, 0:3], axis=0)
    blocks = []
    if (limit_max[0] > size or limit_max[1] > size) and ('test' not in split):
        width = int(np.ceil((limit_max[0] - size) / stride)) + 1
        depth = int(np.ceil((limit_max[1] - size) / stride)) + 1
        cells = [(x * stride, y * stride) for x in range(width) for y in range(depth)]

        for (x, y) in cells:
            xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
            ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
            cond = xcond & ycond
            block = cloud[cond, :]
            blocks.append(block)
    else:
        blocks.append(cloud[:, :])
    return blocks


def data_aug(file, semantic_keep):
    points = pd.read_csv(file, header=None, delimiter=' ').values
    angle = random.randint(1, 359)
    angle_radians = math.radians(angle)
    rotation_matrix = np.array(
        [[math.cos(angle_radians), -math.sin(angle_radians), 0], [math.sin(angle_radians), math.cos(angle_radians), 0],
         [0, 0, 1]])
    points[:, :3] = points[:, :3].dot(rotation_matrix)
    points_kept = points[np.in1d(points[:, 6], semantic_keep)]
    return points_kept


def prepare_pth_files(files_dir, split, output_folder, aug_times=0):
    ### save the coordinates so that we can merge the data to a single scene after segmentation for visualization
    out_json_path = os.path.join(output_folder, 'coord_shift.json')
    coord_shift = {}
    ### used to increase z range if it is smaller than this, over come the issue where spconv may crash for voxlization.
    z_threshold = 6

    # Map relevant classes to {1,...,14}, and ignored classes to -100
    remapper = np.ones(150) * (-100)
    for i, x in enumerate([0, 1, 2, 3, 4, 5, 6]):
        remapper[x] = i
    # Map instance to -100 based on selected semantic (change a semantic to -100 if you want to ignore it for instance)
    remapper_disable_instance_by_semantic = np.ones(150) * (-100)
    for i, x in enumerate([-100, -100, -100, -100, -100, -100, 6]):
        remapper_disable_instance_by_semantic[x] = i

    ### only augment data for these classes
    semantic_keep = [0, 1, 2, 3, 4, 5, 6]

    files = glob.glob(os.path.join(files_dir, "*.txt"))
    counter = 0
    for file in files:

        for aug_time in range(aug_times + 1):
            if aug_time == 0:
                points = pd.read_csv(file, header=None, delimiter=' ').values
            else:
                points = data_aug(file, semantic_keep)
            name = os.path.basename(file).strip('.txt') + '_%d' % aug_time

            if split != 'test':
                coord_shift['globalShift'] = list(points[:, :3].min(0))
            points[:, :3] = points[:, :3] - points[:, :3].min(0)

            blocks = split_pointcloud(points, size=150, stride=75, split=split)
            for blockNum, block in enumerate(blocks):
                if (len(block) > 50000):
                    out_file_path = os.path.join(output_folder, city + "_" + name + str(blockNum) + '_inst_nostuff.pth')
                    if (block[:, 2].max(0) - block[:, 2].min(0) < z_threshold):
                        block = np.append(block, [[block[:, 0].mean(0), block[:, 1].mean(0),
                                                   block[:, 2].max(0) + (
                                                           z_threshold - (block[:, 2].max(0) - block[:, 2].min(0))),
                                                   block[:, 3].mean(0), block[:, 4].mean(0), block[:, 5].mean(0),
                                                   -100, -100]], axis=0)
                        print("range z is smaller than threshold ")
                        print(name + str(blockNum) + '_inst_nostuff')
                    if split != 'test':
                        out_file_name = name + str(blockNum) + '_inst_nostuff'
                        coord_shift[out_file_name] = list(block[:, :3].mean(0))
                    coords = np.ascontiguousarray(block[:, :3] - block[:, :3].mean(0))

                    # coords = block[:, :3]
                    colors = np.ascontiguousarray(block[:, 3:6]) / 127.5 - 1

                    coords = np.float32(coords)
                    colors = np.float32(colors)
                    building_type = block[:, -1]
                    building_type = building_type.astype(np.int32)
                    if split != 'test':
                        sem_labels = np.ascontiguousarray(block[:, -3])
                        sem_labels = sem_labels.astype(np.int32)
                        sem_labels = remapper[np.array(sem_labels)]

                        instance_labels = np.ascontiguousarray(block[:, -2])
                        instance_labels = instance_labels.astype(np.float32)

                        disable_instance_by_semantic_labels = np.ascontiguousarray(block[:, -3])
                        disable_instance_by_semantic_labels = disable_instance_by_semantic_labels.astype(np.int32)
                        disable_instance_by_semantic_labels = remapper_disable_instance_by_semantic[
                            np.array(disable_instance_by_semantic_labels)]
                        instance_labels = np.where(disable_instance_by_semantic_labels == -100, -100, instance_labels)

                        # map instance from 0.
                        # [1:] because there are -100
                        unique_instances = (np.unique(instance_labels))[1:].astype(np.int32)
                        remapper_instance = np.ones(50000) * (-100)
                        for i, j in enumerate(unique_instances):
                            remapper_instance[j] = i

                        instance_labels = remapper_instance[instance_labels.astype(np.int32)]

                        unique_semantics = (np.unique(sem_labels)).astype(np.int32)

                        if (split == 'train' or split == 'val') and (
                                len(unique_instances) < 3 or len(unique_semantics) < 3):
                            print("unique insance: %d" % len(unique_instances))
                            print("unique semantic: %d" % len(unique_semantics))
                            print()
                            counter += 1
                        else:
                            torch.save((coords, colors, sem_labels, instance_labels, building_type), out_file_path)
                            # ## save text file for each pth file
                            # out_file_path = os.path.join(output_folder, name + str(blockNum) + '.txt')
                            # out_file = open(out_file_path, 'w')
                            # for i in range(len(coords)):
                            #     out_file.write(
                            #         "%f,%f,%f,%f,%f,%f,%d,%d,%d\n" % (coords[i][0], coords[i][1], coords[i][2],
                            #                                           colors[i][0], colors[i][1], colors[i][2],
                            #                                           sem_labels[i], instance_labels[i],
                            #                                           building_type[i]))
                    else:
                        torch.save((coords, colors), out_file_path)
                        # # save text file for each pth file
                        # out_file_path = os.path.join(output_folder, name + str(blockNum) + '.txt')
                        # out_file = open(out_file_path, 'w')
                        # for i in range(len(coords)):
                        #     out_file.write("%f,%f,%f,%f,%f,%f\n" % (coords[i][0], coords[i][1], coords[i][2],
                        #                                            colors[i][0], colors[i][1], colors[i][2]))
    print("Total skipped file :%d" % counter)
    json.dump(coord_shift, open(out_json_path, 'w'))


def prepare_inst_gt(val_out_dir, val_gt_folder, semantic_label_idxs):
    val_files_path = sorted(glob.glob('{}/*_inst_nostuff.pth'.format(val_out_dir)))
    blocks = [torch.load(i) for i in val_files_path]

    for i in range(len(blocks)):
        xyz, rgb, label, instance_label, _ = blocks[i]  # label 0~19 -100;  instance_label 0~instance_num-1 -100
        scene_name = os.path.basename(val_files_path[i]).strip('.pth')
        print('{}/{} {}'.format(i + 1, len(blocks), scene_name))

        instance_label_new = np.zeros(instance_label.shape,
                                      dtype=np.int32)  # 0 for unannotated, xx00y: x for semantic_label, y for inst_id (1~instance_num)

        instance_num = int(instance_label.max()) + 1
        for inst_id in range(instance_num):
            instance_mask = np.where(instance_label == inst_id)[0]
            sem_id = int(label[instance_mask[0]])
            if sem_id == -100:
                sem_id = 0
            semantic_label = semantic_label_idxs[sem_id]
            instance_label_new[instance_mask] = semantic_label * 1000 + inst_id + 1

        np.savetxt(os.path.join(val_gt_folder, scene_name + '.txt'), instance_label_new, fmt='%d')


if __name__ == '__main__':
    city = "Qingdao"  # "Wuhu" or "Longhua" or "Yuehai" or "Lihu" or "Yingrenshi" (only has test set)
    data_folder = os.path.join(os.path.dirname(os.getcwd()), 'UrbanBIS')
    os.makedirs(os.path.join(data_folder, "processed", city), exist_ok=True)

    split = 'train'
    train_files_dir = os.path.join(data_folder, "original", city, split)
    train_out_dir = os.path.join(data_folder, "processed", city, split)
    os.makedirs(train_out_dir, exist_ok=True)
    prepare_pth_files(train_files_dir, split, train_out_dir, aug_times=1)

    split = 'val'
    val_files_dir = os.path.join(data_folder, "original", city, split)
    val_out_dir = os.path.join(data_folder, "processed", city, split)
    os.makedirs(val_out_dir, exist_ok=True)
    prepare_pth_files(val_files_dir, split, val_out_dir)

    split = 'test_w_label'
    test_files_dir = os.path.join(data_folder, "original", city, "test")
    test_out_dir = os.path.join(data_folder, "processed", city, split)
    os.makedirs(test_out_dir, exist_ok=True)
    prepare_pth_files(test_files_dir, split, test_out_dir)

    semantic_label_idxs = [0, 1, 2, 3, 4, 5, 6]
    semantic_label_names = ['terrain', 'vegetation', 'water', 'bridge', 'vehicle', 'boat', 'building']

    val_gt_folder = os.path.join(data_folder, "processed", city, 'val_gt')
    os.makedirs(val_gt_folder, exist_ok=True)
    prepare_inst_gt(val_out_dir, val_gt_folder, semantic_label_idxs)

    test_gt_folder = os.path.join(data_folder, "processed", city, 'test_w_label_gt')
    os.makedirs(test_gt_folder, exist_ok=True)
    prepare_inst_gt(test_out_dir, test_gt_folder, semantic_label_idxs)
