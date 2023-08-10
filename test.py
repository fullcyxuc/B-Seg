import torch
import time
import numpy as np
import random
import os

from util.config import cfg
cfg.task = 'test'
from util.log import logger
import util.utils as utils
import util.eval as eval

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # choose gpu id to train on the server

def init():
    global result_dir
    result_dir = os.path.join(cfg.exp_path, 'result', cfg.split)
    backup_dir = os.path.join(result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'predicted_masks'), exist_ok=True)
    os.system('cp test.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    global semantic_label_idx
    semantic_label_idx = [0, 1, 2, 3, 4, 5, 6]  # UrabnBIS

    logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)


def test(model, model_fn, data_name, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    if cfg.dataset is not None:
        if data_name == "urbanbis":
            from dataset.UrbanBIS.urbanbis_inst import Dataset
            dataset = Dataset(test=True)
            dataset.testLoader()
        else:
            print("Error: no data loader - " + data_name)
            exit(0)
    else:
        print("Error: invalid dataset - " + cfg.dataset)
        exit(0)

    dataloader = dataset.test_data_loader

    with torch.no_grad():
        model = model.eval()

        total_end1 = 0.
        matches = {}
        for i, batch in enumerate(dataloader):
            torch.cuda.empty_cache()
            # inference
            start1 = time.time()
            preds = model_fn(batch, model, epoch)
            end1 = time.time() - start1

            # decode results for evaluation
            N = batch['feats'].shape[0]
            test_scene_name = os.path.basename(dataset.test_file_names[int(batch['id'][0])].split('/')[-1]).strip('.pth')
            dataset_name = test_scene_name.split("_")[0]
            print(test_scene_name)
            semantic_scores = preds['semantic']  # (N, nClass=7) float32, cuda
            semantic_pred = semantic_scores.max(1)[1]  # (N) long, cuda
            pt_offsets = preds['pt_offsets']    # (N, 3), float32, cuda
            if (epoch > cfg.prepare_epochs):
                if 'score' in preds:
                    scores = preds['score']   # (nProposal, 1) float, cuda
                    scores_pred = torch.sigmoid(scores.view(-1))

                    proposals_idx, proposals_offset = preds['proposals']
                    # proposals_idx: (sumNPoint, 2), int, cpu, [:, 0] for cluster_id, [:, 1] for corresponding point idxs in N
                    # proposals_offset: (nProposal + 1), int, cpu
                    proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, N), dtype=torch.int, device=scores_pred.device)
                    proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
                    # (nProposal, N), int, cuda

                    semantic_id = torch.tensor(semantic_label_idx, device=scores_pred.device) \
                        [semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]] # (nProposal), long

                    # score threshold
                    score_mask = (scores_pred > cfg.TEST_SCORE_THRESH)
                    scores_pred = scores_pred[score_mask]
                    proposals_pred = proposals_pred[score_mask]
                    semantic_id = semantic_id[score_mask]

                    # npoint threshold
                    proposals_pointnum = proposals_pred.sum(1)
                    npoint_mask = (proposals_pointnum >= cfg.TEST_NPOINT_THRESH)
                    scores_pred = scores_pred[npoint_mask]
                    proposals_pred = proposals_pred[npoint_mask]
                    semantic_id = semantic_id[npoint_mask]

                    clusters = proposals_pred
                    cluster_scores = scores_pred
                    cluster_semantic_id = semantic_id
                    nclusters = clusters.shape[0]
                else:
                    nclusters = 0

                # prepare for evaluation
                if cfg.eval:
                    pred_info = {}
                    pred_info['conf'] = cluster_scores.cpu().numpy() if 'score' in preds else None
                    pred_info['label_id'] = cluster_semantic_id.cpu().numpy() if 'score' in preds else None
                    pred_info['mask'] = clusters.cpu().numpy() if 'score' in preds else None
                    gt_file = os.path.join(cfg.data_root, dataset_name, cfg.split + '_gt', test_scene_name + '.txt')
                    gt2pred, pred2gt = eval.assign_instances_for_scan(test_scene_name, pred_info, gt_file)

                    matches[test_scene_name] = {}
                    matches[test_scene_name]['gt'] = gt2pred
                    matches[test_scene_name]['pred'] = pred2gt
                
                    if cfg.split == 'val' or cfg.split == 'test_w_label':
                        matches[test_scene_name]['seg_gt'] = batch['labels']
                        matches[test_scene_name]['seg_pred'] = semantic_pred
                        matches[test_scene_name]['building_type_gt'] = batch['building_types']

                        # for height type building
                        instance_labels = batch['instance_labels']
                        unique_inst_label = torch.unique(instance_labels)
                        building_height = torch.tensor([-100] * len(semantic_pred)).int()
                        z = batch['locs_float'][:, 2]
                        for ins in unique_inst_label:
                            if ins == -100:
                                continue
                            mask = instance_labels == ins
                            max_z = torch.max(z[mask])
                            min_z = torch.min(z[mask])
                            height = max_z - min_z
                            if height < 27:
                                building_height[mask] = 0
                            elif 27 <= height < 100:
                                building_height[mask] = 1
                            else:
                                building_height[mask] = 2
                        matches[test_scene_name]['building_height_gt'] = building_height

            # save files
            if cfg.save_semantic:
                os.makedirs(os.path.join(result_dir, 'semantic'), exist_ok=True)
                semantic_np = semantic_pred.cpu().numpy()
                np.save(os.path.join(result_dir, 'semantic', test_scene_name + '.npy'), semantic_np)

            if cfg.save_pt_offsets:
                os.makedirs(os.path.join(result_dir, 'coords_offsets'), exist_ok=True)
                pt_offsets_np = pt_offsets.cpu().numpy()
                coords_np = batch['locs_float'].numpy()
                coords_offsets = np.concatenate((coords_np, pt_offsets_np), 1)   # (N, 6)
                np.save(os.path.join(result_dir, 'coords_offsets', test_scene_name + '.npy'), coords_offsets)

            if(epoch > cfg.prepare_epochs and cfg.save_instance):
                f = open(os.path.join(result_dir, test_scene_name + '.txt'), 'w')
                for proposal_id in range(nclusters):
                    clusters_i = clusters[proposal_id].cpu().numpy()  # (N)
                    semantic_label = np.argmax(np.bincount(semantic_pred[np.where(clusters_i == 1)[0]].cpu()))
                    score = cluster_scores[proposal_id]
                    f.write('predicted_masks/{}_{:03d}.txt {} {:.4f}'.format( \
                        test_scene_name, proposal_id, semantic_label_idx[semantic_label], score))
                    if proposal_id < nclusters - 1:
                        f.write('\n')
                    np.savetxt(os.path.join(result_dir, 'predicted_masks', test_scene_name + '_%03d.txt' % (proposal_id)), clusters_i, fmt='%d')
                f.close()

            logger.info("instance iter: {}/{} point_num: {} ncluster: {} inference time: {:.2f}s".format( \
                batch['id'][0] + 1, len(dataset.test_files), N, nclusters, end1))
            total_end1 += end1

        # evaluation
        if cfg.eval:
            ap_scores = eval.evaluate_matches(matches)
            avgs = eval.compute_averages(ap_scores)
            eval.print_results(avgs)

        logger.info("whole set inference time: {:.2f}s, latency per frame: {:.2f}ms".format(total_end1, total_end1 / len(dataloader) * 1000))

        # evaluate semantic segmantation accuracy and mIoU
        if cfg.split == 'val' or cfg.split == 'test_w_label':
            seg_accuracy = evaluate_semantic_segmantation_accuracy(matches)
            logger.info("semantic_segmantation_accuracy: {:.4f}".format(seg_accuracy))
            iou_list = evaluate_semantic_segmantation_miou(matches)
            logger.info(iou_list)
            iou_list = torch.tensor(iou_list)
            miou = iou_list.mean()
            logger.info("semantic_segmantation_mIoU: {:.4f}".format(miou))

            # building type segmentation iou
            evaluate_building_type_segmantation_miou(matches)

            # building height type segmentation iou
            evaluate_building_height_type_segmantation_miou(matches)


def evaluate_semantic_segmantation_accuracy(matches):
    seg_gt_list = []
    seg_pred_list = []
    for k, v in matches.items():
        seg_gt_list.append(v['seg_gt'])
        seg_pred_list.append(v['seg_pred'])
    seg_gt_all = torch.cat(seg_gt_list, dim=0).cuda()
    seg_pred_all = torch.cat(seg_pred_list, dim=0).cuda()
    assert seg_gt_all.shape == seg_pred_all.shape
    correct = (seg_gt_all[seg_gt_all != -100] == seg_pred_all[seg_gt_all != -100]).sum()
    whole = (seg_gt_all != -100).sum()
    seg_accuracy = correct.float() / whole.float()
    return seg_accuracy

def evaluate_semantic_segmantation_miou(matches):
    seg_gt_list = []
    seg_pred_list = []
    for k, v in matches.items():
        seg_gt_list.append(v['seg_gt'])
        seg_pred_list.append(v['seg_pred'])
    seg_gt_all = torch.cat(seg_gt_list, dim=0).cuda()
    seg_pred_all = torch.cat(seg_pred_list, dim=0).cuda()
    assert seg_gt_all.shape == seg_pred_all.shape
    iou_list = [0.0] * 7
    # for _index in seg_gt_all.unique():
    for _index in range(7):
        if _index != -100:
            intersection = ((seg_gt_all == _index) & (seg_pred_all == _index)).sum()
            union = ((seg_gt_all == _index) | (seg_pred_all == _index)).sum()
            iou = intersection.float() / union if union != 0 else 0.0
            # iou_list.append(iou)
            iou_list[_index] = iou

    return iou_list

def evaluate_building_type_segmantation_miou(matches):
    seg_gt_list = []
    seg_pred_list = []
    building_type_gt_list = []
    for k, v in matches.items():
        seg_gt_list.append(v['seg_gt'])
        seg_pred_list.append(v['seg_pred'])
        building_type_gt_list.append(v['building_type_gt'])
    seg_gt_all = torch.cat(seg_gt_list, dim=0).cuda()
    seg_pred_all = torch.cat(seg_pred_list, dim=0).cuda()
    building_type_gt_all = torch.cat(building_type_gt_list, dim=0).cuda()
    assert seg_gt_all.shape == seg_pred_all.shape
    mask1 = seg_gt_all == 6
    mask2 = building_type_gt_all != -100
    assert (mask1 != mask2).sum() == 0
    seg_pred_all[seg_gt_all != 6] = 0
    iou_list = []
    for _index in range(8):
        building_type_mask = (building_type_gt_all == _index)
        # print("building type %d has %d points" % (_index, building_type_mask.sum().item()))
        intersection = ((seg_gt_all[building_type_mask] == 6) & (seg_pred_all[building_type_mask] == 6)).sum()
        union = ((seg_gt_all[building_type_mask] == 6) | (seg_pred_all[building_type_mask] == 6)).sum()
        iou = intersection.float() / union if union != 0 else torch.tensor([0.0]).cuda()
        iou_list.append(iou)

    logger.info([i.item() for i in iou_list])
    iou_list = torch.tensor(iou_list)
    miou = iou_list.mean()
    logger.info("building_type_segmantation_mIoU: {:.4f}".format(miou))

    return iou_list

def evaluate_building_height_type_segmantation_miou(matches):
    seg_gt_list = []
    seg_pred_list = []
    building_height_gt_list = []
    for k, v in matches.items():
        seg_gt_list.append(v['seg_gt'])
        seg_pred_list.append(v['seg_pred'])
        building_height_gt_list.append(v['building_height_gt'])
    seg_gt_all = torch.cat(seg_gt_list, dim=0).cuda()
    seg_pred_all = torch.cat(seg_pred_list, dim=0).cuda()
    building_height_gt_all = torch.cat(building_height_gt_list, dim=0).cuda()
    assert seg_gt_all.shape == seg_pred_all.shape
    mask1 = seg_gt_all == 6
    mask2 = building_height_gt_all != -100
    assert (mask1 != mask2).sum() == 0
    seg_pred_all[seg_gt_all != 6] = 0
    iou_list = []
    for _index in range(3):
        building_height_mask = (building_height_gt_all == _index)
        # print("building type %d has %d points" % (_index, building_type_mask.sum().item()))
        intersection = ((seg_gt_all[building_height_mask] == 6) & (seg_pred_all[building_height_mask] == 6)).sum()
        union = ((seg_gt_all[building_height_mask] == 6) | (seg_pred_all[building_height_mask] == 6)).sum()
        iou = intersection.float() / union if union != 0 else torch.tensor([0.0]).cuda()
        iou_list.append(iou)

    logger.info([i.item() for i in iou_list])
    iou_list = torch.tensor(iou_list)
    miou = iou_list.mean()
    logger.info("building_height_type_segmantation_mIoU: {:.4f}".format(miou))

    return iou_list

def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


if __name__ == '__main__':
    init()
    torch.backends.cudnn.enabled=False

    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    logger.info('=> creating model ...')
    logger.info('Classes: {}'.format(cfg.classes))

    if model_name =='BSeg':
        from model.BSeg import BSeg as Network
        from model.BSeg import model_fn_decorator
    else:
        print("Error: no model version " + model_name)
        exit(0)
    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    logger.info('#classifier parameters (model): {}'.format(sum([x.nelement() for x in model.parameters()])))
    model_fn = model_fn_decorator(test=True)

    # load model
    utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5],
        use_cuda, epoch=cfg.test_epoch, dist=False, f=cfg.pretrain)

    # resume from the latest epoch, or specify the epoch to restore

    # evaluate
    test(model, model_fn, data_name, cfg.test_epoch)
