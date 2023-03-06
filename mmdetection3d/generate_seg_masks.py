import argparse
import numpy as np
from os import path as osp
from tqdm import tqdm

import mmcv
import torch
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, set_random_seed
from mmdet.models import build_detector

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet and MMseg infer a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--data-dir', help='directory where the images are in')
    parser.add_argument(
        '--out-dir', help='directory where results will be saved')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    args = parser.parse_args()
  
    return args

def show_instance_seg_mask(result, height, width, score_thr=0.3, out_file=None):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)
        
    out_img = np.zeros((height, width))

    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]
    
    if segms is not None:
        for i, mask in enumerate(segms):
            out_img[mask] = labels[i]
    
    if out_file is not None:
        mmcv.imwrite(out_img, out_file)

def main():
    args = parse_args()

    assert args.data_dir, ('Please specify the argument "--data-dir"')

    device='cuda:0' # Set the device to be used for evaluation

    config = mmcv.Config.fromfile(args.config)
    config.model.pretrained = None

    # Initialize the detector
    model = build_detector(config.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location=device)

    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']

    # We need to set the model's cfg for inference
    model.cfg = config

    model.to(device)
    model.eval()

    data_root = args.data_dir
    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    img_path_gen = mmcv.utils.scandir(data_root, suffix='.jpg', recursive=True)
    for img_path in tqdm(img_path_gen):
        # Use the detector to do inference
        img = mmcv.imread(osp.join(data_root, img_path))
        result = inference_detector(model, img)

        ori_h, ori_w, channels = img.shape

        if args.out_dir:
            base_file = osp.splitext(img_path)[0]
            out_file = osp.join(args.out_dir, base_file + '.png')
        else:
            out_file = None

        show_instance_seg_mask(result, ori_h, ori_w, out_file=out_file)

if __name__ == '__main__':
    main()