# ------------------------------------------------------------------------
# UP-DETR
# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch

from models.deformable_detr import DeformableDETR
from models.backbone import build_backbone
from models.detr import DETR, SetCriterion as DETRSetCriterion, PostProcess as DETRPostProcess
from models.def_matcher import build_matcher as build_def_matcher
from models.matcher import build_matcher as build_detr_matcher
from models.segmentation import DETRsegm, PostProcessSegm, PostProcessPanoptic
from models.transformer import build_transformer
from models.updetr import UPDETR, UPDEFDETR
from models.deformable_transformer import build_deforamble_transformer
from models.deformable_detr import DeformableDETR, SetCriterion as DefSetCriterion, PostProcess as DefPostProcess


def build_model(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    if args.dataset_file=="ImageNet":
        num_classes = 2 # feel free to this num_classes, positive integer larger than 1 is OK.
    device = torch.device(args.device)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef}
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})

        # only in def detr impl.
        if args.model == 'deformable_detr':
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)


    losses = ['labels', 'boxes', 'cardinality']
    if args.feature_recon:
        losses += ['feature']
        weight_dict['loss_feature'] = 1

    if args.masks:
        losses += ["masks"]

    backbone = build_backbone(args)

    if args.model == 'deformable_detr':
        transformer = build_deforamble_transformer(args)
        model = UPDEFDETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            num_feature_levels=args.num_feature_levels,
            aux_loss=args.aux_loss,
            with_box_refine=args.with_box_refine,
            two_stage=args.two_stage,
            num_patches=args.num_patches,
            feature_recon=args.feature_recon,
            query_shuffle=args.query_shuffle
        )
        matcher = build_def_matcher(args)
        criterion = DefSetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,
                                 invariance_equivariance_loss=args.invariance_equivariance_loss)
        postprocessors = {'bbox': DefPostProcess()}

    elif args.model == 'detr':
        transformer = build_transformer(args)
        model = UPDETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            num_patches=args.num_patches,
            feature_recon=args.feature_recon,
            query_shuffle=args.query_shuffle)
        matcher = build_detr_matcher(args)
        criterion = DETRSetCriterion(num_classes, matcher, weight_dict, args.eos_coef, losses)
        postprocessors = {'bbox': DETRPostProcess()}
    else:
        raise ValueError("Wrong model.")

    criterion.to(device)

    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))


    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors

