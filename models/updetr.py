# ------------------------------------------------------------------------
# UP-DETR
# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved.
"""
UPDETR model
"""
import torch
from torch import nn

from models.detr import MLP
from util.misc import NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid
from .deformable_detr import DeformableDETR
from .detr import DETR
import torch.nn.functional as F

class UPDETR(DETR):
    """ This is the UPDETR module for pre-training.
    UPDETR inherits from DETR with same backbone,transformer,object queries and etc."""

    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False,
                 feature_recon=True, query_shuffle=False, mask_ratio=0.1, num_patches=10):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            feature_recon: if set, feature reconstruction branch is to be used.
            query_shuffle: if set, shuffle object query during the pre-training.
            mask_ratio: mask ratio of query patches.
            It masks some query patches during the pre-training, which is similar to Dropout.
            num_patches: number of query patches, which is added to the decoder.
        """
        super().__init__(backbone, transformer, num_classes, num_queries, aux_loss)
        hidden_dim = transformer.d_model
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # pooling used for the query patch feature
        # align the patch feature dim to query patch dim.
        self.patch2query = nn.Linear(backbone.num_channels, hidden_dim)
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        self.feature_recon = feature_recon
        if self.feature_recon:
            # align the transformer feature to the CNN feature, which is used for the feature reconstruction
            self.feature_align = MLP(hidden_dim, hidden_dim, backbone.num_channels, 2)

        self.query_shuffle = query_shuffle
        assert num_queries % num_patches == 0  # for simplicity
        query_per_patch = num_queries // num_patches
        # the attention mask is fixed during the pre-training
        self.attention_mask = torch.ones(self.num_queries, self.num_queries) * float('-inf')
        for i in range(query_per_patch):
            self.attention_mask[i * query_per_patch:(i + 1) * query_per_patch,
            i * query_per_patch:(i + 1) * query_per_patch] = 0

    def forward(self, samples: NestedTensor, patches: torch.Tensor):
        """ The forward expects a NestedTensor samples and patches Tensor.
            samples consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            patches is a torch Tensor, of shape [batch_size x num_patches x 3 x SH x SW]
            The size of patches are small than samples

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        batch_num_patches = patches.shape[1]
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        bs = patches.size(0)
        patches = patches.flatten(0, 1)
        patch_feature = self.backbone(patches)
        patch_feature_gt = self.avgpool(patch_feature[-1]).flatten(1)

        # align the dim of patch feature with object query with a linear layer
        # pay attention to the difference between "torch.repeat" and "torch.repeat_interleave"
        # it converts the input from "1,2,3,4" to "1,2,3,4,1,2,3,4,1,2,3,4" by torch.repeat
        # "1,2,3,4" to "1,1,1,2,2,2,3,3,3,4,4,4" by torch.repeat_interleave, which is our target.
        patch_feature = self.patch2query(patch_feature_gt) \
            .view(bs, batch_num_patches, -1) \
            .repeat_interleave(self.num_queries // self.num_patches, dim=1) \
            .permute(1, 0, 2) \
            .contiguous()

        # if object query shuffle, we shuffle the index of object query embedding,
        # which simulate to adding patch feature to object query randomly.
        idx = torch.randperm(self.num_queries) if self.query_shuffle else torch.arange(self.num_queries)

        if self.training:
            # for training, it uses fixed number of query patches.
            mask_query_patch = (torch.rand(self.num_queries, bs, 1, device=patches.device) > self.mask_ratio).float()
            # mask some query patch and add query embedding
            patch_feature = patch_feature * mask_query_patch \
                            + self.query_embed.weight[idx, :].unsqueeze(1).repeat(1, bs, 1)
            hs = self.transformer(
                self.input_proj(src), mask, patch_feature, pos[-1], self.attention_mask.to(patch_feature.device))[0]
        else:
            num_queries = batch_num_patches * self.num_queries // self.num_patches
            # for test, it supports x query patches, where x<=self.num_queries.
            patch_feature = patch_feature + self.query_embed.weight[:num_queries, :].unsqueeze(1).repeat(1, bs, 1)
            hs = self.transformer(
                self.input_proj(src), mask, patch_feature, pos[-1], self.attention_mask.to(patch_feature.device)[:num_queries,:num_queries])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        if self.feature_recon:
            outputs_feature = self.feature_align(hs)
            out = {'pred_logits': outputs_class[-1], 'pred_feature': outputs_feature[-1],
                   'gt_feature': patch_feature_gt,
                   'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_feature, patch_feature_gt)
        else:
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = super()._set_aux_loss(outputs_class, outputs_coord)  # use detr func
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_feature, backbone_output):
        # different from _set_aux_loss_base, it has extra feature reconstruction branch
        # The "ground truth" of the feature reconstruction is constructed during the model forward
        # So, we name them as 'gt_feature', which is put with 'pred_feature' together.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_feature': c, 'gt_feature': backbone_output}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_feature[:-1])]

class UPDEFDETR(DeformableDETR):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, feature_recon=True,
                 query_shuffle=False, mask_ratio=0.1, num_patches=10):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__(backbone, transformer, num_classes, num_queries, num_feature_levels, aux_loss)
        hidden_dim = transformer.d_model
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # pooling used for the query patch feature
        self.patch2query = nn.Linear(backbone.num_channels[-1], 2*hidden_dim)
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        self.feature_recon = feature_recon
        if self.feature_recon:
            # align the transformer feature to the CNN feature, which is used for the feature reconstruction
            self.feature_align = MLP(hidden_dim, hidden_dim, backbone.num_channels[-1], 2)

        self.query_shuffle = query_shuffle
        assert num_queries % num_patches == 0  # for simplicity

    def forward(self, samples: NestedTensor, patches: torch.Tensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        batch_num_patches = patches.shape[1]
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)


        bs = patches.size(0)
        patches = patches.flatten(0, 1)
        patch_feature = self.backbone(patches)
        patch_feature_gt = self.avgpool(patch_feature[-1]).flatten(1)

        patch_feature = self.patch2query(patch_feature_gt) \
            .view(bs, batch_num_patches, -1) \
            .repeat_interleave(self.num_queries // self.num_patches, dim=1) \
            .permute(1, 0, 2) \
            .contiguous()

        idx = torch.randperm(self.num_queries) if self.query_shuffle else torch.arange(self.num_queries)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        mask_query_patch = (torch.rand(self.num_queries, bs, 1, device=patches.device) > self.mask_ratio).float()
        # mask some query patch and add query embedding
        patch_feature = patch_feature * mask_query_patch \
                        + self.query_embed.weight[idx, :].unsqueeze(1).repeat(1, bs, 1)

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, patch_feature, pos)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        if self.feature_recon:
            outputs_feature = self.feature_align(hs)
            out = {'pred_logits': outputs_class[-1],
                   'pred_feature': outputs_feature[-1],
                   'gt_feature': patch_feature_gt,
                   'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_feature, patch_feature_gt)
        else:
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = super()._set_aux_loss(outputs_class, outputs_coord)  # use detr func

        if self.two_stage:
            raise ValueError("Not supported here.")
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_feature, backbone_output):  # , outputs_features):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_feature': c, 'gt_feature': backbone_output}  # , 'pred_features': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_feature[:-1])]  # , outputs_features[:-1])]
