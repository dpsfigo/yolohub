# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Build, train-step, and inference checks for the YOLO26 RoI mask head."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.nn.modules.head import Segment26, Segment26RoI
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import ops


def test_roi_align_matches_torchvision():
    """Aligned RoIAlign crops and their feature gradients match torchvision."""
    from torchvision.ops import roi_align as tv_roi_align

    torch.manual_seed(0)
    feat = torch.randn(2, 3, 8, 10, requires_grad=True)
    feat_ref = feat.detach().clone().requires_grad_(True)
    rois = torch.tensor([[0, 1.2, 2.0, 7.5, 9.0], [1, 0.0, 0.0, 4.0, 5.0], [0, -1.0, 1.0, 3.0, 6.0]])
    got = ops.roi_align(feat, rois, 3, spatial_scale=0.5, sampling_ratio=2)
    ref = tv_roi_align(feat_ref, rois, output_size=3, spatial_scale=0.5, sampling_ratio=2, aligned=True)
    assert torch.allclose(got, ref, atol=1e-5)
    got.sum().backward()
    ref.sum().backward()
    assert torch.allclose(feat.grad, feat_ref.grad, atol=1e-5)
    assert ops.roi_align(feat, rois[:0], 3).shape == (0, 3, 3, 3)


def test_paste_roi_masks_places_logits_inside_the_box():
    """A positive window in the RoI logit map lands inside the box and nowhere else."""
    logits = torch.full((1, 4, 4), -5.0)
    logits[0, 1:3, 1:3] = 5
    boxes = torch.tensor([[10.0, 10.0, 20.0, 30.0]])
    pasted = ops.paste_roi_masks(logits, boxes, (40, 40))
    assert pasted.shape == (1, 40, 40)
    assert pasted.dtype == torch.uint8
    assert pasted[0, 15, 15] == 1
    assert pasted[0, 0, 0] == 0
    assert ops.paste_roi_masks(logits[:0], boxes[:0], (40, 40)).shape == (0, 40, 40)


def test_yolo26_seg_roi_forward_and_loss():
    """The RoI YAML builds, backprops a mask, and returns one mask per one-to-one box."""
    proto = SegmentationModel("yolo26n-seg.yaml", nc=4, verbose=False)
    assert type(proto.model[-1]) is Segment26
    assert not getattr(proto.model[-1], "roi_masks", False)

    yolo = YOLO("yolo26n-seg-roi.yaml")
    assert yolo.task == "segment"
    assert type(yolo.model.model[-1]) is Segment26RoI

    model = SegmentationModel("yolo26n-seg-roi.yaml", nc=4, verbose=False)
    head = model.model[-1]
    assert type(head) is Segment26RoI
    assert head.end2end is True
    assert not hasattr(head, "proto")
    model.args = get_cfg()

    imgsz = 64
    img = torch.rand(2, 3, imgsz, imgsz)
    model.train()
    raw = model(img)
    assert "roi_feats" in raw["one2one"]
    assert "roi_feats" not in raw["one2many"]
    assert "proto" not in raw["one2one"]
    boxes = torch.tensor([[[8.0, 8.0, 40.0, 48.0]], [[12.0, 16.0, 36.0, 44.0]]])
    logits = head.mask_logits(raw["one2one"]["roi_feats"], boxes)
    assert logits.shape == (2, 1, head.mask_size, head.mask_size)
    F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()
    assert next(head.mask_head.parameters()).grad.abs().sum() > 0
    model.zero_grad()

    masks = torch.zeros(2, imgsz // 4, imgsz // 4)
    masks[0, 2:10, 2:10] = 1
    masks[1, 4:10, 3:9] = 1
    batch = {
        "img": img,
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.45, 0.45, 0.6, 0.6], [0.40, 0.50, 0.5, 0.5]]),
        "masks": masks,
    }
    loss, items = model.loss(batch)
    assert torch.isfinite(loss).all()
    assert "seg_loss" in items
    assert torch.isfinite(items["seg_loss"])
    loss.sum().backward()

    model.eval()
    pred = model(img)
    det, roi = ops.roi_mask_pair(pred)
    assert det.shape[-1] == 6
    assert roi.shape[:2] == det.shape[:2]
    assert roi.shape[-2:] == (head.mask_size, head.mask_size)
    assert head.mask_imgsz == (imgsz, imgsz)

    head.export = True
    with pytest.raises(NotImplementedError):
        model(img)
    head.export = False

    forced = roi.clone()
    forced[:, 0].fill_(5)
    predictor = SegmentationPredictor(overrides={"conf": 0, "imgsz": imgsz})
    predictor.model = model
    predictor.batch = (["a.jpg", "b.jpg"], None, None)
    blank = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    results = predictor.postprocess(((det, forced), pred[1]), img, [blank, blank])
    assert len(results) == 2
    assert results[0].masks is not None
    assert tuple(results[0].masks.data.shape[-2:]) == (imgsz, imgsz)

    assert SegmentationValidator._roi_head(model) is head
    validator = SegmentationValidator.__new__(SegmentationValidator)
    validator.args = get_cfg(overrides={"conf": 0, "iou": 0.7})
    validator.nc = head.nc
    validator.roi_head = head
    processed = validator._postprocess_roi(det, forced)
    assert processed[0]["masks"].shape[-2:] == (imgsz // 4, imgsz // 4)
    assert processed[0]["bboxes"].shape[-1] == 4

    head_index = len(model.model) - 1
    frozen = set(head.mask_only_freeze(head_index))
    assert f"{head_index}.roi_proj" not in frozen
    assert f"{head_index}.mask_head" not in frozen
    assert f"{head_index}.cv2" in frozen
    assert f"{head_index}.one2one_cv2" in frozen

    model.fuse()
    fused = model(img)
    fused_det, fused_roi = ops.roi_mask_pair(fused)
    assert fused_roi.shape[:2] == fused_det.shape[:2]
