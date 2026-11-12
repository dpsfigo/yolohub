# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, nms, ops


class SegmentationPredictor(DetectionPredictor):
    """A class extending the DetectionPredictor class for prediction based on a segmentation model.

    This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
    prediction results.

    Attributes:
        args (dict): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO segmentation model.
        batch (list): Current batch of images being processed.

    Methods:
        postprocess: Apply non-max suppression and process segmentation detections.
        construct_results: Construct a list of result objects from predictions.
        construct_result: Construct a single result object from a prediction.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.segment import SegmentationPredictor
        >>> args = dict(model="yolo26n-seg.pt", source=ASSETS)
        >>> predictor = SegmentationPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize the SegmentationPredictor with configuration, overrides, and callbacks.

        This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
        prediction results.

        Args:
            cfg (dict): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (dict, optional): Dictionary of callback functions to be invoked during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        """Apply non-max suppression and process segmentation detections for each image in the input batch.

        Args:
            preds (tuple): Model predictions, containing bounding boxes, scores, classes, and mask coefficients.
            img (torch.Tensor): Input image tensor in model format, with shape (B, C, H, W).
            orig_imgs (list | torch.Tensor | np.ndarray): Original image or batch of images.

        Returns:
            (list): List of Results objects containing the segmentation predictions for each image in the batch. Each
                Results object includes both bounding boxes and segmentation masks.

        Examples:
            >>> predictor = SegmentationPredictor(overrides=dict(model="yolo26n-seg.pt"))
            >>> results = predictor.postprocess(preds, img, orig_img)
        """
        # RoI heads already decoded one-to-one boxes and per-box mask logits. Prototype heads still ship coefficients.
        roi = ops.roi_mask_pair(preds)
        if roi is not None:
            return self._postprocess_roi(*roi, img, orig_imgs)
        # Extract protos - tuple if PyTorch model or array if exported
        protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        return super().postprocess(preds[0], img, orig_imgs, protos=protos)

    def _postprocess_roi(self, det, roi_masks, img, orig_imgs):
        """Filter one-to-one boxes by confidence and paste each RoI mask into the image."""
        preds, keepi = nms.non_max_suppression(
            det,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            end2end=True,
            return_idxs=True,
        )
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        return [
            self._construct_roi_result(pred, idx, masks, img, orig_img, img_path)
            for pred, idx, masks, orig_img, img_path in zip(preds, keepi, roi_masks, orig_imgs, self.batch[0])
        ]

    def _construct_roi_result(self, pred, idx, roi_masks, img, orig_img, img_path):
        """Build one Results object by pasting RoI logits into the letterboxed or original image."""
        if pred.shape[0] == 0:
            masks = None
        else:
            roi = roi_masks[idx]
            if self.args.retina_masks or self.args.tile:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.paste_roi_masks(roi, pred[:, :4], orig_img.shape[:2])
            else:
                masks = ops.paste_roi_masks(roi, pred[:, :4], img.shape[2:])
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            keep = masks.amax((-2, -1)) > 0  # only keep predictions with masks
            if not keep.all():
                pred, masks = pred[keep], masks[keep]
            if pred.shape[0] == 0:
                masks = None
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)

    def construct_results(self, preds, img, orig_imgs, protos):
        """Construct a list of result objects from the predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.
            protos (torch.Tensor): Prototype masks tensor with shape (B, C, H, W).

        Returns:
            (list[Results]): List of result objects containing the original images, image paths, class names, bounding
                boxes, and masks.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path, proto)
            for pred, orig_img, img_path, proto in zip(preds, orig_imgs, self.batch[0], protos)
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto):
        """Construct a single result object from the prediction.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.
            proto (torch.Tensor): The prototype masks.

        Returns:
            (Results): Result object containing the original image, image path, class names, bounding boxes, and masks.
        """
        if pred.shape[0] == 0:  # save empty boxes
            masks = None
        elif self.args.retina_masks or self.args.tile:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # NHW
        else:
            masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # NHW
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.amax((-2, -1)) > 0  # only keep predictions with masks
            if getattr(self, "_feats", None) is None and not keep.all():  # skip filter if native ReID enabled
                pred, masks = pred[keep], masks[keep]  # indexing is slow
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
