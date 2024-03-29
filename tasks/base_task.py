import torch
import pytorch_lightning as pl
import os.path as osp
import json
import time

from optimizers import create_optimizer
from schedulers import create_scheduler
from models import create_model
from utils import *


class BaseTask(pl.LightningModule):

    def __init__(self, cfg, log):
        super().__init__()
        self.cfg = cfg
        self.txt_log = log
        self.model = create_model(cfg.model_cfg, log)
        self.txt_log.info('Model size = %.2f MB' % self.compute_model_size())
        if 'KITTI' in cfg.dataset_cfg.dataset_type:
            self.prec = TorchPrecision()
            self.succ = TorchSuccess()
            self.runtime = TorchRuntime()
            self.n_frames = TorchNumFrames()
            if cfg.save_test_result:
                self.pred_bboxes = []
        elif 'Waymo' in cfg.dataset_cfg.dataset_type:
            self.succ_total = AverageMeter()
            self.prec_total = AverageMeter()
            self.succ_easy = AverageMeter()
            self.prec_easy = AverageMeter()
            self.succ_medium = AverageMeter()
            self.prec_medium = AverageMeter()
            self.succ_hard = AverageMeter()
            self.prec_hard = AverageMeter()
            self.n_frames_total = 0
            self.n_frames_easy = 0
            self.n_frames_medium = 0
            self.n_frames_hard = 0
        elif 'NuScenes' in cfg.dataset_cfg.dataset_type:
            self.prec = TorchPrecision()
            self.succ = TorchSuccess()
            self.n_frames_total = 0
            self.n_frames_key = 0
        else:
            raise NotImplementedError(
                '%s has not been supported!' % cfg.dataset_cfg.dataset_type)

    def compute_model_size(self):
        num_param = sum([p.numel() for p in self.model.parameters()])
        param_size = num_param * 4 / 1024 / 1024  # MB
        return param_size

    def configure_optimizers(self):
        optimizer = create_optimizer(self.cfg.optimizer_cfg, self.parameters())
        scheduler = create_scheduler(self.cfg.scheduler_cfg, optimizer)
        return dict(
            optimizer=optimizer,
            lr_scheduler=scheduler
        )

    def training_step(self, *args, **kwargs):
        raise NotImplementedError(
            'Training_step has not been implemented!')

    def on_validation_epoch_start(self):
        self.prec.reset()
        self.succ.reset()
        self.runtime.reset()
        self.n_frames.reset()

    def forward_on_tracklet(self, tracklet):
        raise NotImplementedError(
            'Forward_on_tracklet has not been implemented!')

    def validation_step(self, batch, batch_idx):
        tracklet = batch[0]
        start_time = time.time()
        pred_bboxes, gt_bboxes = self.forward_on_tracklet(tracklet)
        end_time = time.time()
        runtime = end_time-start_time
        n_frames = len(tracklet)
        overlaps, accuracies = [], []
        for pred_bbox, gt_bbox in zip(pred_bboxes, gt_bboxes):
            overlap = estimateOverlap(
                gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space, up_axis=self.cfg.dataset_cfg.up_axis)
            accuracy = estimateAccuracy(
                gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space, up_axis=self.cfg.dataset_cfg.up_axis)
            overlaps.append(overlap)
            accuracies.append(accuracy)

        self.succ(torch.tensor(overlaps, device=self.device))
        self.prec(torch.tensor(accuracies, device=self.device))
        self.runtime(torch.tensor(runtime, device=self.device),
                     torch.tensor(n_frames, device=self.device))
        self.n_frames(torch.tensor(n_frames, device=self.device))

    def on_validation_epoch_end(self):

        self.log('precesion', self.prec.compute(), prog_bar=True)
        self.log('success', self.succ.compute(), prog_bar=True)
        self.log('runtime', self.runtime.compute(), prog_bar=True)
        self.log('n_frames', self.n_frames.compute(), prog_bar=True)

    def _on_test_epoch_start_kitti_format(self):
        self.prec.reset()
        self.succ.reset()
        self.runtime.reset()
        self.n_frames.reset()
        if self.cfg.model_cfg.model_type == 'MBPTrack':
            with torch.no_grad():
                for _ in range(100):
                    backbone_input = dict(
                        pcds=torch.randn(1, 1, 1024, 3).cuda(),
                    )
                    trfm_input = dict(
                        xyz=torch.randn(1, 128, 3).cuda(),
                        feat=torch.randn(1, 128, 128).cuda(),
                        memory=dict(
                            feat=torch.randn(2, 1, 128, 3, 128).cuda(),
                            xyz=torch.randn(1, 3, 128, 3).cuda(),
                            mask=torch.randn(1, 3, 128).cuda(),
                        ),
                    )
                    loc_input = dict(
                        xyz=torch.randn(1, 128, 3).cuda(),
                        geo_feat=torch.randn(1, 128, 128).cuda(),
                        mask_feat=torch.randn(1, 128, 128).cuda(),
                        lwh=torch.ones(1, 3).cuda()
                    )
                    _ = self.model(backbone_input, 'embed')
                    _ = self.model(trfm_input, 'propagate')
                    _ = self.model(loc_input, 'localize')

    def _on_test_epoch_start_waymo_format(self):
        self.succ_total.reset()
        self.prec_total.reset()
        self.succ_easy.reset()
        self.prec_easy.reset()
        self.succ_medium.reset()
        self.prec_medium.reset()
        self.succ_hard.reset()
        self.prec_hard.reset()
        self.n_frames_total = 0
        self.n_frames_easy = 0
        self.n_frames_medium = 0
        self.n_frames_hard = 0

    def _on_test_epoch_start_nuscenes_format(self):
        self.prec.reset()
        self.succ.reset()
        self.n_frames_total = 0
        self.n_frames_key = 0

    def on_test_epoch_start(self):
        if 'KITTI' in self.cfg.dataset_cfg.dataset_type:
            self._on_test_epoch_start_kitti_format()
        elif 'Waymo' in self.cfg.dataset_cfg.dataset_type:
            self._on_test_epoch_start_waymo_format()
        elif 'NuScenes' in self.cfg.dataset_cfg.dataset_type:
            self._on_test_epoch_start_nuscenes_format()

    def _test_step_kitti_format(self, batch, batch_idx):
        tracklet = batch[0]
        torch.cuda.synchronize()
        start_time = time.time()
        pred_bboxes, gt_bboxes = self.forward_on_tracklet(tracklet)
        torch.cuda.synchronize()
        end_time = time.time()
        runtime = end_time-start_time
        n_frames = len(tracklet)
        if self.cfg.save_test_result:
            self.pred_bboxes.append((batch_idx, pred_bboxes))
        overlaps, accuracies = [], []
        for pred_bbox, gt_bbox in zip(pred_bboxes, gt_bboxes):
            overlap = estimateOverlap(
                gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space, up_axis=self.cfg.dataset_cfg.up_axis)
            accuracy = estimateAccuracy(
                gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space, up_axis=self.cfg.dataset_cfg.up_axis)
            overlaps.append(overlap)
            accuracies.append(accuracy)
        self.succ(torch.tensor(overlaps, device=self.device))
        self.prec(torch.tensor(accuracies, device=self.device))
        self.runtime(torch.tensor(runtime, device=self.device),
                     torch.tensor(n_frames, device=self.device))
        self.n_frames(torch.tensor(n_frames, device=self.device))
        self.txt_log.info('Prec=%.3f Succ=%.3f Frames=%d RunTime=%.6f' % (
            self.prec.compute(), self.succ.compute(), self.n_frames.compute(), self.runtime.compute()))
        self.log('precesion', self.prec.compute(), prog_bar=True, logger=False)
        self.log('success', self.succ.compute(), prog_bar=True, logger=False)
        self.log('n_frames', self.n_frames.compute(),
                 prog_bar=True, logger=False)

    def _test_step_waymo_format(self, batch, batch_idx):
        # if batch_idx != 0:
        #     return
        tracklet = batch[0]
        tracklet_length = len(tracklet) - 1
        pred_bboxes, gt_bboxes = self.forward_on_tracklet(tracklet)
        n_frames = len(tracklet)

        success = TorchSuccess()
        precision = TorchPrecision()

        overlaps, accuracies = [], []
        for pred_bbox, gt_bbox in zip(pred_bboxes, gt_bboxes):
            overlap = estimateWaymoOverlap(
                gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space)
            accuracy = estimateAccuracy(
                gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space, up_axis=self.cfg.dataset_cfg.up_axis)
            overlaps.append(overlap)
            accuracies.append(accuracy)
        success(torch.tensor(overlaps, device=self.device))
        precision(torch.tensor(accuracies, device=self.device))
        success = success.compute() if type(
            success.compute()) == float else success.compute().item()
        precision = precision.compute() if type(
            precision.compute()) == float else precision.compute().item()

        self.succ_total.update(success, n=tracklet_length)
        self.prec_total.update(precision, n=tracklet_length)
        self.n_frames_total += n_frames
        if tracklet[0]['mode'] == 'easy':
            self.succ_easy.update(success, n=tracklet_length)
            self.prec_easy.update(precision, n=tracklet_length)
            self.n_frames_easy += n_frames
        elif tracklet[0]['mode'] == 'medium':
            self.succ_medium.update(success, n=tracklet_length)
            self.prec_medium.update(precision, n=tracklet_length)
            self.n_frames_medium += n_frames
        elif tracklet[0]['mode'] == 'hard':
            self.succ_hard.update(success, n=tracklet_length)
            self.prec_hard.update(precision, n=tracklet_length)
            self.n_frames_hard += n_frames

        self.txt_log.info('Total: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_total.avg, self.succ_total.avg,  self.n_frames_total))
        self.txt_log.info('easy: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_easy.avg, self.succ_easy.avg,  self.n_frames_easy))
        self.txt_log.info('medium: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_medium.avg, self.succ_medium.avg,  self.n_frames_medium))
        self.txt_log.info('hard: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_hard.avg, self.succ_hard.avg,  self.n_frames_hard))

    def _test_step_nuscenes_format(self, batch, batch_idx):
        # if batch_idx != 0:
        #     return
        tracklet = batch[0]
        if tracklet[0]['anno']['num_lidar_pts'] == 0:
            return
        n_frames = len(tracklet)
        self.n_frames_total += n_frames
        pred_bboxes, gt_bboxes = self.forward_on_tracklet(tracklet)
        overlaps, accuracies = [], []
        for i, (pred_bbox, gt_bbox) in enumerate(zip(pred_bboxes, gt_bboxes)):
            anno = tracklet[i]['anno']
            if anno['is_key_frame'] == 1:
                self.n_frames_key += 1
                overlap = estimateOverlap(
                    gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space, up_axis=self.cfg.dataset_cfg.up_axis)
                accuracy = estimateAccuracy(
                    gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space, up_axis=self.cfg.dataset_cfg.up_axis)
                overlaps.append(overlap)
                accuracies.append(accuracy)

        self.succ(torch.tensor(overlaps, device=self.device))
        self.prec(torch.tensor(accuracies, device=self.device))
        self.txt_log.info('Key: Prec=%.3f Succ=%.3f Key Frames=(%d/%d)' % (
            self.prec.compute(), self.succ.compute(), self.n_frames_key, self.n_frames_total))

    def test_step(self, batch, batch_idx):
        if 'KITTI' in self.cfg.dataset_cfg.dataset_type:
            return self._test_step_kitti_format(batch, batch_idx)
        elif 'Waymo' in self.cfg.dataset_cfg.dataset_type:
            return self._test_step_waymo_format(batch, batch_idx)
        elif 'NuScenes' in self.cfg.dataset_cfg.dataset_type:
            return self._test_step_nuscenes_format(batch, batch_idx)

    def _on_test_epoch_end_kitti_format(self):
        self.log('precesion', self.prec.compute(), prog_bar=True)
        self.log('success', self.succ.compute(), prog_bar=True)
        self.log('runtime', self.runtime.compute(), prog_bar=True)
        self.txt_log.info('Avg Prec/Succ=%.3f/%.3f Frames=%d Runtime=%.6f' % (
            self.prec.compute(), self.succ.compute(), self.n_frames.compute(), self.runtime.compute()))
        if self.cfg.save_test_result:
            self.pred_bboxes.sort(key=lambda x: x[0])
            data = []
            for idx, bbs in self.pred_bboxes:
                pred_bboxes = []
                for bb in bbs:
                    pred_bboxes.append(bb.encode())
                data.append(pred_bboxes)
            with open(osp.join(self.cfg.work_dir, 'result.json'), 'w') as f:
                json.dump(data, f)

    def _on_test_epoch_end_waymo_format(self):
        self.txt_log.info('============ Final ============')
        self.txt_log.info('Total: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_total.avg, self.succ_total.avg,  self.n_frames_total))
        self.txt_log.info('easy: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_easy.avg, self.succ_easy.avg,  self.n_frames_easy))
        self.txt_log.info('medium: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_medium.avg, self.succ_medium.avg,  self.n_frames_medium))
        self.txt_log.info('hard: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_hard.avg, self.succ_hard.avg,  self.n_frames_hard))

    def _on_test_epoch_end_nuscenes_format(self):
        self.txt_log.info('============ Final ============')
        self.txt_log.info('Key: Prec=%.3f Succ=%.3f Key Frames=(%d/%d)' % (
            self.prec.compute(), self.succ.compute(), self.n_frames_key, self.n_frames_total))

    def on_test_epoch_end(self):
        if 'KITTI' in self.cfg.dataset_cfg.dataset_type:
            return self._on_test_epoch_end_kitti_format()
        elif 'Waymo' in self.cfg.dataset_cfg.dataset_type:
            return self._on_test_epoch_end_waymo_format()
        elif 'NuScenes' in self.cfg.dataset_cfg.dataset_type:
            return self._on_test_epoch_end_nuscenes_format()
        