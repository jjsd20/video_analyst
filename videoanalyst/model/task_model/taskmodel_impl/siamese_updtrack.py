# -*- coding: utf-8 -*

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from videoanalyst.model.common_opr.common_block import (conv_bn_relu,
                                                        xcorr_depthwise)
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_model.taskmodel_base import (TRACK_TASKMODELS,
                                                          VOS_TASKMODELS)

torch.set_printoptions(precision=8)


class FeatureFusionModule(nn.Module):

    def __init__(self, channels, use_relu=True):
        super().__init__()
        # 通道对齐卷积（保持通道数不变）
        self.align_conv = nn.Conv2d(channels, channels, kernel_size=1)

        # 激活函数开关（与Metal版本ReLU对应）
        self.relu = nn.ReLU(inplace=True) if use_relu else None

    def forward(self, a, b):
        """
        参数:
        a: 基准特征 [B, C, H, W]
        b: 待融合特征 [B, C, H', W'] (H' < H, W' < W)

        返回:
        fused: 融合后特征 [B, C, H, W]
        """
        # 特征对齐（1x1卷积）
        aligned_a = self.align_conv(a)

        # 双线性插值上采样b到a的尺寸
        resized_b = F.interpolate(b,
                                  size=a.shape[-2:],
                                  mode='bilinear',
                                  align_corners=True)

        # 特征融合（逐元素相加）
        fused = aligned_a + resized_b

        # 可选激活函数
        if self.relu is not None:
            fused = self.relu(fused)

        return fused


# 添加通道注意力机制
class FeatureFusionWithAttention(FeatureFusionModule):

    def __init__(self, channels):
        super().__init__(channels)
        # 通道注意力模块
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                       nn.Conv2d(channels, channels // 16, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(channels // 16, channels, 1),
                                       nn.Sigmoid())

    def forward(self, a, b):
        aligned_a = self.align_conv(a)
        resized_b = F.interpolate(b,
                                  a.shape[-2:],
                                  mode='bilinear',
                                  align_corners=True)

        # 生成注意力权重
        att = self.attention(aligned_a + resized_b)

        # 加权融合
        return att * aligned_a + (1 - att) * resized_b


@TRACK_TASKMODELS.register
@VOS_TASKMODELS.register
class SiamUpdTrack(ModuleBase):
    r"""
    SiamTrack model for tracking

    Hyper-Parameters
    ----------------
    pretrain_model_path: string
        path to parameter to be loaded into module
    head_width: int
        feature width in head structure
    """

    default_hyper_params = dict(pretrain_model_path="",
                                head_width=256,
                                conv_weight_std=0.01,
                                neck_conv_bias=[True, True, True, True],
                                corr_fea_output=False,
                                trt_mode=False,
                                trt_fea_model_path="",
                                trt_track_model_path="",
                                amp=False)

    support_phases = ["train", "feature", "track", "freeze_track_fea"]

    def __init__(self, backbone, head, loss=None):
        super(SiamUpdTrack, self).__init__()
        self.basemodel = backbone
        self.head = head
        self.loss = loss
        self.trt_fea_model = None
        self.trt_track_model = None
        self._phase = "train"
        self.fusion = FeatureFusionWithAttention(256)

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, p):
        assert p in self.support_phases
        self._phase = p

    def train_forward(self, training_data):
        target_img = training_data["im_z"]  #32*3*127*127
        search_img = training_data["im_x"]  #32*3*303*303
        # backbone feature
        f_z = self.basemodel(target_img)  #32*256*6*6
        f_x = self.basemodel(search_img)  #32*256*28*28
        # feature adjustment
        c_z_k = self.c_z_k(f_z)  #32*256*4*4
        r_z_k = self.r_z_k(f_z)  #32*256*4*4
        c_x = self.c_x(f_x)  #32*256*26*26
        r_x = self.r_x(f_x)  #32*256*26*26
        # update template
        c_z_k = self.fusion(c_z_k, c_x)
        # feature matching
        c_out = xcorr_depthwise(c_x, c_z_k)  #32*256*23*23
        r_out = xcorr_depthwise(r_x, r_z_k)  # 32*256*23*23

        # head
        fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(
            c_out, r_out)
        predict_data = dict(
            cls_pred=fcos_cls_score_final,  #32*289*1
            ctr_pred=fcos_ctr_score_final,  #32*289*1
            box_pred=fcos_bbox_final,  #32*289*4
        )
        if self._hyper_params["corr_fea_output"]:
            predict_data["corr_fea"] = corr_fea  #32*256*17*17
        return predict_data

    def instance(self, img):
        f_z = self.basemodel(img)
        # template as kernel
        c_x = self.c_x(f_z)
        self.cf = c_x

    def forward(self, *args, phase=None):
        r"""
        Perform tracking process for different phases (e.g. train / init / track)

        Arguments
        ---------
        target_img: torch.Tensor
            target template image patch
        search_img: torch.Tensor
            search region image patch

        Returns
        -------
        fcos_score_final: torch.Tensor
            predicted score for bboxes, shape=(B, HW, 1)
        fcos_bbox_final: torch.Tensor
            predicted bbox in the crop, shape=(B, HW, 4)
        fcos_cls_prob_final: torch.Tensor
            classification score, shape=(B, HW, 1)
        fcos_ctr_prob_final: torch.Tensor
            center-ness score, shape=(B, HW, 1)
        """
        if phase is None:
            phase = self._phase
        # used during training
        if phase == 'train':
            # resolve training data
            if self._hyper_params["amp"]:
                with torch.cuda.amp.autocast():
                    return self.train_forward(args[0])
            else:
                return self.train_forward(args[0])

        # used for template feature extraction (normal mode)
        elif phase == 'feature':
            target_img, = args
            if self._hyper_params["trt_mode"]:
                # extract feature with trt model
                out_list = self.trt_fea_model(target_img)
            else:
                # backbone feature
                f_z = self.basemodel(target_img)
                # template as kernel
                c_z_k = self.c_z_k(f_z)
                r_z_k = self.r_z_k(f_z)
                # output
                out_list = [c_z_k, r_z_k]
        # used for template feature extraction (trt mode)
        elif phase == "freeze_track_fea":
            search_img, = args
            # backbone feature
            f_x = self.basemodel(search_img)
            # feature adjustment
            c_x = self.c_x(f_x)
            r_x = self.r_x(f_x)
            # head
            return [c_x, r_x]
        # [Broken] used for template feature extraction (trt mode)
        #   currently broken due to following issue of "torch2trt" package
        #   c.f. https://github.com/NVIDIA-AI-IOT/torch2trt/issues/251
        elif phase == "freeze_track_head":
            c_out, r_out = args
            # head
            outputs = self.head(c_out, r_out, 0, True)
            return outputs
        # used for tracking one frame during test
        elif phase == 'track':
            if len(args) == 3:
                search_img, c_z_k, r_z_k = args
                if self._hyper_params["trt_mode"]:
                    c_x, r_x = self.trt_track_model(search_img)
                else:
                    # backbone feature
                    f_x = self.basemodel(search_img)
                    # feature adjustment
                    c_x = self.c_x(f_x)
                    r_x = self.r_x(f_x)
            elif len(args) == 4:
                # c_x, r_x already computed
                c_z_k, r_z_k, c_x, r_x = args
            else:
                raise ValueError("Illegal args length: %d" % len(args))

            # feature matching
            r_out = xcorr_depthwise(r_x, r_z_k)
            c_out = xcorr_depthwise(c_x, c_z_k)

            # update template
            c_z_k = self.fusion(c_z_k, c_out)

            # head
            fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(
                c_out, r_out, search_img.size(-1))
            # apply sigmoid
            fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
            fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)
            # apply centerness correction
            fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final
            # register extra output
            extra = dict(c_x=c_x, r_x=r_x, corr_fea=corr_fea)
            self.cf = c_x
            # output
            out_list = fcos_score_final, fcos_bbox_final, fcos_cls_prob_final, fcos_ctr_prob_final, extra
        else:
            raise ValueError("Phase non-implemented.")

        return out_list

    def update_params(self):
        r"""
        Load model parameters
        """
        self._make_convs()
        self._initialize_conv()
        super().update_params()
        if self._hyper_params["trt_mode"]:
            logger.info("trt mode enable")
            from torch2trt import TRTModule
            self.trt_fea_model = TRTModule()
            self.trt_fea_model.load_state_dict(
                torch.load(self._hyper_params["trt_fea_model_path"]))
            self.trt_track_model = TRTModule()
            self.trt_track_model.load_state_dict(
                torch.load(self._hyper_params["trt_track_model_path"]))
            logger.info("loading trt model succefully")

    def _make_convs(self):
        head_width = self._hyper_params['head_width']

        # feature adjustment
        self.r_z_k = conv_bn_relu(head_width,
                                  head_width,
                                  1,
                                  3,
                                  0,
                                  has_relu=False)
        self.c_z_k = conv_bn_relu(head_width,
                                  head_width,
                                  1,
                                  3,
                                  0,
                                  has_relu=False)
        self.r_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.c_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)

    def _initialize_conv(self, ):
        conv_weight_std = self._hyper_params['conv_weight_std']
        conv_list = [
            self.r_z_k.conv, self.c_z_k.conv, self.r_x.conv, self.c_x.conv
        ]
        for ith in range(len(conv_list)):
            conv = conv_list[ith]
            torch.nn.init.normal_(conv.weight,
                                  std=conv_weight_std)  # conv_weight_std=0.01

    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)
