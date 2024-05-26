# # Copyright (c) Lixin YANG. All Rights Reserved.
# r"""
# Networks for heatmap estimation from RGB images using Hourglass Network
# "Stacked Hourglass Networks for Human Pose Estimation", Alejandro Newell, Kaiyu Yang, Jia Deng, ECCV 2016
# """
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from termcolor import colored, cprint

# from bihand.models.bases.bottleneck import BottleneckBlock
# from bihand.models.bases.hourglass import HourglassBisected
# from bihand.models.bases.hourglass import NetStackedHourglass
# from bihand.models.bases.hourglass import Hourglass


# class SeedNet(nn.Module):
#     def __init__(
#         self,
#         nstacks=2,
#         nblocks=1,
#         njoints=21,
#         block=BottleneckBlock,
#     ):
#         super(SeedNet, self).__init__()
#         self.njoints  = njoints
#         self.nstacks  = nstacks
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=True)
#         self.bn1 = nn.BatchNorm2d(self.in_planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(2, stride=2)#如果输入的张量具有形状[N, C, W_in, H_in]，则输出的张量将具有形状[N, C, W_out, H_out]。W_out = (W_in - 2) / 2 + 1；H_out = (H_in - 2) / 2 + 1
#         self.layer1 = self._make_residual(block, nblocks, self.in_planes, 2*self.in_planes)
#         # current self.in_planes is 64 * 2 = 128
#         self.layer2 = self._make_residual(block, nblocks, self.in_planes, 2*self.in_planes)
#         # current self.in_planes is 128 * 2 = 256

#         self.layer3 = self._make_residual(block, nblocks, self.in_planes, self.in_planes)
        
#         ch = self.in_planes # 256

#         hg2b, res1, res2, fc1, _fc1, fc2, _fc2= [],[],[],[],[],[],[]
#         hm, _hm, mask, _mask = [], [], [], []
#         for i in range(nstacks): # 2
#             hg2b.append(HourglassBisected(block, nblocks, ch, depth=4))
#             # hg2b.append(NetStackedHourglass(nstacks,nblocks,nclasses=21,block=BottleneckBlock))
#             res1.append(self._make_residual(block, nblocks, ch, ch))
#             res2.append(self._make_residual(block, nblocks, ch, ch))
#             fc1.append(self._make_fc(ch, ch))
#             fc2.append(self._make_fc(ch, ch))
#             hm.append(nn.Conv2d(ch, njoints, kernel_size=1, bias=True))
#             mask.append(nn.Conv2d(ch, 1, kernel_size=1, bias=True))

#             if i < nstacks-1:
#                 _fc1.append(nn.Conv2d(ch, ch, kernel_size=1, bias=False))
#                 _fc2.append(nn.Conv2d(ch, ch, kernel_size=1, bias=False))
#                 _hm.append(nn.Conv2d(njoints, ch, kernel_size=1, bias=False))
#                 _mask.append(nn.Conv2d(1, ch, kernel_size=1, bias=False))

#         self.hg2b  = nn.ModuleList(hg2b) # hgs: hourglass stack
#         self.res1  = nn.ModuleList(res1)
#         self.fc1   = nn.ModuleList(fc1)
#         self._fc1  = nn.ModuleList(_fc1)
#         self.res2  = nn.ModuleList(res2)
#         self.fc2   = nn.ModuleList(fc2)
#         self._fc2  = nn.ModuleList(_fc2)
#         self.hm   = nn.ModuleList(hm)
#         self._hm  = nn.ModuleList(_hm)
#         self.mask  = nn.ModuleList(mask)
#         self._mask = nn.ModuleList(_mask)


#     def _make_fc(self, in_planes, out_planes):
#         bn = nn.BatchNorm2d(in_planes)
#         conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
#         return nn.Sequential(conv, bn, self.relu)


#     def _make_residual(self, block, nblocks, in_planes, out_planes):
#         layers = []
#         layers.append( block( in_planes, out_planes) )
#         self.in_planes = out_planes
#         for i in range(1, nblocks):
#             layers.append(block( self.in_planes, out_planes))
#         return nn.Sequential(*layers)

#     def forward(self, x):

#         l_hm, l_mask, l_enc = [], [], []
#         #x :torch.Size([16, 3, 256, 256])    N=batch/4
#         x = self.conv1(x) # x: (N,64,128,128)
#         x = self.bn1(x)
#         x = self.relu(x)

#         x = self.layer1(x)
#         x = self.maxpool(x) # x: (N,128,64,64)
#         x = self.layer2(x)
#         x = self.layer3(x)

#         for i in range(self.nstacks): #2

#             y_1, y_2, _ = self.hg2b[i](x)
#             ##x     torch.Size([16, 256, 64, 64])
#             ##y_1   torch.Size([16, 256, 64, 64])
#             ##y_2   torch.Size([16, 256, 64, 64])

#             y_1 = self.res1[i](y_1)
#             y_1 = self.fc1[i](y_1)
#             est_hm = self.hm[i](y_1)
#             # est_hm    torch.Size([16, 21, 64, 64])     
#             l_hm.append(est_hm)

#             y_2 = self.res2[i](y_2)
#             y_2 = self.fc2[i](y_2)
#             est_mask = self.mask[i](y_2)
#             #est_mask       torch.Size([16, 1, 64, 64])
#             l_mask.append(est_mask)

#             if i < self.nstacks-1:
#                 _fc1 = self._fc1[i](y_1)
#                 _hm = self._hm[i](est_hm)
#                 _fc2 = self._fc2[i](y_2)
#                 _mask = self._mask[i](est_mask)
#                 x = x + _fc1 + _fc2 + _hm + _mask
#                 l_enc.append(x)
#             else:
#                 l_enc.append(x + y_1 + y_2)
#         assert len(l_hm) == self.nstacks
#         #len(l_hm)) len(l_mask))  2
#         return l_hm, l_mask, l_enc

# class SeedNet2(nn.Module):
#     def __init__(
#         self,
#         nstacks=2,
#         nblocks=1,
#         njoints=21,
#         block=BottleneckBlock,
#     ):
#         super(SeedNet2, self).__init__()
#         self.njoints  = njoints
#         self.nstacks  = nstacks
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=True)
#         self.bn1 = nn.BatchNorm2d(self.in_planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(2, stride=2)#如果输入的张量具有形状[N, C, W_in, H_in]，则输出的张量将具有形状[N, C, W_out, H_out]。W_out = (W_in - 2) / 2 + 1；H_out = (H_in - 2) / 2 + 1
#         self.layer1 = self._make_residual(block, nblocks, self.in_planes, 2*self.in_planes)
#         # current self.in_planes is 64 * 2 = 128
#         self.layer2 = self._make_residual(block, nblocks, self.in_planes, 2*self.in_planes)
#         # current self.in_planes is 128 * 2 = 256
#         self.layer3 = self._make_residual(block, nblocks, self.in_planes, self.in_planes)
        
#         ch = self.in_planes # 256

#         hgs, res, fc, _fc, score, _score = [], [], [], [], [], []
#         for i in range(nstacks):  # stacking the hourglass
#             hgs.append(Hourglass(block, nblocks, ch, depth=4))
#             res.append(self._make_residual(block, nblocks, ch, ch))
#             fc.append(self._make_residual(block, nblocks, ch, ch))
#             score.append(nn.Conv2d(ch, njoints, kernel_size=1, bias=True))

#             if i < nstacks - 1:
#                 _fc.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
#                 _score.append(nn.Conv2d(njoints, ch, kernel_size=1, bias=True))

#         self.hgs = nn.ModuleList(hgs)  # hgs: hourglass stack
#         self.res = nn.ModuleList(res)
#         self.fc = nn.ModuleList(fc)  ### change back to use the pre-trainded
#         self._fc = nn.ModuleList(_fc)
#         self.score = nn.ModuleList(score)
#         self._score = nn.ModuleList(_score)


#     def _make_residual(self, block, nblocks, in_planes, out_planes):
#         layers = []
#         layers.append(block(in_planes, out_planes))
#         self.in_planes = out_planes
#         for i in range(1, nblocks):
#             layers.append(block(self.in_planes, out_planes))
#         return nn.Sequential(*layers)

#     def forward(self, x):

#         out = []
#         hm_enc = []  # heatmaps encoding
#         # x: (N,3,256,256)
#         x = self.conv1(x)  # x: (N,64,128,128)
#         x = self.bn1(x)
#         x = self.relu(x)

#         x = self.layer1(x)  # x: (N,128,128,128)
#         x = self.maxpool(x)  # x: (N,128,64,64)
#         x = self.layer2(x)  # x: (N,256,64,64)
#         x = self.layer3(x)  # x: (N,256,64,64)
#         hm_enc.append(x)

#         for i in range(self.nstacks):
#             y = self.hgs[i](x)
#             y = self.res[i](y)
#             y = self.fc[i](y)
#             score = self.score[i](y)
#             out.append(score)
#             if i < self.nstacks - 1:
#                 _fc = self._fc[i](y)
#                 _score = self._score[i](score)
#                 x = x + _fc + _score
#                 hm_enc.append(x)
#             else:
#                 hm_enc.append(y)
#         return out, hm_enc

# class SeedNet3(nn.Module):
#     def __init__(self, block, layers, cfg, **kwargs):
#         super(SeedNet, self).__init__()
#         self.njoints  = njoints
#         self.nstacks  = nstacks
#         self.inplanes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
#         self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

#         # used for deconv layers
#         self.deconv_layers = self._make_deconv_layer(
#             extra.NUM_DECONV_LAYERS,
#             extra.NUM_DECONV_FILTERS,
#             extra.NUM_DECONV_KERNELS,
#         )

#         self.final_layer = nn.Conv2d(
#             in_channels=extra.NUM_DECONV_FILTERS[-1],
#             out_channels=cfg.MODEL.NUM_JOINTS,
#             kernel_size=extra.FINAL_CONV_KERNEL,
#             stride=1,
#             padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
#         )


#     def _make_layer(self, block, planes, blocks, stride=1):#planes: 表示当前层的输出通道数；stride: 表示当前层的步幅。
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(#首先根据步幅和输入输出通道数计算出一个下采样的模块downsample，用于将特征图的大小降低一半
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
#             )
#         #使用一个列表layers来存储构建的残差块。首先加入一个带有下采样模块的残差块，其余的残差块不需要进行下采样操作
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def _get_deconv_cfg(self, deconv_kernel, index):
#         if deconv_kernel == 4:
#             padding = 1
#             output_padding = 0
#         elif deconv_kernel == 3:
#             padding = 1
#             output_padding = 1
#         elif deconv_kernel == 2:
#             padding = 0
#             output_padding = 0

#         return deconv_kernel, padding, output_padding

#     def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
#         #num_layers表示反卷积层的数量;num_filters是一个列表，表示每个反卷积层的输出通道数；num_kernels也是一个列表，表示每个反卷积层的卷积核大小。
#         assert num_layers == len(num_filters), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
#         assert num_layers == len(num_kernels), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'

#         layers = []
#         for i in range(num_layers):#通过循环迭代创建反卷积层
#             kernel, padding, output_padding = \
#                 self._get_deconv_cfg(num_kernels[i], i)

#             planes = num_filters[i]
#             layers.append(
#                 nn.ConvTranspose2d(
#                     in_channels=self.inplanes,
#                     out_channels=planes,
#                     kernel_size=kernel,
#                     stride=2,
#                     padding=padding,
#                     output_padding=output_padding,
#                     bias=self.deconv_with_bias))
#             layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
#             layers.append(nn.ReLU(inplace=True))
#             self.inplanes = planes

#         return nn.Sequential(*layers)


#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.deconv_layers(x)
#         x = self.final_layer(x)

#         return x

    
# Copyright (c) Lixin YANG. All Rights Reserved.
r"""
Networks for heatmap estimation from RGB images using Hourglass Network
"Stacked Hourglass Networks for Human Pose Estimation", Alejandro Newell, Kaiyu Yang, Jia Deng, ECCV 2016
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored, cprint

from bihand.models.bases.bottleneck import BottleneckBlock
from bihand.models.bases.hourglass import HourglassBisected


class SeedNet(nn.Module):
    def __init__(
        self,
        nstacks=2,
        nblocks=1,
        njoints=21,
        block=BottleneckBlock,
    ):
        super(SeedNet, self).__init__()
        self.njoints  = njoints
        self.nstacks  = nstacks
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.layer1 = self._make_residual(block, nblocks, self.in_planes, 2*self.in_planes)
        # current self.in_planes is 64 * 2 = 128
        self.layer2 = self._make_residual(block, nblocks, self.in_planes, 2*self.in_planes)
        # current self.in_planes is 128 * 2 = 256
        self.layer3 = self._make_residual(block, nblocks, self.in_planes, self.in_planes)

        ch = self.in_planes # 256

        hg2b, res1, res2, fc1, _fc1, fc2, _fc2= [],[],[],[],[],[],[]
        hm, _hm, mask, _mask = [], [], [], []
        for i in range(nstacks): # 2
            hg2b.append(HourglassBisected(block, nblocks, ch, depth=4))
            res1.append(self._make_residual(block, nblocks, ch, ch))
            res2.append(self._make_residual(block, nblocks, ch, ch))
            fc1.append(self._make_fc(ch, ch))
            fc2.append(self._make_fc(ch, ch))
            hm.append(nn.Conv2d(ch, njoints, kernel_size=1, bias=True))
            mask.append(nn.Conv2d(ch, 1, kernel_size=1, bias=True))

            if i < nstacks-1:
                _fc1.append(nn.Conv2d(ch, ch, kernel_size=1, bias=False))
                _fc2.append(nn.Conv2d(ch, ch, kernel_size=1, bias=False))
                _hm.append(nn.Conv2d(njoints, ch, kernel_size=1, bias=False))
                _mask.append(nn.Conv2d(1, ch, kernel_size=1, bias=False))

        self.hg2b  = nn.ModuleList(hg2b) # hgs: hourglass stack
        self.res1  = nn.ModuleList(res1)
        self.fc1   = nn.ModuleList(fc1)
        self._fc1  = nn.ModuleList(_fc1)
        self.res2  = nn.ModuleList(res2)
        self.fc2   = nn.ModuleList(fc2)
        self._fc2  = nn.ModuleList(_fc2)
        self.hm   = nn.ModuleList(hm)
        self._hm  = nn.ModuleList(_hm)
        self.mask  = nn.ModuleList(mask)
        self._mask = nn.ModuleList(_mask)


    def _make_fc(self, in_planes, out_planes):
        bn = nn.BatchNorm2d(in_planes)
        conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        return nn.Sequential(conv, bn, self.relu)


    def _make_residual(self, block, nblocks, in_planes, out_planes):
        layers = []
        layers.append( block( in_planes, out_planes) )
        self.in_planes = out_planes
        for i in range(1, nblocks):
            layers.append(block( self.in_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        l_hm, l_mask, l_enc = [], [], []
        x = self.conv1(x) # x: (N,64,128,128)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x) # x: (N,128,64,64)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.nstacks): #2
            y_1, y_2, _ = self.hg2b[i](x)

            y_1 = self.res1[i](y_1)
            y_1 = self.fc1[i](y_1)
            est_hm = self.hm[i](y_1)
            l_hm.append(est_hm)

            y_2 = self.res2[i](y_2)
            y_2 = self.fc2[i](y_2)
            est_mask = self.mask[i](y_2)
            l_mask.append(est_mask)

            if i < self.nstacks-1:
                _fc1 = self._fc1[i](y_1)
                _hm = self._hm[i](est_hm)
                _fc2 = self._fc2[i](y_2)
                _mask = self._mask[i](est_mask)
                x = x + _fc1 + _fc2 + _hm + _mask
                l_enc.append(x)
            else:
                l_enc.append(x + y_1 + y_2)
        assert len(l_hm) == self.nstacks
        # print("111",len(l_hm),len(l_mask),len(l_enc)) #2 2 2
        return l_hm, l_mask, l_enc