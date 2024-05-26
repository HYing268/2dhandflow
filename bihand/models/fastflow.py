# import FrEIA.framework as Ff
# import FrEIA.modules as Fm
# import timm
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import constants as const
# from bihand.models.bases.bottleneck import BottleneckBlock

# def subnet_conv_func(kernel_size, hidden_ratio):
#     def subnet_conv(in_channels, out_channels):
#         hidden_channels = int(in_channels * hidden_ratio)
#         return nn.Sequential(
#             nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
#             nn.ReLU(),
#             nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
#         )

#     return subnet_conv


# def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
#     nodes = Ff.SequenceINN(*input_chw)
#     for i in range(flow_steps):
#         if i % 2 == 1 and not conv3x3_only:
#             kernel_size = 1
#         else:
#             kernel_size = 3
#         nodes.append(
#             Fm.AllInOneBlock,
#             subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
#             affine_clamping=clamp,
#             permute_soft=False,
#         )
#     return nodes


# class FastFlow(nn.Module):
#     def __init__(
#         self,
#         backbone_name,
#         flow_steps,
#         input_size,
#         conv3x3_only=False,
#         hidden_ratio=1.0,
#     ):
#         super(FastFlow, self).__init__()
#         assert (
#             backbone_name in const.SUPPORTED_BACKBONES
#         ), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)

#         if backbone_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
#             self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
#             channels = [768]
#             scales = [16]
#         else:
#             self.feature_extractor = timm.create_model(
#                 backbone_name,
#                 pretrained=True,
#                 features_only=True,
#                 out_indices=[1, 2, 3],
#             )
#             channels = self.feature_extractor.feature_info.channels()
#             scales = self.feature_extractor.feature_info.reduction()

#             # for transformers, use their pretrained norm w/o grad
#             # for resnets, self.norms are trainable LayerNorm
#             self.norms = nn.ModuleList()
#             for in_channels, scale in zip(channels, scales):
#                 self.norms.append(
#                     nn.LayerNorm(
#                         [in_channels, int(input_size / scale), int(input_size / scale)],
#                         elementwise_affine=True,
#                     )
#                 )

#         for param in self.feature_extractor.parameters():
#             param.requires_grad = False

#         self.nf_flows = nn.ModuleList()
#         for in_channels, scale in zip(channels, scales):
#             self.nf_flows.append(
#                 nf_fast_flow(
#                     [in_channels, int(input_size / scale), int(input_size / scale)],
#                     conv3x3_only=conv3x3_only,
#                     hidden_ratio=hidden_ratio,
#                     flow_steps=flow_steps,
#                 )
#             )

#         # self.nf_flows2 = nn.ModuleList()
#         # for in_channels, scale in zip(channels, scales):
#         #     self.nf_flows2.append(
#         #         nf_fast_flow(
#         #             [in_channels, int(input_size / scale), int(input_size / scale)],
#         #             conv3x3_only=conv3x3_only,
#         #             hidden_ratio=hidden_ratio,
#         #             flow_steps=flow_steps,
#         #         )
#         #     )

#         self.input_size = input_size

#         self.con = nn.Conv2d(256, 256, kernel_size=3, stride=4, padding=1)
#         self.res = BottleneckBlock(256,256)
#         self.conv = nn.Conv2d(256,256, kernel_size=1, bias=False)
#         self.bn = nn.BatchNorm2d(256)
#         self.relu = nn.ReLU(inplace=True)


#         # self.con2 = nn.Conv2d(256, 256, kernel_size=3, stride=4, padding=1)
#         # self.res2 = BottleneckBlock(256,256)
#         # self.conv2 = nn.Conv2d(256,256, kernel_size=1, bias=False)
#         # self.bn2 = nn.BatchNorm2d(256)
#         # self.relu2 = nn.ReLU(inplace=True)

#         self.hm = nn.Conv2d(256, 21, kernel_size=1, bias=True)
#         self.mask = nn.Conv2d(256, 1, kernel_size=1, bias=True)

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
#         self.feature_extractor.eval()
#         if isinstance(
#             self.feature_extractor, timm.models.vision_transformer.VisionTransformer
#         ):
#             x = self.feature_extractor.patch_embed(x)
#             cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
#             if self.feature_extractor.dist_token is None:
#                 x = torch.cat((cls_token, x), dim=1)
#             else:
#                 x = torch.cat(
#                     (
#                         cls_token,
#                         self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
#                         x,
#                     ),
#                     dim=1,
#                 )
#             x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
#             for i in range(8):  # paper Table 6. Block Index = 7
#                 x = self.feature_extractor.blocks[i](x)
#             x = self.feature_extractor.norm(x)
#             x = x[:, 2:, :]
#             N, _, C = x.shape
#             x = x.permute(0, 2, 1)
#             x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
#             features = [x]
#         elif isinstance(self.feature_extractor, timm.models.cait.Cait):
#             x = self.feature_extractor.patch_embed(x)
#             x = x + self.feature_extractor.pos_embed
#             x = self.feature_extractor.pos_drop(x)
#             for i in range(41):  # paper Table 6. Block Index = 40
#                 x = self.feature_extractor.blocks[i](x)
#             N, _, C = x.shape
#             x = self.feature_extractor.norm(x)
#             x = x.permute(0, 2, 1)
#             x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
#             features = [x]
#         else:
#             features = self.feature_extractor(x)
#             features = [self.norms[i](feature) for i, feature in enumerate(features)]
#             #features[0]   torch.Size([Batchsize/gpus, 64, 64, 64])   len(features)   3

#         loss = 0
#         outputs = []
#         hm_outs = []
#         mask_outs = []
#         for i, feature in enumerate(features):
#             output, log_jac_dets = self.nf_flows[i](feature)
#             #output     torch.Size([B/gpus, 256, 16, 16])
#             #log_jac_dets     torch.Size([B/gpus])
#             loss += torch.mean(
#                 0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
#             )

#             # output2, log_jac_dets2 = self.nf_flows2[i](feature)
#             # loss += torch.mean(
#             #     0.5 * torch.sum(output2**2, dim=(1, 2, 3)) - log_jac_dets2
#             # )

#         output = self.con(output)
#         output = nn.functional.interpolate(output, size=[64, 64], mode='bilinear', align_corners=False)
        
#         output = self.res(output)
#         output = self.conv(output)
#         output = self.bn(output)
#         output = self.relu(output)
        

#         # output2 = self.con(output2)
#         # output2 = nn.functional.interpolate(output2, size=[64, 64], mode='bilinear', align_corners=False)
        
#         # output2 = self.res(output2)
#         # output2 = self.conv(output2)
#         # output2 = self.bn(output2)
#         # output2 = self.relu(output2)



#         hm_out = self.hm(output) # torch.Size([16, 21, 64, 64])
#         mask_out = self.mask(output) #torch.Size([16, 1, 64, 64])

#         # mask_out = self.mask(output2)

#         hm_outs.append(hm_out)
#         mask_outs.append(mask_out)
#         outputs.append(output)

        
#         '''
#         if not self.training:
#             anomaly_map_list = []
#             for output in outputs:
#                 log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
#                 prob = torch.exp(log_prob)
#                 a_map = F.interpolate(
#                     -prob,
#                     size=[self.input_size, self.input_size],
#                     mode="bilinear",
#                     align_corners=False,
#                 )
#                 anomaly_map_list.append(a_map)
#             anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
#             anomaly_map = torch.mean(anomaly_map_list, dim=-1)
#             ret["anomaly_map"] = anomaly_map
#         '''

#         return loss,hm_outs,mask_outs




# def subnet_conv_func(kernel_size, hidden_ratio):
#     def subnet_conv(in_channels, out_channels):
#         hidden_channels = int(in_channels * hidden_ratio)
#         return nn.Sequential(
#             nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
#             nn.ReLU(),
#             nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
#         )

#     return subnet_conv


# def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
#     nodes = Ff.SequenceINN(*input_chw)
#     for i in range(flow_steps):
#         if i % 2 == 1 and not conv3x3_only:
#             kernel_size = 1
#         else:
#             kernel_size = 3
#         nodes.append(
#             Fm.AllInOneBlock,
#             subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
#             affine_clamping=clamp,
#             permute_soft=False,
#         )
#     return nodes


# class FastFlow(nn.Module):
#     def __init__(
#         self,
#         backbone_name,
#         flow_steps,
#         input_size,
#         conv3x3_only=False,
#         hidden_ratio=1.0,
#     ):
#         super(FastFlow, self).__init__()
#         assert (
#             backbone_name in const.SUPPORTED_BACKBONES
#         ), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)

#         if backbone_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
#             self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
#             channels = [768]
#             scales = [16]
#         else:
#             self.feature_extractor = timm.create_model(
#                 backbone_name,
#                 pretrained=True,
#                 features_only=True,
#                 out_indices=[1, 2, 3],
#             )
#             channels = self.feature_extractor.feature_info.channels()
#             scales = self.feature_extractor.feature_info.reduction()

#             # for transformers, use their pretrained norm w/o grad
#             # for resnets, self.norms are trainable LayerNorm
#             self.norms = nn.ModuleList()
#             for in_channels, scale in zip(channels, scales):
#                 self.norms.append(
#                     nn.LayerNorm(
#                         [in_channels, int(input_size / scale), int(input_size / scale)],
#                         elementwise_affine=True,
#                     )
#                 )

#         for param in self.feature_extractor.parameters():
#             param.requires_grad = False

#         self.nf_flows = nn.ModuleList()
#         for in_channels, scale in zip(channels, scales):
#             self.nf_flows.append(
#                 nf_fast_flow(
#                     [in_channels, int(input_size / scale), int(input_size / scale)],
#                     conv3x3_only=conv3x3_only,
#                     hidden_ratio=hidden_ratio,
#                     flow_steps=flow_steps,
#                 )
#             )

#         self.input_size = input_size

#         self.con = nn.Conv2d(256, 256, kernel_size=3, stride=4, padding=1)
#         self.res = BottleneckBlock(256,256)
#         self.conv = nn.Conv2d(256,256, kernel_size=1, bias=False)
#         self.bn = nn.BatchNorm2d(256)
#         self.relu = nn.ReLU(inplace=True)

#         self.hm = nn.Conv2d(256, 21, kernel_size=1, bias=True)
#         self.mask = nn.Conv2d(256, 1, kernel_size=1, bias=True)

#         #cait
#         self.model_conv = nn.Sequential(
#             nn.Conv2d(768, 256, kernel_size=5, stride=2, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=7, stride=5)
#         )

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
#         self.feature_extractor.eval()
#         if isinstance(
#             self.feature_extractor, timm.models.vision_transformer.VisionTransformer
#         ):
#             x = self.feature_extractor.patch_embed(x)
#             cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
#             if self.feature_extractor.dist_token is None:
#                 x = torch.cat((cls_token, x), dim=1)
#             else:
#                 x = torch.cat(
#                     (
#                         cls_token,
#                         self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
#                         x,
#                     ),
#                     dim=1,
#                 )
#             x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
#             for i in range(8):  # paper Table 6. Block Index = 7
#                 x = self.feature_extractor.blocks[i](x)
#             x = self.feature_extractor.norm(x)
#             x = x[:, 2:, :]
#             N, _, C = x.shape
#             x = x.permute(0, 2, 1)
#             x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
#             features = [x]
#         elif isinstance(self.feature_extractor, timm.models.cait.Cait):
#             # print(x.shape,"11111111111")[64, 3, 256, 256]
#             x = nn.functional.interpolate(x, size=(448, 448), mode='bilinear')
#             x = self.feature_extractor.patch_embed(x)
#             x = x + self.feature_extractor.pos_embed
#             x = self.feature_extractor.pos_drop(x)
#             for i in range(41):  # paper Table 6. Block Index = 40
#                 x = self.feature_extractor.blocks[i](x)
#             N, _, C = x.shape
#             x = self.feature_extractor.norm(x)
#             x = x.permute(0, 2, 1)
#             x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
#             features = [x]
#         else:
#             features = self.feature_extractor(x)
#             features = [self.norms[i](feature) for i, feature in enumerate(features)]
#             #features[0]   torch.Size([Batchsize/gpus, 64, 64, 64])   len(features)   3

#         loss = 0
#         outputs = []
#         hm_outs = []
#         mask_outs = []
#         for i, feature in enumerate(features):
#             output, log_jac_dets = self.nf_flows[i](feature)
#             loss += torch.mean(
#                 0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
#             )

# #resnet18
#         #output    torch.Size([B/gpus, 256, 16, 16])  log_jac_dets     torch.Size([B/gpus])
            
# #cait_448
#         # output ([64, 768, 28, 28])
            
#         # output = self.model_conv(output)


# #common
#         print("1",output.size())
#         output = self.con(output) #torch.Size([64, 256, 4, 4])
#         output = nn.functional.interpolate(output, size=[64, 64], mode='bilinear', align_corners=False) #torch.Size([64, 256, 64, 64])
#         print("2",output.size())
#         output = self.res(output)
#         output = self.conv(output)
#         output = self.bn(output)
#         output = self.relu(output)


#         print("3",output.size())
#         hm_out = self.hm(output) # torch.Size([16, 21, 64, 64])
#         mask_out = self.mask(output) #torch.Size([16, 1, 64, 64])

#         hm_outs.append(hm_out)
#         mask_outs.append(mask_out)
#         outputs.append(output)

        
#         return loss,hm_outs,mask_outs
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import constants as const
from bihand.models.bases.bottleneck import BottleneckBlock
from PIL import Image
import numpy as np

def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        # padding = 2 * (kernel_size // 2 - ((1 + kernel_size) % 2), kernel_size // 2)
        return nn.Sequential(
            # nn.ZeroPad2d(padding),
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            # nn.ZeroPad2d(padding),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv


def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3 #3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class FastFlow(nn.Module):
    def __init__(
        self,
        backbone_name,
        flow_steps,
        input_size,
        conv3x3_only=False,
        hidden_ratio=1.0,
        resorcait=True
    ):
        super(FastFlow, self).__init__()
        assert ( backbone_name in const.SUPPORTED_BACKBONES), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)
        if backbone_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
            self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
            channels = [768]
            scales = [16]
        else:
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3],
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # for transformers, use their pretrained norm w/o grad
            # for resnets, self.norms are trainable LayerNorm
            self.norms = nn.ModuleList()
            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [in_channels, int(input_size / scale), int(input_size / scale)],
                        elementwise_affine=True,
                    )
                )

        for param in self.feature_extractor.parameters():
            param.requires_grad = True #fixme

        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, int(input_size / scale), int(input_size / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )

        self.input_size = input_size

        # self.con = nn.Conv2d(256, 256, kernel_size=3, stride=4, padding=1)
        # self.res = BottleneckBlock(256,256)
        # self.conv = nn.Conv2d(256,256, kernel_size=1, bias=False)
        # self.bn = nn.BatchNorm2d(256)
        # self.relu = nn.ReLU(inplace=True)

        self.hm_cait = nn.Conv2d(768, 21, kernel_size=1, bias=True)
        self.mask_cait = nn.Conv2d(768, 1, kernel_size=1, bias=True)
        self.hm = nn.Conv2d(256, 21, kernel_size=1, bias=True)
        self.mask = nn.Conv2d(256, 1, kernel_size=1, bias=True)

        #cait
        # self.model_conv = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=7, stride=5)
        # ) #768
        # self.model_conv = nn.Conv2d(768, 256, kernel_size=1, bias=True)

        self.resorcait = resorcait

    def forward(self, x,info):
        self.feature_extractor.eval()

        # for i, layer in enumerate(self.feature_extractor):
        #     print(f'Layer {i}: {layer}')

        if isinstance(self.feature_extractor, timm.models.vision_transformer.VisionTransformer):
            # x = nn.functional.interpolate(x, size=(384, 384), mode='bilinear')
            x = self.feature_extractor.patch_embed(x)
            cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
            if self.feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (
                        cls_token,
                        self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )
            x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
            for i in range(8):  # paper Table 6. Block Index = 7
                x = self.feature_extractor.blocks[i](x)
            x = self.feature_extractor.norm(x)
            x = x[:, 2:, :]
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        elif isinstance(self.feature_extractor, timm.models.cait.Cait):
            # x = nn.functional.interpolate(x, size=(448, 448), mode='bilinear') 
            x = self.feature_extractor.patch_embed(x)
            x = x + self.feature_extractor.pos_embed
            x = self.feature_extractor.pos_drop(x)
            for i in range(41):  # paper Table 6. Block Index = 40
                x = self.feature_extractor.blocks[i](x)
            N, _, C = x.shape
            x = self.feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        else:
            # x = nn.functional.interpolate(x, size=(256, 256), mode='bilinear') 
            features = self.feature_extractor(x)
            #通过遍历 self.norms 中的每个 nn.LayerNorm 层，将对应的特征 feature 应用于归一化操作，从而得到经过归一化的特征 features。
            features = [self.norms[i](feature) for i, feature in enumerate(features)] 
            #features[0]   torch.Size([Batchsize/gpus, 64, 64, 64])   len(features)   3

        loss = 0
        outputs = []
        hm_outs = []
        mask_outs = []
        for i, feature in enumerate(features):
            # print("1",feature.size())
# 1 torch.Size([64, 64, 64, 64])
# 1 torch.Size([64, 128, 32, 32])
# 1 torch.Size([64, 256, 16, 16])
            output, log_jac_dets = self.nf_flows[i](feature)
            # print("2",output.size())
# 2 torch.Size([64, 64, 64, 64])
# 2 torch.Size([64, 128, 32, 32])
# 2 torch.Size([64, 256, 16, 16])
            loss += torch.mean(
                0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
            )

#resnet18
        #output    torch.Size([B/gpus, 256, 16, 16])  log_jac_dets     torch.Size([B/gpus])
            
#cait_448
        # output ([64, 768, 28, 28])
            
        # output = self.model_conv(output)

#common
        # print("1",output.size()) #torch.Size([64, 256, 16, 16])
        # output = self.con(output) #torch.Size([64, 256, 4, 4])
        # print("cccc",output.size())
        output = nn.functional.interpolate(output, size=[64, 64], mode='bilinear', align_corners=False) #torch.Size([64, 256, 64, 64])
        # print("2",output.size()) #torch.Size([64, 256, 64, 64])
        # output = self.res(output)
        # output = self.conv(output)
        # output = self.bn(output)
        # output = self.relu(output)

        if self.resorcait:
            hm_out = self.hm(output) # torch.Size([16, 21, 64, 64])
            mask_out = self.mask(output)
        else:
            hm_out = self.hm_cait(output) # torch.Size([16, 21, 64, 64])
            mask_out = self.mask_cait(output) #torch.Size([16, 1, 64, 64])

        hm_outs.append(hm_out)
        mask_outs.append(mask_out)
        outputs.append(output)

        #fixme3
        sik_reslut = {}
        l_uvd = []
        l_dep = []
        l_joint = []
        ups_result = {
            "l_ff": loss,
            "l_hm": hm_outs,
            "l_mask": mask_outs,
            "l_uvd": l_uvd,
            "l_joint": l_joint,
            "l_dep": l_dep,
        }
        result = {**ups_result, **sik_reslut}
        # return result
    
        return loss,hm_outs,mask_outs
        