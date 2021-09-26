import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.resnet import resnet50, resnet18
from archs.mobilenet_v2 import mobilenet_v2
import time
from torchvision.ops import roi_align
from archs.st_gcn_pretrained import Model as st_gcn
from opts import parser

args = parser.parse_args()


class fusion(nn.Module):

    def __init__(self, num_class, pretrain, first, second, patch_size, stride,
                concat_layer, xyc, bn, arch_cnn, dropout=0.5):
        super(fusion, self).__init__()
        # from opts import parser
        # args = parser.parse_args()
        self.first = first
        self.second = second
        self.patch_size = patch_size
        if arch_cnn == 'mobilenetv2':
            self.cnn = mobilenet_v2(pretrain)
            classify_channels = 1280
            mlp_channels = 1280
            middle_channels = 1024
        elif arch_cnn == 'resnet50':
            classify_channels = 2048
            self.cnn = resnet50(pretrain)
            if self.second == 'layer2':
                mlp_channels = 512
                middle_channels = 256
            elif self.second == 'layer3':
                mlp_channels = 1024
                middle_channels = 512
            elif self.second == 'layer4':
                mlp_channels = 2048
                middle_channels = 1024
        else:
            raise NotImplementedError('{} is not supported'.format(arch_cnn))
        in_channels = 2
        if xyc:
            in_channels = 3
        self.st_gcn = st_gcn(
            in_channels=in_channels, num_class=num_class, edge_importance_weighting=True,stride=stride,
            concat_layer=concat_layer, bn=bn, dropout=dropout,
            graph_args={'layout': 'drive', 'strategy': 'spatial'}, arch_cnn=arch_cnn
        )
        self.softmax = nn.Softmax(dim=args.softmax_dim)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(classify_channels, num_class)
        self.mlp1 = nn.Linear(mlp_channels, middle_channels)
        self.relu1 = nn.ReLU()
        self.mlp2 = nn.Linear(middle_channels, mlp_channels)
        self.ske_mlp = nn.Linear(256, mlp_channels)

    def forward(self, rgb, ske, boxes=None,):  # ske[ N, C, T, V, M]
        # [BT,C,H,W]
        rgb_first, rgb = self.cnn(rgb, first=self.first, second=self.second)
        patch = self.get_roi(rgb_first, boxes)

        ske_fusion, ske_result = self.st_gcn(ske, patch)# [BT,C,H,W]

        rgb = self.fusion(ske_fusion, rgb)
        if self.second in ['layer2', 'layer3']:
            rgb = self.cnn(rgb, first=None, second=self.second)

        rgb = self.avgpool(rgb)  # torch.Size([64, 2048, 1, 1])
        rgb = torch.flatten(rgb, 1)  # torch.Size([64, 2048])
        rgb = self.fc(rgb)  # torch.Size([64, 2048])
        return ske_result, rgb

    def get_roi(self, rgb_first, boxes):

        B, T, N, box = boxes.size()
        boxes = boxes.reshape(B * T * N, box).float()
        boxes = list(boxes.split(N, dim=0))
        patch = roi_align(rgb_first, boxes, 7)
        patch = self.avgpool(patch).squeeze()
        patch = patch.reshape(B, T, N, -1).permute(0, 3, 1, 2)
        return patch

    def fusion(self, ske_feature, rgb_feature):
        '''
        :param ske_feature: [B,C,T,V]
        :param rgb_feature: [BT,C,H,W]
        :return:[B,C,T,H,W]
        '''
        ske_feature = ske_feature.permute(0, 2, 3, 1)
        ske_feature = self.ske_mlp(ske_feature)
        ske_feature = ske_feature.permute(0, 3, 1, 2)
        b, c, t, v = ske_feature.size()
        ske_feature = ske_feature.permute(0, 2, 3, 1).reshape(b * t, v, c)  # [BT,V,C]
        BT, C, H, W = rgb_feature.size()
        rgb_feature = rgb_feature.reshape(BT, C, H * W)
        weight = torch.matmul(ske_feature, rgb_feature)  # [BT,V,HW]
        # softmax
        weight = self.softmax(weight).permute(0, 2, 1)  # [BT,HW,V]
        ske_feature = torch.matmul(weight, ske_feature).permute(0, 2, 1)
        feature = ske_feature + rgb_feature
        feature = feature.reshape(BT, C, H, W).permute(0, 2, 3, 1)
        mlp_feature = self.mlp1(feature)
        mlp_feature = self.relu1(mlp_feature)
        mlp_feature = self.mlp2(mlp_feature)
        fusion_feature = mlp_feature + feature
        fusion_feature = fusion_feature.permute(0, 3, 1, 2)

        return fusion_feature
