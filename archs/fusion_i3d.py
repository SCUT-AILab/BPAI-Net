import torch
import torch.nn as nn
import torch.nn.functional as F

from .i3d_model import I3D
from torchvision.ops import roi_align
from archs.st_gcn_i3d import Model as st_gcn
class fusion(nn.Module):

    def __init__(self,first,second,patch_size,stride,concat_layer,xyc,bn):
        super(fusion, self).__init__()
        self.first = first
        self.second = second
        self.patch_size = patch_size

        self.cnn = I3D(num_classes=34)
        index = torch.linspace(1, 29, 8, dtype=torch.long)
        self.register_buffer('index',index)
        mlp_channels = 1024

        in_channels = 2
        if xyc:
            in_channels = 3
        self.st_gcn = st_gcn(
            in_channels=in_channels, num_class=34, edge_importance_weighting=True,stride=stride,
            concat_layer=concat_layer,bn=bn,graph_args={'layout': 'drive', 'strategy': 'spatial'}
        )
        self.softmax = nn.Softmax(dim=-1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(mlp_channels, 34)
        self.mlp1 = nn.Linear(mlp_channels, 1024)
        self.relu1 = nn.ReLU()
        self.mlp2 = nn.Linear(1024, mlp_channels)
        self.ske_mlp = nn.Linear(256, mlp_channels)

    def forward(self,rgb,ske,boxes=None): # ske[ N, C, T, V, M]
        #[BT,C,H,W]
        rgb_first,rgb = self.cnn(rgb,self.first)
        patch = self.get_roi(rgb_first,boxes)
        ske_fusion,ske_result = self.st_gcn(ske,patch)
        #[BT,C,H,W]
        rgb = self.fusion(ske_fusion,rgb)
        rgb = self.cnn(rgb,first=None,second = self.second)
        return rgb,ske_result

    def get_roi(self,rgb_first,boxes):

        B, T, N, box = boxes.size() #8
        boxes = boxes.reshape(B * T * N, box).float()
        boxes = list(boxes.split(N, dim=0))
        B, C, t, H, W = rgb_first.size() # 8,64,32,56,56
        rgb_first = rgb_first.index_select(dim=2,index=self.index) #8,64,8,56,56
        rgb_first = rgb_first.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        patch = roi_align(rgb_first, boxes, 7) # 64,13,64 7,7
        patch = self.avgpool(patch).squeeze() #64,13,64
        patch = patch.reshape(B, T, N, -1).permute(0, 3, 1, 2)

        return patch

    def fusion(self,ske_feature,rgb_feature):
        '''
        :param ske_feature: [B,C,T,V]
        :param rgb_feature: [B,C,T,H,W] [BT,C,H,W]
        :return:[B,C,T,H,W]
        '''
        ske_feature = ske_feature.permute(0,2,3,1)
        ske_feature = self.ske_mlp(ske_feature)
        ske_feature = ske_feature.permute(0,3,1,2)
        b, c, t, v = ske_feature.size()
        ske_feature = ske_feature.permute(0, 2, 3, 1).reshape(b * t, v, c)  # [BT,V,C]
        B, C, T, H, W = rgb_feature.size()
        rgb_feature = rgb_feature.permute(0,2,1,3,4).reshape(B*T, C, H * W) #[BT,C,HW]
        weight = torch.matmul(ske_feature, rgb_feature)  # [BT,V,HW]
        # softmax
        weight = self.softmax(weight).permute(0, 2, 1)  # [BT,HW,V]
        ske_feature = torch.matmul(weight, ske_feature).permute(0, 2, 1)  #[BT,C,HW]
        feature = ske_feature + rgb_feature #[BT,C,HW]
        feature = feature.reshape(B*T, C, H, W).permute(0,2,3,1) #[BT,HW,C]
        mlp_feature = self.mlp1(feature)
        mlp_feature = self.relu1(mlp_feature)
        mlp_feature = self.mlp2(mlp_feature)
        fusion_feature = mlp_feature + feature #[BT,HW,C]
        fusion_feature = fusion_feature.permute(0,3,1,2) #[BT,C,HW]
        fusion_feature = fusion_feature.reshape(B,T,C,H,W).permute(0,2,1,3,4) #[B,C,T,H,W]

        return fusion_feature
