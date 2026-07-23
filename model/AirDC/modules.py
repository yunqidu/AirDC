from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

class DepthPyramidGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.sparsepooling = SparseDownSampleClose(stride=2)

    def forward(self, d):
        # Generate the valid-region mask.
        valid_mask = torch.where(d > 0, torch.ones_like(d), torch.zeros_like(d))
        # Apply cascaded downsampling.
        d_s2, vm_s2 = self.sparsepooling(d, valid_mask)  # 1/2
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)  # 1/4
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)  # 1/8
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)  # 1/16
        d_s6, vm_s6 = self.sparsepooling(d_s5, vm_s5)  # 1/32
        return [d_s3, d_s4, d_s5, d_s6]

class Custom_Hourglass(nn.Module):
    def __init__(self, args,out_feat_channels=8):
        super(Custom_Hourglass, self).__init__()
        self.args = args
        if self.args.volume_fusion == "corr":
            self.in_hg_channels=args.embed_channels+1
        elif self.args.volume_fusion == "original":
            self.in_hg_channels = args.embed_channels*2 + 1
        self.conv0 = nn.Sequential(
            Conv3dGn(in_channels=self.in_hg_channels, out_channels=args.embed_channels,
                     kernel_size=3, stride=1,
                     padding=1,
                     dilation=1,
                     use_relu=True,
                     depth_wise=True if self.args.depth_wise else False),
            Conv3dGn(in_channels=args.embed_channels, out_channels=args.embed_channels,
                     kernel_size=3, stride=1,
                     padding=1, dilation=1, use_relu=True,
                     depth_wise=True if self.args.depth_wise else False)
        )
        self.conv1 = nn.Sequential(
            Conv3dGn(in_channels=args.embed_channels, out_channels=args.embed_channels,
                     kernel_size=3, stride=1,
                     padding=1, dilation=1,
                     use_relu=True,
                     depth_wise=True if self.args.depth_wise else False),
            Conv3dGn(in_channels=args.embed_channels, out_channels=args.embed_channels,
                     kernel_size=3, stride=1,
                     padding=1, dilation=1,
                     use_relu=False,
                     depth_wise=True if self.args.depth_wise else False)
        )
        self.hourglass1 = Hourglass(args, in_channels=args.embed_channels)
        self.classifier = nn.Sequential(
            Conv3dGn(in_channels=args.embed_channels, out_channels=args.embed_channels, kernel_size=3, stride=1,
                     padding=1, dilation=1,
                     use_relu=True, depth_wise=True if self.args.depth_wise else False),
            nn.Conv3d(in_channels=args.embed_channels, out_channels=1, kernel_size=3, stride=1, padding=1,
                      dilation=1, bias=False)
        )
        self.out_feat=Conv3dGn(in_channels=args.embed_channels,out_channels=out_feat_channels,
                               kernel_size=3, stride=1, padding=1,dilation=1, use_relu=True,
                               depth_wise=True if self.args.depth_wise else False)

    def forward(self, volume,h_img,w_img,max_depth,interval):
        conv0_out_all = self.conv0(volume)
        conv1_out_all = self.conv1(conv0_out_all) + conv0_out_all
        _, _, hourglass_feat1 = self.hourglass1(conv1_out_all, scale1=None,
                                              scale2=None, scale3=conv1_out_all)  # b,32,D,64,128
        out_all = self.classifier(hourglass_feat1)  # [B, 1, 1/4D, 1/4H, 1/4W]
        prob_all = F.softmax(
            F.interpolate(out_all, size=(max_depth,h_img, w_img), mode='trilinear').squeeze(1),
            dim=1)
        init_depth_all = depth_regression(prob_all, max_depth=max_depth, interval=1)
        out_feat=self.out_feat(hourglass_feat1)
        return init_depth_all,out_feat


def round_half_up(x):
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)
def convbnrelu(in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,bn=True):
    if not bn:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                   stride=stride, padding=dilation if dilation > 1 else padding,
                                   dilation = dilation, bias=False),
                             nn.ReLU(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                   stride=stride, padding=dilation if dilation > 1 else padding,
                                   dilation = dilation, bias=False),
                             nn.GroupNorm(8,out_planes),
                             nn.ReLU(inplace=True))


def convtbnrelu(in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,bn=True,output_padding=0):
    if not bn:
        return nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size,
                                   stride=stride, padding=dilation if dilation > 1 else padding,
                                    output_padding=output_padding,
                                   dilation = dilation, bias=False),
                             nn.ReLU(inplace=True))
    else:
        return nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size,
                                   stride=stride, padding=dilation if dilation > 1 else padding,
                                   output_padding=output_padding,
                                   dilation = dilation, bias=False),
                             nn.GroupNorm(8,out_planes),
                             nn.ReLU(inplace=True))

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation = dilation,
                                   bias=False),
                         nn.GroupNorm(8,out_planes))

class Conv_3d(nn.Module):
    def __init__(self,in_planes, out_planes,relu=False,deconv=False,**kwargs):
        super(Conv_3d,self).__init__()
        if deconv:
            self.conv = nn.ConvTranspose3d(in_planes, out_planes, bias=False, **kwargs)
        else:
            self.conv = nn.Conv3d(in_planes, out_planes, bias=False, **kwargs)
        self.relu = relu
    def forward(self, x):
        x = self.conv(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=False))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

# Convolutional blocks use group normalization.

class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2,1,1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)


        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, (output_skip.size()[2],output_skip.size()[3])
                                        ,mode='bilinear', align_corners=False)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, (output_skip.size()[2],
                                                        output_skip.size()[3]),mode='bilinear', align_corners=False)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, (output_skip.size()[2],
                                                    output_skip.size()[3]),mode='bilinear', align_corners=False)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, (output_skip.size()[2],
                                                    output_skip.size()[3]),mode='bilinear', align_corners=False)

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature

class Conv3dGn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 use_relu=True,use_norm=True,depth_wise=False,deconv=False,output_padding=0):
        super().__init__()
        layers = []
        if not deconv:
            # Standard or depth-wise convolution.
            if not depth_wise:
                layers.append(
                    nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride,
                              padding, dilation, bias=False)
                )
            else:
                # depth-wise + point-wise
                layers += [
                    nn.Conv3d(in_channels, in_channels,
                              kernel_size, stride,
                              padding, dilation,
                              groups=in_channels, bias=False),
                    nn.Conv3d(in_channels, out_channels,
                              kernel_size=1, bias=False),
                ]
        else:
            # Transposed convolution or upsampling.
            if not depth_wise:
                layers.append(
                    nn.ConvTranspose3d(in_channels, out_channels,
                                       kernel_size, stride,
                                       padding, output_padding,
                                       dilation, bias=False)
                )
            else:
                # Depth-wise transposed convolution followed by point-wise convolution.
                layers += [
                    nn.ConvTranspose3d(in_channels, in_channels,
                                       kernel_size, stride,
                                       padding, output_padding,
                                       dilation, groups=in_channels, bias=False),
                    nn.Conv3d(in_channels, out_channels,
                              kernel_size=1, bias=False),
                ]
        if use_norm:
            # Select the number of groups from the channel count.
            num_groups = 16 if out_channels >= 16 else 8
            layers.append(
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            )
        if use_relu:
            layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        out = self.net(inputs)
        return out

class ChannelAttentionEnhancement(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionEnhancement, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentionExtractor(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionExtractor, self).__init__()
        self.samconv = nn.Conv2d(2, 1, kernel_size,
                                 padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.samconv(x)
        return self.sigmoid(x)

class AttentionModule(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(AttentionModule, self).__init__()
        # Channel attention branch.
        self.cam = ChannelAttentionEnhancement(in_planes, ratio)
        # Spatial attention branch.
        self.sam = SpatialAttentionExtractor(kernel_size)
    def forward(self, x):
        # Apply channel attention first.
        x = self.cam(x) * x
        # Apply spatial attention next.
        x = self.sam(x) * x
        return x

class ASDA(nn.Module):
    def __init__(self,
                 feat_channels: int,
                 in_channels: int,
                 embed_channels: int,
                 ratio: int = 16,
                 small_kernel: int = 1,
                 large_kernel: int = 7):
        super().__init__()
        self.proj3d_small = nn.Conv3d(in_channels, embed_channels, 1, bias=False)
        self.proj3d_large = nn.Conv3d(in_channels, embed_channels, 1, bias=False)
        self.proj2d_small = nn.Conv2d(feat_channels, embed_channels, 1, bias=False)
        self.proj2d_large = nn.Conv2d(feat_channels, embed_channels, 1, bias=False)
        # Small-receptive-field branch.
        self.attn_small = AttentionModule(in_planes=embed_channels,
                                          ratio=ratio,
                                          kernel_size=small_kernel)
        # Large-receptive-field branch.
        self.attn_large = AttentionModule(in_planes=embed_channels,
                                          ratio=ratio,
                                          kernel_size=large_kernel)
        self.fuse3d_small = nn.Conv3d(embed_channels, in_channels, 1, bias=False)
        self.fuse3d_large = nn.Conv3d(embed_channels, in_channels, 1, bias=False)

        # Adaptive gate.
        self.gate_conv3d = nn.Conv3d(embed_channels, 1, kernel_size=1, bias=False)

    def forward(self, x3d, feat2d):
        B, C3d, D, H, W = x3d.shape
        # Two ASAU branches.
        # Small-kernel branch.
        p3d_s = self.proj3d_small(x3d)                            # [B,E,D,H,W]
        p2d_s = self.proj2d_small(feat2d).unsqueeze(2).expand(-1,-1,D,-1,-1)
        fused_s = p3d_s + p2d_s                                   # [B,E,D,H,W]
        flat_s = fused_s.permute(0,2,1,3,4).reshape(B*D, self.proj3d_small.out_channels, H, W)
        attn_s = self.attn_small(flat_s)                          # [B*D,E,H,W]
        attn_s = attn_s.view(B, D, -1, H, W).permute(0,2,1,3,4)    # [B,E,D,H,W]
        out_s = self.fuse3d_small(attn_s)                         # [B,C3d,D,H,W]
        # Large-kernel branch.
        p3d_l = self.proj3d_large(x3d)
        p2d_l = self.proj2d_large(feat2d).unsqueeze(2).expand(-1,-1,D,-1,-1)
        fused_l = p3d_l + p2d_l
        flat_l = fused_l.permute(0,2,1,3,4).reshape(B*D, self.proj3d_large.out_channels, H, W)
        attn_l = self.attn_large(flat_l)
        attn_l = attn_l.view(B, D, -1, H, W).permute(0,2,1,3,4)
        out_l = self.fuse3d_large(attn_l)

        # Fuse branches with a 3D gate.
        gate_feat = attn_s+attn_l        # [B,E,D,H,W]
        gate = torch.sigmoid(self.gate_conv3d(gate_feat))       # [B,1,D,H,W]
        return out_s * gate + out_l * (1 - gate)

class Hourglass(nn.Module):
    def __init__(self,args,in_channels=32,feat_channels=None):
        super().__init__()
        self.args = args
        self.feat_channels = feat_channels
        self.net1 = nn.Sequential(
            Conv3dGn(in_channels=in_channels, out_channels=in_channels*2,
                     kernel_size=3, stride=2, padding=1, dilation=1, use_relu=True,
                     depth_wise=True if self.args.depth_wise else False),
            Conv3dGn(in_channels=in_channels*2, out_channels=in_channels*2, kernel_size=3,
                     stride=1, padding=1, dilation=1, use_relu=False,
                     depth_wise=True if self.args.depth_wise else False)
        )
        self.net2 = nn.Sequential(
            Conv3dGn(in_channels=in_channels*2, out_channels=in_channels*2,
                     kernel_size=3, stride=2, padding=1, dilation=1, use_relu=True,
                     depth_wise=True if self.args.depth_wise else False),
            Conv3dGn(in_channels=in_channels*2, out_channels=in_channels*2,
                     kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True,
                     depth_wise=True if self.args.depth_wise else False)
        )
        self.net3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels*2, out_channels=in_channels*2,
                               kernel_size=3, stride=2, padding=1, output_padding=1, bias=False,
                               ),
            nn.GroupNorm(num_groups=16,num_channels=in_channels*2)
        )
        self.net4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels*2, out_channels=in_channels,
                               kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.GroupNorm(num_groups=16 if in_channels>16 else 8,num_channels=in_channels)
        )

        att_embed_channels = self.args.get("asda_embed_channels", args.embed_channels)
        if self.args.get("volume_att", False):
            self.att1=ASDA(feat_channels[1],
                                          in_channels*2,
                                          att_embed_channels,
                                          small_kernel=1,large_kernel=5)
            self.att2=ASDA(feat_channels[2],
                                          in_channels*2,
                                          att_embed_channels,
                                          small_kernel=1,large_kernel=7)
            #                               in_channels,args.embed_channels,

    def forward(self, inputs, scale1=None,
                    scale2=None, scale3=None,
                    features=None):
        net1_out = self.net1(inputs)  # [B, 64, 1/8D, 1/8H, 1/8W]
        if features is not None:
            net1_out = self.att1(net1_out,features[1])
        net1_out = F.relu(net1_out, inplace=True)

        net2_out = self.net2(net1_out)  # [B, 64, 1/16D, 1/16H, 1/16W]
        if features is not None:
            net2_out = self.att2(net2_out,features[2])
        net3_out = self.net3(net2_out)  # [B, 64, 1/8D, 1/8H, 1/8W]

        net3_out = F.relu(F.interpolate(net3_out,size=net1_out.size()[2:] ,
                             mode='trilinear', align_corners=False)+ net1_out, inplace=True)

        net4_out = self.net4(net3_out) # [B, 32, 1/4D, 1/4H, 1/4W]

        if scale3 is not None:
            net4_out = net4_out + F.interpolate(scale3,size=net4_out.size()[2:] ,
                                    mode='trilinear', align_corners=False)
            net4_out = F.interpolate(net4_out, size=inputs.size()[2:] ,
                                    mode='trilinear', align_corners=False)

        return net4_out


class ThreeHourglass(nn.Module):
    """Three-stage hourglass used when hg_full=True."""

    def __init__(self, args, in_channels, feat_channels, embed_channels=16, out_channels=32):
        super(ThreeHourglass, self).__init__()
        self.args = args
        self.feat_channels = feat_channels

        # Initial convolution block.
        self.conv0 = nn.Sequential(
            Conv3dGn(in_channels=in_channels, out_channels=args.embed_channels,
                     kernel_size=3, stride=1, padding=1, dilation=1,
                     use_relu=True, depth_wise=True if self.args.depth_wise else False),
            Conv3dGn(in_channels=args.embed_channels, out_channels=args.embed_channels,
                     kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True,
                     depth_wise=True if self.args.depth_wise else False)
        )
        self.conv1 = nn.Sequential(
            Conv3dGn(in_channels=args.embed_channels, out_channels=args.embed_channels,
                     kernel_size=3, stride=1, padding=1, dilation=1,
                     use_relu=True, depth_wise=True if self.args.depth_wise else False),
            Conv3dGn(in_channels=args.embed_channels, out_channels=args.embed_channels,
                     kernel_size=3, stride=1, padding=1, dilation=1, use_relu=False,
                     depth_wise=True if self.args.depth_wise else False)
        )

        # Three hourglass modules.
        self.hourglass1 = Hourglass(args, in_channels=args.embed_channels, feat_channels=feat_channels)
        self.hourglass2 = Hourglass(args, in_channels=args.embed_channels, feat_channels=feat_channels)
        self.hourglass3 = Hourglass(args, in_channels=args.embed_channels, feat_channels=feat_channels)

        # Three output heads.
        self.out1 = nn.Sequential(
            Conv3dGn(in_channels=args.embed_channels, out_channels=args.embed_channels,
                     kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True,
                     depth_wise=True if self.args.depth_wise else False),
            nn.Conv3d(in_channels=args.embed_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )
        self.out2 = nn.Sequential(
            Conv3dGn(in_channels=args.embed_channels, out_channels=args.embed_channels,
                     kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True,
                     depth_wise=True if self.args.depth_wise else False),
            nn.Conv3d(in_channels=args.embed_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )
        self.out3 = nn.Sequential(
            Conv3dGn(in_channels=args.embed_channels, out_channels=args.embed_channels,
                     kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True,
                     depth_wise=True if self.args.depth_wise else False),
            nn.Conv3d(in_channels=args.embed_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )
        att_embed_channels = self.args.get("asda_embed_channels", args.embed_channels)
        if self.args.get("volume_att", False):
            self.att0 = ASDA(feat_channels[0],
                             args.embed_channels,
                             att_embed_channels, small_kernel=1,
                             large_kernel=3)

    def forward(self, volume, features=None):
        """
        Args:
            volume: Input volume tensor.
            features: Optional features used by volume attention.
        Returns:
            Final output volume compatible with the hourglass interface.
        """
        # Initial convolution.
        conv0_out = self.conv0(volume)  # [B, 32, 1/4D, 1/4H, 1/4W]
        conv1_out = self.conv1(conv0_out)
        conv1_out = conv0_out + conv1_out  # [B, 32, 1/4D, 1/4H, 1/4W]
        fused_out = self.att0(conv1_out, features[0])
        conv1_out = self.hourglass1(fused_out, scale1=None,
                                    scale2=None, scale3=fused_out,
                                    features=features)  # b,32,D,64,128

        # Three-stage hourglass.
        hourglass1_out = self.hourglass1(conv1_out, features=features) # b,32,D,64,128
        hourglass2_out = self.hourglass2(hourglass1_out, scale3=conv1_out, features=features) # b,32,D,64,128
        hourglass3_out = self.hourglass3(hourglass2_out, scale3=conv1_out, features=features) # b,32,D,64,128

        # Three output heads.
        out1 = self.out1(hourglass1_out)
        out2 = self.out2(hourglass2_out) + out1
        out3 = self.out3(hourglass3_out) + out2

        # Return the final output while preserving the original interface.
        return out3


class hourglass(nn.Module):
    def __init__(self, args,in_channels,feat_channels,embed_channels=16,out_channels=32):
        super(hourglass, self).__init__()
        self.args = args
        self.feat_channels=feat_channels
        self.conv0 = nn.Sequential(
            Conv3dGn(in_channels=in_channels, out_channels=args.embed_channels,
                     kernel_size=3, stride=1,
                     padding=1,
                     dilation=1,
                     use_relu=True,
                     depth_wise=True if self.args.depth_wise else False),
            Conv3dGn(in_channels=args.embed_channels, out_channels=args.embed_channels,
                     kernel_size=3, stride=1,
                     padding=1, dilation=1, use_relu=True,
                     depth_wise=True if self.args.depth_wise else False)
        )
        self.conv1 = nn.Sequential(
            Conv3dGn(in_channels=args.embed_channels, out_channels=args.embed_channels,
                     kernel_size=3, stride=1,
                     padding=1, dilation=1,
                     use_relu=True,
                     depth_wise=True if self.args.depth_wise else False),
            Conv3dGn(in_channels=args.embed_channels, out_channels=args.embed_channels,
                     kernel_size=3, stride=1,
                     padding=1, dilation=1,
                     use_relu=False,
                     depth_wise=True if self.args.depth_wise else False)
        )
        self.hourglass1 = Hourglass(args, in_channels=args.embed_channels,feat_channels=feat_channels)
        att_embed_channels = self.args.get("asda_embed_channels", args.embed_channels)
        if self.args.get("volume_att", False):
            self.att0 = ASDA(feat_channels[0],
                                            args.embed_channels,
                                            att_embed_channels,small_kernel=1,
                                            large_kernel=3)

    def forward(self, volume, features=None):
        if self.args.get("volume_att", False):
            # x: b x C_in x D x H x W
            conv0_out = self.conv0(volume)  # [B, 32, 1/4D, 1/4H, 1/4W]
            conv1_out = self.conv1(conv0_out)
            conv1_out = conv0_out + conv1_out  # [B, 32, 1/4D, 1/4H, 1/4W]
            fused_out=self.att0(conv1_out,features[0])
            volume_out = self.hourglass1(fused_out, scale1=None,
                                             scale2=None, scale3=fused_out,
                                         features=features)  # b,32,D,64,128
            return volume_out
        else:

            conv0_out = self.conv0(volume)  # [B, 32, 1/4D, 1/4H, 1/4W]
            conv1_out = self.conv1(conv0_out)
            conv1_out = conv0_out + conv1_out  # [B, 32, 1/4D, 1/4H, 1/4W]
            volume_out = self.hourglass1(conv1_out, scale1=None,
                                         scale2=None, scale3=conv1_out)  # b,32,D,64,128
            return volume_out

class GeometryFeature(nn.Module):
    def __init__(self):
        super(GeometryFeature, self).__init__()

    def forward(self, z, vnorm, unorm, h, w, ch, cw, fh, fw):
        x = z * (0.5 * h * (vnorm + 1) - ch) / fh
        y = z * (0.5 * w * (unorm + 1) - cw) / fw
        return torch.cat((x, y, z), 1)

def depthwise_separable_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """Depthwise separable convolution: depthwise convolution followed by pointwise convolution."""
    return nn.Sequential(
        # Depthwise convolution with groups equal to the input channels.
        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, groups=in_channels),
        # Pointwise convolution for channel fusion.
        nn.Conv2d(in_channels, out_channels, kernel_size=1)
    )

class DepthAttentionModule(nn.Module):
    def __init__(self,ratio=16):
        super(DepthAttentionModule, self).__init__()
        # Use convolutions to learn depth-aware features.
        self.convl1=nn.Conv2d(32,32,kernel_size=1,padding=0)
        self.convd1 = nn.Conv2d(1, 32, kernel_size=1, padding=0)
        self.conv = nn.Conv2d(32+32, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, left_rgb,depth_map):
        # Predict depth attention with convolutional layers.
        lx=F.relu(self.convl1(left_rgb))
        dx = F.relu(self.convd1(depth_map))
        attention_map = self.conv(torch.cat([lx,dx],dim=1))
        attention_map = self.sigmoid(attention_map)  # (B, 1, H, W)
        return attention_map

class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.GroupNorm(16 if out_channels%16==0 else 8 ,out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.GroupNorm(16 if out_channels%16==0 else 8 ,out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x
class Conv2x(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x
class BasicConv_IN(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, depth_wise=False, **kwargs):
        super(BasicConv_IN, self).__init__()
        self.depth_wise = depth_wise
        self.relu = relu
        self.use_in = IN

        if self.depth_wise and not deconv:  # Replace standard convolution only.
            self.conv = depthwise_separable(
                in_channels, out_channels,
                kernel_size=kwargs.get('kernel_size', 3),
                stride=kwargs.get('stride', 1),
                padding=kwargs.get('padding', 0),
                is_3d=is_3d
            )
        else:
            if is_3d:
                if deconv:
                    self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
                else:
                    self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                if deconv:
                    self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
                else:
                    self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

        if out_channels % 16 == 0:
            self.IN = nn.GroupNorm(16, out_channels)
        elif out_channels % 10 == 0:
            self.IN = nn.GroupNorm(10, out_channels)
        elif out_channels % 8 == 0:
            self.IN = nn.GroupNorm(8, out_channels)
        elif out_channels % 4 == 0:
            self.IN = nn.GroupNorm(4, out_channels)
        else:
            self.IN = nn.GroupNorm(out_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x

class Conv2x_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True,
                 keep_concat=True, IN=True, relu=True, keep_dispc=False,depth_wise=False):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv,
                                      is_3d, IN=True, relu=True, kernel_size=kernel,
                                      stride=stride, padding=padding,depth_wise=depth_wise)
        else:
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d,
                                      IN=True, relu=True, kernel_size=kernel,
                                      stride=2, padding=1,depth_wise=depth_wise)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv_IN(out_channels*2, out_channels*mul,
                                      False, is_3d, IN, relu, kernel_size=3,
                                      stride=1, padding=1,depth_wise=depth_wise)
        else:
            self.conv2 = BasicConv_IN(out_channels, out_channels, False,
                                      is_3d, IN, relu, kernel_size=3,
                                      stride=1, padding=1,depth_wise=depth_wise)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x

def depthwise_separable(in_channels, out_channels, kernel_size, stride=1, padding=0, deconv=False, is_3d=False):
    """Depthwise separable convolution for 2D and 3D inputs."""
    if deconv:  # Keep transposed convolution unchanged.
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding) if is_3d \
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)

    if is_3d:
        return nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels),
            nn.Conv3d(in_channels, out_channels, 1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1)
        )

def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, act=nn.LeakyReLU,last=True):
        super().__init__()
        bias = False
        self.conv1 = nn.Conv2d(inplanes, planes, stride=stride, kernel_size=3,padding=1, bias=bias)
        self.norm1 = nn.GroupNorm(16,planes)
        self.relu = act()
        self.conv2 = nn.Conv2d(planes, planes,kernel_size=3, padding=1, bias=bias)
        self.norm2 = nn.GroupNorm(16,planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last
        if last:
            self.relu2 = act()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if self.last:
            out = self.relu2(out)
        return out

class UpCC(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels,  kernel_size=3, padding=1):
        super().__init__()
        self.upf = convtbnrelu(in_channels, out_channels,kernel_size=4,
                                       stride=2, padding=1)
        self.conv = convbnrelu(mid_channels + out_channels,
                               out_channels,kernel_size=kernel_size,
                                padding=padding)

    def forward(self, x, y):
        """
        """
        out = self.upf(x)
        if out.shape != y.shape:
            out = F.interpolate(
                out,
                size=(y.shape[-2], y.shape[-1]),
                mode='nearest')
        fout = torch.cat([out, y], dim=1)
        fout = self.conv(fout)
        return fout

class ResNetD(nn.Module):
    def __init__(self, inplanes, dplanes=1, act=nn.ReLU,
                 blocknum=2, uplayer=UpCC, depth=1, block=BasicBlock):
        super().__init__()
        self._act = act
        bc = inplanes // 2
        self.inplanes = bc * 2
        encoder_list = [nn.Sequential(
            convbnrelu(inplanes + dplanes, bc * 2,  kernel_size=3, padding=1),
            self._make_layer(block, bc * 2, blocknum, stride=1)
        ), ]
        decoder_list = []
        in_channels = bc * 2
        for i in range(depth):
            self.inplanes = in_channels
            out_channels = min(in_channels * 2, 256)
            encoder_list.append(self._make_layer(block, out_channels, blocknum, stride=2))
            decoder_list.append(uplayer(out_channels, in_channels, in_channels))
            in_channels = min(in_channels * 2, 256)
        self.encoder = nn.ModuleList(encoder_list)
        self.decoder = nn.ModuleList(decoder_list)

    def forward(self, x_feat, d_feat):
        feat = []
        x = torch.cat([x_feat, d_feat], dim=1)
        for layer in self.encoder:
            x = layer(x)
            feat.append(x)
        out = feat[-1]
        for idx in range(len(feat) - 2, -1, -1):
            out = self.decoder[idx](out, feat[idx])
        return out

    def _make_layer(self, block, planes, blocks, stride=1):
        act = self._act
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1,stride=stride),
                nn.GroupNorm(16,planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,act=act))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, act=act))
        return nn.Sequential(*layers)

class LiftD(nn.Module):
    """
    Fuse same-resolution feature and depth-pyramid levels.
    Args:
        features: list of Tensors, len = L, each (B, C_i, H_i, W_i)
        depths:   list of Tensors, len = L, each (B, 1,   H_i, W_i)
    Outputs:
        fused_feats: list of Tensors, len = L, each (B, C_i, H_i, W_i)
    """
    def __init__(self, in_chs_list):
        super().__init__()
        assert isinstance(in_chs_list, (list, tuple)) and len(in_chs_list) > 0
        # Project depth to the same channel dimension as the feature map.
        self.resnetd = nn.ModuleList([
            ResNetD(inplanes=C, dplanes=1)
            for C in in_chs_list
        ])

    def forward(self, in_features, in_depths):
        if isinstance(in_features, (list, tuple)) and isinstance(in_depths, (list, tuple)):
            features = [x.clone() for x in in_features]
            depths= [x.clone() for x in in_depths]
        else:
            features = [in_features]
            depths = [in_depths]
        fused = []
        for lvl, (feat, d) in enumerate(zip(features, depths)):
            # feat: (B, C, H, W), d: (B,1,H,W)
            fused_lvl = self.resnetd[lvl](feat, d)
            fused.append(fused_lvl)
        return fused

class Feature(nn.Module):
    def __init__(self,in_channels=3,out_channels=64):
        super(Feature, self).__init__()
        pretrained =  False
        model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
        layers = [1,2,3,5,6]
        chans = [16, 24, 32, 96, 160]

        # Optional convolution for x0 features.
        if in_channels != 3:
            # Redefine conv_stem for non-RGB input channels.
            out_channels = model.conv_stem.out_channels  # Original conv_stem output channels.
            self.conv_stem = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.conv_stem = model.conv_stem

        self.bn1 = nn.GroupNorm(16,model.conv_stem.out_channels)
        self.act1 = nn.ReLU6(inplace=True)

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

        self.deconv32_16 = Conv2x_IN(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x_IN(chans[3]*2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x_IN(chans[2]*2, chans[1], deconv=True, concat=True)
        self.conv4 = BasicConv_IN(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, input):
        x = self.act1(self.bn1(self.conv_stem(input)))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)

        x16 = self.deconv32_16(x32, x16)
        x8 = self.deconv16_8(x16, x8)
        x4 = self.deconv8_4(x8, x4)
        x4 = self.conv4(x4)
        return [x4, x8, x16, x32]

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, (1, 1), bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, (1, 1), bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, (kernel_size, kernel_size), padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1,
                 geo_planes=3, use_geo=False, depth_wise=False,
                 attention=False, ratio=16):
        super(ResidualBlock, self).__init__()
        self.geo_planes = geo_planes
        if depth_wise:
            self.conv1 = depthwise_separable_conv(
                in_planes + geo_planes if use_geo else in_planes,
                planes,
                stride=stride
            )
            self.conv2 = depthwise_separable_conv(
                planes + geo_planes if use_geo else planes,
                planes
            )
        else:
            self.conv1 = nn.Conv2d(in_planes + geo_planes if use_geo else in_planes, planes, kernel_size=3, padding=1,
                                   stride=stride)
            self.conv2 = nn.Conv2d(planes + geo_planes if use_geo else planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        if attention:
            self.ca = ChannelAttention(planes, ratio=ratio)
            self.sa = SpatialAttention()
        self.attention = attention
        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None

        else:
            if depth_wise:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, groups=in_planes),  # Depthwise convolution.
                    nn.Conv2d(in_planes, planes, kernel_size=1),  # Pointwise convolution.
                    self.norm3
                )
            else:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x, g1=None, g2=None):
        y = x
        if g1 is not None:
            y = torch.cat((x, g1), 1)
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        if g2 is not None:
            y = torch.cat((g2, y), 1)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)
        if self.attention:
            y = self.ca(y) * y
            y = self.sa(y) * y

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)

class CustomSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)  # Keep layer names as 0, 1, 2, ...

    def forward(self, x, **kwargs):
        for name, module in self.named_children():
            if name == '0':
                x = module(x, g1=kwargs.get('g1'), g2=kwargs.get('g2'))
            elif name == '1':
                    x = module(x, g1=kwargs.get('g2'),g2=kwargs.get('g2'))
        return x


class MultiBasicEncoder(nn.Module):
    def __init__(self, args,output_dim=[128], norm_fn='group', dropout=0.0,
                 downsample=3,hidden_dim=128,geo_planes=3):
        super(MultiBasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample
        self.hidden_dim = hidden_dim
        self.geo_planes = geo_planes
        self.args = args


        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        if self.args.depth_wise:
            self.conv1_rgb = depthwise_separable_conv(3, 48)
            self.conv1_d = depthwise_separable_conv(1, 16)
            self.conv1= nn.Sequential(
                depthwise_separable_conv(3, 64, kernel_size=7, padding=3),
                nn.MaxPool2d(kernel_size=3, stride=1 + (downsample > 2), padding=1)  # Compensated stride.
            )
        else:
            self.conv1_rgb = nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1)
            self.conv1_d = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)

            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64

        self.layer1 = self._make_layer(64, stride=1,use_geo=False)

        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1),use_geo=self.args.convolutional_layer_encoding!='std')
        self.layer3 = self._make_layer(self.hidden_dim, stride=1 + (downsample > 0),use_geo=self.args.convolutional_layer_encoding!='std')
        self.layer4 = self._make_layer(self.hidden_dim, stride=2,use_geo=self.args.convolutional_layer_encoding!='std')
        self.layer5 = self._make_layer(self.hidden_dim, stride=2,use_geo=self.args.convolutional_layer_encoding!='std')


        # Deconvolution layers for upsampling
        if self.args.update_with!="igev":
            self.deconv1 = convtbnrelu(self.hidden_dim, self.hidden_dim, kernel_size=4, stride=2, padding=1)  # Upsample to x/8
            self.deconv2 = convtbnrelu(self.hidden_dim, self.hidden_dim, kernel_size=4, stride=2, padding=1)  # Upsample to x/4
            self.deconv3 = convtbnrelu(self.hidden_dim, self.hidden_dim, kernel_size=4, stride=2, padding=1)  # Upsample to x/2

        output_list = []

        for dim in output_dim:
            cnet_att = getattr(self.args, 'cnet_attention', False) if hasattr(self, 'args') else False
            conv_out = nn.Sequential(
                ResidualBlock(self.hidden_dim, self.hidden_dim, self.norm_fn,
                              stride=1,
                              depth_wise=self.args.depth_wise,
                              attention=cnet_att),
                nn.Conv2d(self.hidden_dim, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs04 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            cnet_att = getattr(self.args, 'cnet_attention', False) if hasattr(self, 'args') else False
            conv_out = nn.Sequential(
                ResidualBlock(self.hidden_dim, self.hidden_dim,
                              self.norm_fn, stride=1,
                              depth_wise=self.args.depth_wise,
                              attention=cnet_att),
                nn.Conv2d(self.hidden_dim, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(self.hidden_dim, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1,use_geo=False):
        cnet_att = getattr(self.args, 'cnet_attention', False) if hasattr(self, 'args') else False
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride,
                               geo_planes=self.geo_planes,use_geo=use_geo,
                               depth_wise=self.args.depth_wise,
                               attention=cnet_att)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1,
                               geo_planes=self.geo_planes,use_geo=use_geo,
                               depth_wise=self.args.depth_wise,
                               attention=cnet_att)
        layers = (layer1, layer2)

        self.in_planes = dim
        return CustomSequential(*layers)

    def forward(self, x,
                depth_geo_feat=None,
                depth=None):
        if self.args.convolutional_layer_encoding!='std' and depth_geo_feat is not None  :
             x_rgb = self.conv1_rgb(x)
             x_d = self.conv1_d(depth)
             x = torch.cat([x_rgb,x_d],dim=1)
        elif self.args.convolutional_layer_encoding=='std' and depth is not None:
            x_rgb = self.conv1_rgb(x)
            x_d = self.conv1_d(depth)
            x = torch.cat([x_rgb,x_d],dim=1)
        else:
             x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x) #64 H W
        if self.args.convolutional_layer_encoding!='std' and depth_geo_feat is not None:
            x2 = self.layer2(x,g1=depth_geo_feat[0],g2=depth_geo_feat[1])
            x4 = self.layer3(x2,g1=depth_geo_feat[1],g2=depth_geo_feat[2])
        elif self.args.convolutional_layer_encoding=='std' and depth is not None:
            x2 = self.layer2(x) #96 H/2 W/2
            x4 = self.layer3(x2) #128 H/4 W/4
        else:
            x2 = self.layer2(x)
            x4 = self.layer3(x2)

        if self.args.convolutional_layer_encoding != 'std' and depth_geo_feat is not None:
            x8  = self.layer4(x4,g1=depth_geo_feat[2],g2=depth_geo_feat[3])
            x16 = self.layer5(x8,g1=depth_geo_feat[3],g2=depth_geo_feat[4])
        elif self.args.convolutional_layer_encoding=='std':
            x8  = self.layer4(x4) #128 H/8 W/8
            x16 = self.layer5(x8) #128 H/16 W/16
        else:
            x8  = self.layer4(x4) #128 H/8 W/8
            x16 = self.layer5(x8)#128 H/16 W/16
        if self.args.update_with=="igev":
            outputs04 = [f(x4) for f in self.outputs04]
            outputs08 = [f(x8) for f in self.outputs08]
            outputs16 = [f(x16) for f in self.outputs16]
            return (outputs04, outputs08, outputs16)
        else:
            outputs16 = [f(x16) for f in self.outputs16]
        if self.args.convolutional_layer_encoding != 'std' and depth_geo_feat is not None:
            x_up8 = self.deconv1(x16) #128 H/16 W/16
        elif self.args.convolutional_layer_encoding=='std':
            x_up8 = self.deconv1(x16) #128 H/16 W/16
        else:
            x_up8 = self.deconv1(x16)
        fuse_8=x8+x_up8
        outputs08 = [f(fuse_8) for f in self.outputs08]

        x_up4 = self.deconv2(x_up8)
        fuse_4= x4 + x_up4
        outputs04 = [f(fuse_4) for f in self.outputs04]
        return (outputs04, outputs08, outputs16)


class SparseDownSampleClose(nn.Module):
    def __init__(self, stride):
        super(SparseDownSampleClose, self).__init__()
        self.pooling = nn.MaxPool2d(stride, stride)
        self.large_number = 600

    def forward(self, d, mask=None):
        if mask is None:
            # Encode depth directly without mask-dependent pooling.
            encode_d = - self.large_number - d
            d = - self.pooling(encode_d)
            return d
        else:
            encode_d = - (1 - mask) * self.large_number - d
            d = - self.pooling(encode_d)
            mask_result = self.pooling(mask)
            d_result = d - (1 - mask_result) * self.large_number

            return d_result, mask_result
