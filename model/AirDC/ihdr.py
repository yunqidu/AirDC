import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthRegressionLogHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh()  # Constrain the output range.
        )

    def forward(self, x):
        delta = self.conv(x)
        return delta

class DepthRegressionHead(nn.Module):
    def __init__(self, in_channels,embed_dim=128):
        super().__init__()
        # Regression head.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 1, 3, padding=1),
            # nn.Sigmoid()  # Output range(0,1)
            # nn.Tanh()  # Output range(-1,1)
        )

    def forward(self, x):
        return self.conv(x)  # [-1,1]

class DepthClassify(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(DepthClassify, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, args,hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.args = args
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        # Generate the spatial attention map with a 1x1 convolution.
        self.att_conv = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, h, cz, cr, cq, *x_list):
            x = torch.cat(x_list, dim=1)
            hx = torch.cat([h, x], dim=1)
            z = torch.sigmoid(self.convz(hx) + cz)
            r = torch.sigmoid(self.convr(hx) + cr)
            q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)
            # Spatial attention gate.
            if self.args.attention_active:
                att_map = torch.sigmoid(self.att_conv(q))  # [B,1,H,W]
                q = q * att_map  # Apply spatially adaptive modulation.
            h = (1 - z) * h + z * q
            return h


def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)


def interp(x, dest):
    original_dtype = x.dtype
    x_fp32 = x.float()
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    with torch.cuda.amp.autocast(enabled=False):
        output_fp32 = F.interpolate(x_fp32, dest.shape[2:], **interp_args)
    if original_dtype != torch.float32:
        output = output_fp32.to(original_dtype)
    else:
        output = output_fp32
    return output
# from .submodule3 import AttentionModule
class MMGIG(nn.Module):
    def __init__(self, args,hidden_dim=128,geo_planes=3):
        super(MMGIG, self).__init__()
        self.args = args
        if self.args.confidence_guide==True:
            cor_planes = args.corr_levels * (2*args.corr_radius + 1) * (8+1+1+(1 if 'depth' in args.geo_fn_mode else 0))
        else:
            cor_planes = args.corr_levels * (2*args.corr_radius + 1) * (8+1+(1 if 'depth' in args.geo_fn_mode else 0))

        if self.args.update_with == "igevplusplus":
            cor_planes = 96 +args.corr_levels * (2*args.corr_radius + 1) *(1+(1 if 'depth' in args.geo_fn_mode else 0))
        # elif self.args.update_with == "selective" and self.args.attention_active:
            cor_planes=96
        self.convc1 = (nn.Conv2d(cor_planes, 64, 1, padding=0))
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)


        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        if args.convolutional_layer_encoding != 'std':
            self.convs1 = nn.Conv2d(geo_planes, 64, 7, padding=3)
        else:
            self.convs1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convs2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, hidden_dim-1, 3, padding=1)
        self.conv_final = nn.Conv2d(64+64+64, hidden_dim-1, 3, padding=1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim - 1, (hidden_dim - 1) // 16, 1),  # Reduce channel dimension.
            nn.ReLU(),
            nn.Conv2d((hidden_dim - 1) // 16, hidden_dim - 1, 1),  # Restore channel dimension.
            nn.Sigmoid()
        )
    def forward(self, pred, corr,sparse):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        pred_ = F.relu(self.convd1(pred))
        pred_ = F.relu(self.convd2(pred_))
        if(self.args.update_mode=="rgbd"):
            sparse_= F.relu(self.convs1(sparse))
            sparse_ = F.relu(self.convs2(sparse_))
            cor_pred = torch.cat([cor, pred_,sparse_], dim=1)
            out = F.relu(self.conv_final(cor_pred))
            se_weight = self.se(out)  # [B,C,1,1]
            out = out * se_weight  # Channel reweighting.
        else:
            cor_pred = torch.cat([cor, pred_], dim=1)
            out = F.relu(self.conv(cor_pred))
        return torch.cat([out, pred], dim=1)



class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=[],encoder_output_dim=128,geo_planes=3):
        super().__init__()
        self.args = args
        self.encoder = MMGIG(args,encoder_output_dim,geo_planes)
        self.encoder_output_dim = encoder_output_dim

        self.gru04 = ConvGRU(args,hidden_dims[2], self.encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(args,hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru16 = ConvGRU(args,hidden_dims[0], hidden_dims[1])
        if args.predict_head=='linear_regression':
            self.linear_delta_head = DepthRegressionHead(hidden_dims[2],256)
        elif args.predict_head=='classification':
            self.disp_head = DepthClassify(hidden_dims[2], hidden_dim=self.encoder_output_dim*2, output_dim=2*args.corr_radius+1)
        elif args.predict_head=='log_regression':
            self.log_delta_head = DepthRegressionLogHead(hidden_dims[2])
        factor = 2**self.args.n_downsample

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, corr=None, curr_depth=None,depth=None,update=True):

        if self.args.n_gru_layers == 3:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if self.args.n_gru_layers >= 2:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        # Fuse multi-source correlation features.

        motion_features = self.encoder(curr_depth, corr,depth)
        if self.args.n_gru_layers > 1:
            net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
        else:
            net[0] = self.gru04(net[0], *(inp[0]), motion_features)

        if not update:
            return net
        if self.args.predict_head=='linear_regression':
            delta_disp = self.linear_delta_head(net[0])
        elif self.args.predict_head=='classification':
            delta_disp = self.disp_head(net[0])
        elif self.args.predict_head=='log_regression':
            delta_disp = self.log_delta_head(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp

class CustomEncoder(nn.Module):
    def __init__(self, geo_planes):
        super(CustomEncoder, self).__init__()
        self.convg1 = nn.Conv2d(geo_planes, 128, 1, padding=0)
        self.convg2 = nn.Conv2d(128, 96, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, geo):
        return self.convg2(self.relu(self.convg1(geo)))

class BasicMultiUpdateBlockPLUSPLUS(nn.Module):
    def __init__(self, args, hidden_dims=[],encoder_output_dim=128,geo_planes=3):
        super().__init__()
        self.args = args
        self.encoder = MMGIG(args,encoder_output_dim,geo_planes)
        self.encoder_output_dim = encoder_output_dim
        self.geo_encoder0 = CustomEncoder(geo_planes=(2*args.corr_radius + 1) * 8)
        self.geo_encoder1 = CustomEncoder(geo_planes=(2*args.corr_radius + 1) * 8)
        self.geo_encoder2 = CustomEncoder(geo_planes= args.corr_levels * (2*args.corr_radius + 1) * 8)
        self.geo_conv = CustomEncoder(geo_planes=3*96)
        if self.args.attention_active:
            self.sam = SpatialAttentionExtractor()  # Spatial attention.
            self.gru04 = SelectiveConvGRU(hidden_dims[2],
                                          encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1) + hidden_dims[
                                              2])
            self.gru08 = SelectiveConvGRU(hidden_dims[1],
                                          hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[1] + hidden_dims[2])
            self.gru16 = SelectiveConvGRU(hidden_dims[0], hidden_dims[0] + hidden_dims[1])
        else:
            self.gru04 = ConvGRU(args,hidden_dims[2], self.encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
            self.gru08 = ConvGRU(args,hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
            self.gru16 = ConvGRU(args,hidden_dims[0], hidden_dims[1])
        if args.predict_head=='linear_regression':
            self.linear_delta_head = DepthRegressionHead(hidden_dims[2])
        elif args.predict_head=='classification':
            self.disp_head = DepthClassify(hidden_dims[2], hidden_dim=self.encoder_output_dim*2, output_dim=2*args.corr_radius+1)
        elif args.predict_head=='log_regression':
            self.log_delta_head = DepthRegressionLogHead(hidden_dims[2])
        # factor = 2**self.args.n_downsample

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp,geo_feat_near0=None,
                geo_feat_near1=None,
                geo_feat_all=None,
                curr_depth=None,
                depth=None,
                init_corr=None,
                depth_feat=None,
                selective_weights=None,
                att=None,
                update=True):

        if self.args.attention_active:
            if self.args.n_gru_layers == 3:
                net[2] = self.gru16(att[2],net[2], inp[2], pool2x(net[1]))
            if self.args.n_gru_layers >= 2:
                if self.args.n_gru_layers > 2:
                    net[1] = self.gru08(att[1],net[1], inp[1], pool2x(net[0]), interp(net[2], net[1]))
                else:
                    net[1] = self.gru08(att[1],net[1], inp[1], pool2x(net[0]))
        else:
            if self.args.n_gru_layers == 3:
                net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
            if self.args.n_gru_layers >= 2:
                if self.args.n_gru_layers > 2:
                    net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
                else:
                    net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        geo_feat_near0 = self.geo_encoder0(geo_feat_near0)
        geo_feat_near1 = self.geo_encoder1(geo_feat_near1)
        geo_feat_all = self.geo_encoder2(geo_feat_all)
        combined_geo_feat = torch.cat([geo_feat_near0,geo_feat_near1, geo_feat_all], dim=1)  # Concatenate multi-range geometry features.
        geo_feat = self.geo_conv(combined_geo_feat)
        if self.args.attention_active:
            geo_sa = self.sam(geo_feat)  # Spatial attention.
            geo_feat = geo_feat * geo_sa  # Fuse features after spatial-attention weighting.
        update_feat=torch.cat([geo_feat,init_corr,depth_feat], dim=1)
        motion_features = self.encoder(curr_depth, update_feat,depth)
        if self.args.attention_active:
            motion_features = torch.cat([inp[0], motion_features], dim=1)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(att[0], net[0], motion_features, interp(net[1], net[0]))
        else:
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)
        if not update:
            return net
        if self.args.predict_head=='linear_regression':
            delta_disp = self.linear_delta_head(net[0])
        elif self.args.predict_head=='classification':
            delta_disp = self.disp_head(net[0])
        elif self.args.predict_head=='log_regression':
            delta_disp = self.log_delta_head(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp

class FusionWithAttention(nn.Module):
    def __init__(self,in_planes):
        super(FusionWithAttention, self).__init__()
        # Define attention layers for feature-fusion weights.
        self.query_conv = nn.Conv2d(in_planes, in_planes // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_planes, in_planes // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, geo_feat, corr_feat, depth_feat):
        # Project features to query, key, and value tensors.
        feat=torch.cat([geo_feat,corr_feat,depth_feat], dim=1)
        # Project features to query, key, and value tensors.
        query = self.query_conv(feat)  # [B, 90, H, W]
        key = self.key_conv(feat)  # [B, 90, H, W]
        value = self.value_conv(feat)  # [B, 180, H, W]

        # Flatten spatial dimensions to obtain [B, 90, H*W] and [B, 180, H*W].
        query_flat = query.view(query.size(0), query.size(1), -1)  # [B, 90, H*W]
        key_flat = key.view(key.size(0), key.size(1), -1)  # [B, 90, H*W]
        value_flat = value.view(value.size(0), value.size(1), -1)  # [B, 180, H*W]

        # Compute attention weights.
        attn_weights = torch.matmul(query_flat.transpose(1, 2), key_flat)  # [B, H*W, H*W]
        attn_weights = self.softmax(attn_weights)

        # Apply attention weights to the value tensor.
        out_flat = torch.matmul(attn_weights, value_flat.transpose(1, 2))  # [B, H*W, 180]
        out = out_flat.transpose(1, 2).view(feat.size())  # [B, 180, H, W]

        return out


class RaftConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, kernel_size=3, dilation=1):
        super(RaftConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=(kernel_size+(kernel_size-1)*(dilation-1))//2, dilation=dilation)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=(kernel_size+(kernel_size-1)*(dilation-1))//2, dilation=dilation)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=(kernel_size+(kernel_size-1)*(dilation-1))//2, dilation=dilation)
        # Adaptive update gate.
        # self.adaptive_update_gate = nn.Conv2d(hidden_dim + input_dim, 1, kernel_size=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q

        # # Compute the adaptive update gate from input features.
        # adaptive_gate = torch.sigmoid(self.adaptive_update_gate(hx))
        #
        # # Update with the adaptive gate.
        # h = (1 - z) * h + z * q
        # h = h * adaptive_gate + (1 - adaptive_gate) * h  # Apply the adaptive update gate.

        return h


class SelectiveConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, small_kernel_size=1, large_kernel_size=3):
        super(SelectiveConvGRU, self).__init__()
        self.small_gru = RaftConvGRU(hidden_dim, input_dim, small_kernel_size)
        self.large_gru = RaftConvGRU(hidden_dim, input_dim, large_kernel_size)

    def forward(self, att, h, *x):
        x = torch.cat(x, dim=1)
        h = self.small_gru(h, x) * att + self.large_gru(h, x) * (1 - att)

        return h

class IHCFR(nn.Module):
    def __init__(self, args, hidden_dims=[],encoder_output_dim=128,geo_planes=3):
        super().__init__()
        self.args = args
        self.encoder = MMGIG(args,encoder_output_dim,geo_planes)
        self.encoder_output_dim = encoder_output_dim

        if args.n_gru_layers == 3:
            self.gru16 = SelectiveConvGRU(hidden_dims[0], hidden_dims[0] + hidden_dims[1])
        if args.n_gru_layers >= 2:
            self.gru08 = SelectiveConvGRU(hidden_dims[1],
                                          hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[1] + hidden_dims[2])
        self.gru04 = SelectiveConvGRU(hidden_dims[2],
                                      encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1) + hidden_dims[2])
        if args.predict_head=='linear_regression':
            self.linear_delta_head = DepthRegressionHead(hidden_dims[2],getattr(self.args,"update_dim", 64))
        elif args.predict_head=='classification':
            self.disp_head = DepthClassify(hidden_dims[2], hidden_dim=self.encoder_output_dim*2, output_dim=2*args.corr_radius+1)
        elif args.predict_head=='log_regression':
            self.log_delta_head = DepthRegressionLogHead(hidden_dims[2])

        # factor = 2**self.args.n_downsample
        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp,corr=None, curr_depth=None,
                att=None,depth=None,
                selective_weights=None,
                 update=True):
        if self.args.n_gru_layers == 3:
            net[2] = self.gru16(att[2], net[2], inp[2], pool2x(net[1]))
        if self.args.n_gru_layers >= 2:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(att[1], net[1], inp[1], pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(att[1], net[1], inp[1], pool2x(net[0]))

        motion_features = self.encoder(curr_depth, corr, depth)

        motion_features = torch.cat([inp[0], motion_features], dim=1)
        if self.args.n_gru_layers > 1:
            net[0] = self.gru04(att[0], net[0], motion_features, interp(net[1], net[0]))

        if not update:
            return net
        if self.args.predict_head=='linear_regression':
            delta_disp = self.linear_delta_head(net[0])
        elif self.args.predict_head=='classification':
            delta_disp = self.disp_head(net[0])
        elif self.args.predict_head=='log_regression':
            delta_disp = self.log_delta_head(net[0])

        # scale mask to balence gradients
        # mask_feat_4 = .25 * self.mask_feat_4(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp