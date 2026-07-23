
from torch.cuda.amp import autocast
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .modules import (GeometryFeature,Feature,
                        BasicConv_IN,Conv2x_IN,Conv2x,
                        Conv3dGn,convbnrelu,convtbnrelu,
                        Custom_Hourglass,LiftD,
                        MultiBasicEncoder,
                        DepthPyramidGenerator,
                        hourglass,ThreeHourglass,
                        ChannelAttentionEnhancement,
                        SpatialAttentionExtractor,
                        SparseDownSampleClose
                        )

from .model_func import (context_upsample,soft_code,depth_regression,
                         project_left_to_right_with_crop_batch,groupwise_correlation)
from .ihdr import (BasicMultiUpdateBlock, IHCFR, BasicMultiUpdateBlockPLUSPLUS)
from .hypothesis_guidance import Hypothesis_guided_Cross_scale_Feature_Fusion
from utils.fit_utils import LayerTimer


class AirDC(nn.Module):
    def __init__(self, args, dtype=torch.float16):
        super(AirDC, self).__init__()
        self.args = args
        self.baseline = 0.54
        self.embed_channels = args.embed_channels
        self.num_groups = args.embed_channels
        self.hidden_dim = args.GRU_hidden_dim
        self.target_final_ratio=getattr(args,"target_final_ratio",0.005)
        self.decay = 0 if self.args.num_iters == 1 else self.target_final_ratio ** (1 / (self.args.num_iters - 1))  # Compute the iterative decay factor.
        self.geoplanes = 3
        if self.args.convolutional_layer_encoding == "xyz":
            self.geofeature = GeometryFeature()
        elif self.args.convolutional_layer_encoding == "std":
            self.geoplanes = 0
        elif self.args.convolutional_layer_encoding == "uv":
            self.geoplanes = 2
        elif self.args.convolutional_layer_encoding == "z":
            self.geoplanes = 1
        self.timer=None

        self.shared_feature_extractor = Feature()

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.GroupNorm(16, 32), nn.ReLU()
        )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.GroupNorm(16, 48), nn.ReLU()
        )
        self.conv_feat = BasicConv_IN(96, self.embed_channels, kernel_size=3, stride=1, padding=1)
        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)
        self.feat_chs_list=[96,64,192,160]
        if self.args.get("volume_att", False):
            self.sd_fpnet=LiftD(self.feat_chs_list)
        if self.args.volume_fusion == "corr":
            self.embed_channels=self.num_groups
            self.in_hg_channels=self.num_groups+1

        elif self.args.volume_fusion == "original":
            self.in_hg_channels = self.embed_channels*2 + 1

        if self.args.update_with== "igevplusplus":
            self.agg_all=Custom_Hourglass(args,8)
            self.agg_near0=Custom_Hourglass(args,8)
            self.agg_near1=Custom_Hourglass(args,8)
            self.classifier0 = nn.Conv3d(8, 1, 3, 1, 1, bias=False)
            self.classifier1 = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

            self.near_patch0 = nn.Conv3d(self.in_hg_channels, self.in_hg_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            self.near_patch1 = nn.Conv3d(self.in_hg_channels, self.in_hg_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            self.all_patch = nn.Conv3d(self.in_hg_channels, self.in_hg_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                                     bias=False)
        else:
            if self.args.volume_fusion == "corr":
                self.in_hg_channels = args.embed_channels + 1
            elif self.args.volume_fusion == "original":
                if ('d' in self.args.get("volume_source","rgbd")):
                    self.in_hg_channels = args.embed_channels * 2 + 1
                else:
                    self.in_hg_channels = args.embed_channels * 2
        if args.get("hg_full", False):
            self.asr_net = ThreeHourglass(args=args, in_channels=self.in_hg_channels,
                                           feat_channels=self.feat_chs_list,
                                           embed_channels=32,
                                           out_channels=self.embed_channels)
        else:
            self.asr_net = hourglass(args=args, in_channels=self.in_hg_channels,
                                      feat_channels=self.feat_chs_list,
                                      embed_channels=32,
                                      out_channels=self.embed_channels)
        self.classifier = nn.Sequential(
            Conv3dGn(in_channels=args.embed_channels, out_channels=args.embed_channels, kernel_size=3, stride=1,
                     padding=1, dilation=1,
                     use_relu=True, depth_wise=True if self.args.depth_wise else False),
            nn.Conv3d(in_channels=args.embed_channels, out_channels=1, kernel_size=3, stride=1, padding=1,
                      dilation=1, bias=False),

        )

        self.convout = Conv3dGn(in_channels=args.embed_channels, out_channels=8,
                                kernel_size=3, stride=1, padding=1,
                                dilation=1, use_relu=False, use_norm=False,
                                depth_wise=True if self.args.depth_wise else False)

        self.depth_generator = DepthPyramidGenerator()
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)
        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(48 + 48, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.GroupNorm(8, 24), nn.ReLU()
        )
        context_dims = [self.hidden_dim, self.hidden_dim, self.hidden_dim]
        gru_hidden_dims = [self.hidden_dim, self.hidden_dim, self.hidden_dim]
        if "i" in self.args.train_strategy:
            self.context_extractor = MultiBasicEncoder(args,output_dim=[gru_hidden_dims, context_dims],
                                          norm_fn="group",
                                          downsample=args.n_downsample,
                                          hidden_dim=self.hidden_dim,
                                          geo_planes=self.geoplanes)

            if self.args.confidence_guide==True:
                self.alpha_unc = nn.Parameter(torch.tensor(1.0))
                # Per-level channels: 8 SLDV channels + 1 correlation channel + optional depth prior.
                per_level_ch = 8 + 1 + (1 if 'depth' in args.geo_fn_mode else 0)
                self.num_hypo = args.corr_levels * (2 * args.corr_radius + 1)
                total_ch = self.num_hypo * per_level_ch

                # Multi-hypothesis uncertainty for the update module.
                self.head_update = nn.Sequential(
                    nn.Conv2d(total_ch+32*2+1, self.num_hypo, 3, padding=1),
                    nn.Sigmoid()
                )
                # Pixel-wise uncertainty for step-size control.
                self.head_step = nn.Sequential(
                    nn.Conv2d(total_ch, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 1, 3, padding=1),
                    nn.Sigmoid()
                )

            if self.args.update_with == "igev" :
                self.ihcfr_context_projections = nn.ModuleList([nn.Conv2d(context_dims[i],
                                                                  gru_hidden_dims[i] * 3,
                                                                  3, padding=3 // 2)
                                                        for i in range(args.n_gru_layers)])
                self.ihcfr = BasicMultiUpdateBlock(args, hidden_dims=gru_hidden_dims,
                                                      encoder_output_dim=self.hidden_dim,geo_planes=self.geoplanes)
            elif self.args.update_with == "selective":
                self.ihcfr_channel_attention = ChannelAttentionEnhancement(128)
                self.ihcfr_spatial_attention = SpatialAttentionExtractor()
                self.ihcfr = IHCFR(args, hidden_dims=gru_hidden_dims,
                                                      encoder_output_dim=self.hidden_dim,geo_planes=self.geoplanes)
            elif self.args.update_with == "igevplusplus":
                self.ihcfr_context_projections = nn.ModuleList([nn.Conv2d(context_dims[i],
                                                                  gru_hidden_dims[i] * 3,
                                                                  3, padding=3 // 2)
                                                        for i in range(args.n_gru_layers)])
                self.ihcfr = BasicMultiUpdateBlockPLUSPLUS(args, hidden_dims=gru_hidden_dims,
                                                                  encoder_output_dim=self.hidden_dim,geo_planes=self.geoplanes)

        # Initialize learnable weights.
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
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()


    def enable_timing(self):
        """Enable runtime profiling when requested."""
        self.timer = LayerTimer(self)

    def disable_timing(self):
        if self.timer:
            self.timer.clear()
            self.timer = None

    def print_timings(self):
        if self.timer:
            timings = self.timer.get_timings()
            sorted_timings = sorted(timings.items(), key=lambda x: -x[1])
            print("\n=== Module Timing Analysis ===")
            print(f"{'Module':<30} | {'Time (ms)':>10}")
            print("-" * 43)
            for name, t in sorted_timings:
                print(f"{name:<30} | {t:>10.2f}")
            print("=" * 43)
    def upsample_depth(self, disp, mask_feat_4, stem_2x):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp, spx_pred).unsqueeze(1)
        return up_disp

    def cal_geo_feat(self,K,position,d):
        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))
        d_s2, vm_s2 = self.sparsepooling(d, valid_mask)
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)
        d_s6, vm_s6 = self.sparsepooling(d_s5, vm_s5)
        if self.args.convolutional_layer_encoding == "z":
            return [d, d_s2, d_s3, d_s4, d_s5, d_s6]
        geo_s1 = None
        geo_s2 = None
        geo_s3 = None
        geo_s4 = None
        geo_s5 = None
        geo_s6 = None
        unorm = position[:, 0:1, :, :]
        vnorm = position[:, 1:2, :, :]
        f352 = K[:, 1, 1]
        f352 = f352.unsqueeze(1)
        f352 = f352.unsqueeze(2)
        f352 = f352.unsqueeze(3)
        c352 = K[:, 1, 2]
        c352 = c352.unsqueeze(1)
        c352 = c352.unsqueeze(2)
        c352 = c352.unsqueeze(3)
        f1216 = K[:, 0, 0]
        f1216 = f1216.unsqueeze(1)
        f1216 = f1216.unsqueeze(2)
        f1216 = f1216.unsqueeze(3)
        c1216 = K[:, 0, 2]
        c1216 = c1216.unsqueeze(1)
        c1216 = c1216.unsqueeze(2)
        c1216 = c1216.unsqueeze(3)

        vnorm_s2 = self.pooling(vnorm)
        vnorm_s3 = self.pooling(vnorm_s2)
        vnorm_s4 = self.pooling(vnorm_s3)
        vnorm_s5 = self.pooling(vnorm_s4)
        vnorm_s6 = self.pooling(vnorm_s5)

        unorm_s2 = self.pooling(unorm)
        unorm_s3 = self.pooling(unorm_s2)
        unorm_s4 = self.pooling(unorm_s3)
        unorm_s5 = self.pooling(unorm_s4)
        unorm_s6 = self.pooling(unorm_s5)
        if self.args.convolutional_layer_encoding == "xyz":
            geo_s1 = self.geofeature(d, vnorm, unorm, self.args.crop_height, self.args.crop_width, c352, c1216, f352, f1216)
            geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, self.args.crop_height / 2, self.args.crop_width / 2, c352, c1216, f352, f1216)
            geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, self.args.crop_height / 4, self.args.crop_width / 4, c352, c1216, f352, f1216)
            geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, self.args.crop_height / 8, self.args.crop_width / 8, c352, c1216, f352, f1216)
            geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, self.args.crop_height / 16, self.args.crop_width / 16, c352, c1216, f352, f1216)
            geo_s6 = self.geofeature(d_s6, vnorm_s6, unorm_s6, 352 / 32, 1216 / 32, c352, c1216, f352, f1216)
        elif self.args.convolutional_layer_encoding == "uv":
            geo_s1 = torch.cat((vnorm, unorm), dim=1)
            geo_s2 = torch.cat((vnorm_s2, unorm_s2), dim=1)
            geo_s3 = torch.cat((vnorm_s3, unorm_s3), dim=1)
            geo_s4 = torch.cat((vnorm_s4, unorm_s4), dim=1)
            geo_s5 = torch.cat((vnorm_s5, unorm_s5), dim=1)
            geo_s6 = torch.cat((vnorm_s6, unorm_s6), dim=1)

        return [geo_s1, geo_s2, geo_s3, geo_s4, geo_s5, geo_s6]

    # Construct the geometry volume by applying depth-dependent shifts.
    # Use grid_sample for sub-pixel feature shifts.
    def get_volume(self,points_d,d_index,mask_d,left_rgb_feat, right_rgb_feat,
                   baseline, f, down_factor, z_max, D_max):
        """
        Args:
          points_d:       # (B, N, 3) stores x, y, z coordinates for each sparse point
          mask_d:         # (B, N) indicates valid sparse points
          d_index:        # (B, N, 2) stores pixel coordinates for each sparse depth point
          left_rgb_feat:  (B, C, H, W)
          right_rgb_feat: (B, C, H, W)
          baseline:       float or tensor (B,)
          f:              float or tensor (B,)
          down_factor:    int
          z_max:          float, maximum depth in meters
          D_max:          int, number of depth planes
        Returns:
          volume: (B, 2*C, H, W, D_max)
        """
        B, C, H, W = left_rgb_feat.shape
        device = left_rgb_feat.device
        dtype = left_rgb_feat.dtype

        # ============ Populate left-view features. ============
        volume = left_rgb_feat.new_zeros((B, self.in_hg_channels, H, W, D_max), device=device, dtype=dtype)
        volume[:, :C, ..., :] = left_rgb_feat.unsqueeze(-1).expand(-1, -1, -1, -1, D_max)

        # Convert baseline and focal length to tensors with shape (B,).
        if not torch.is_tensor(baseline):
            baseline = torch.full((B,), baseline, device=device)
        if not torch.is_tensor(f):
            f = torch.full((B,), f, device=device)

        # Construct depth planes for batched sampling.
        step = z_max / D_max
        depths = (torch.arange(D_max, device=device, dtype=dtype).view(1, D_max, 1, 1)+0.5) * step
        disp = baseline.view(B, 1, 1, 1) * f.view(B, 1, 1, 1) / (depths * down_factor + 1e-4)

        # 2) Construct the sampling grid (B, 1, H, W)
        xs = torch.arange(W, device=device).reshape(1, 1, 1, W).expand(B, D_max, H, W)
        ys = torch.arange(H, device=device).reshape(1, 1, H, 1).expand(B, D_max, H, W)

        # 3) Apply batched horizontal shifts (B, D_max, H, W)
        xs_shift = xs.to(dtype) - disp  # -> (B, D_max, H, W)
        ys_shift = ys  # (B, D_max, H, W)

        # Normalize coordinates to the grid_sample range [-1, 1].
        # Normalize coordinates with x_norm = x / (W - 1) * 2 - 1.
        x_norm = xs_shift / (W - 1) * 2.0 - 1.0
        y_norm = ys_shift / (H - 1) * 2.0 - 1.0
        # Create (B, D_max, H, W, 2)
        grid = torch.stack((x_norm, y_norm), dim=-1)
        # Sample right-view features with bilinear interpolation.
        # Merge batch and depth dimensions before sampling.
        feat = right_rgb_feat.unsqueeze(1).expand(-1, D_max, -1, -1, -1)  # (B, D_max, C, H, W)
        feat = feat.reshape(B * D_max, C, H, W)  # (B*D_max, C, H, W)

        grid_flat = grid.reshape(B * D_max, H, W, 2)  # (B*D_max, H, W, 2)

        sampled = F.grid_sample(
            feat,  # (B*D_max, C, H, W)
            grid_flat,  # (B*D_max, H, W, 2)
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )  # -> (B*D_max, C, H, W)

        # Reshape back to (B, C, D_max, H, W)
        sampled = sampled.view(B, D_max, C, H, W).permute(0, 2, 3 ,4 ,1).contiguous()

        # Store sampled features in the right-view volume channels.
        volume[:, C:2*C] = sampled

        # Populate the sparse-depth channel.
        # Downsample sparse pixel coordinates.
        uv = (d_index.clone() / down_factor).long()  # (B, N, 2)
        h_idx = uv[..., 0].clamp(0, H - 1)  # Clamp to [0, H - 1].
        w_idx = uv[..., 1].clamp(0, W - 1)  # Clamp to [0, W - 1].

        # Raw depth values without clamping.
        Z = points_d[..., 2]  # (B, N)

        # Keep points with valid masks and depths in (0, z_max).
        mask_valid = mask_d.to(torch.bool) & (Z > 0) & (Z < z_max)  # (B, N)

        # Batch indices for flattening.
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, Z.shape[1])  # (B, N)

        # Flatten valid point coordinates and batch indices.
        b_flat = b_idx[mask_valid]  # (M,)
        h_flat = h_idx[mask_valid]  # (M,)
        w_flat = w_idx[mask_valid]  # (M,)

        # Continuous depth values and discrete depth-plane indices.
        d_cont = Z / z_max * D_max  # (B, N), in [0, D_max]

        d_idx = torch.floor(d_cont).long()  # (B, N)
        # Clamp indices to [0, D_max - 1].
        d_idx = d_idx.clamp(0, D_max - 1)
        d_flat = d_idx[mask_valid]  # (M,)

        depth_ch = 2 * C  # Index of the sparse-depth channel.

        if self.args.depth_fusion == "hard_code":
            # Set the corresponding voxel value to 1.
            volume[b_flat,
                    depth_ch,
                    h_flat,
                    w_flat,
                    d_flat] = 1.0

        elif self.args.depth_fusion == "soft_code":
            # Distribute linear interpolation weights to adjacent depth planes.
            k0 = d_flat
            k1 = (k0 + 1).clamp(max=D_max - 1)
            w1 = (d_cont[mask_valid] - k0.float()).clamp(min=0.0).to(dtype=volume.dtype)
            w0 = (1.0 - w1).to(dtype=volume.dtype)

            # Accumulate values with index_put_.
            idx0 = (b_flat,
                    torch.full_like(b_flat, depth_ch),
                    h_flat,
                    w_flat,
                    k0)
            idx1 = (b_flat,
                    torch.full_like(b_flat, depth_ch),
                    h_flat,
                    w_flat,
                    k1)

            volume.index_put_(idx0, w0, accumulate=True)
            volume.index_put_(idx1, w1, accumulate=True)

        return volume.permute(0, 1, 4, 2, 3).contiguous() #B,C,H,W,D->B,C,D,H,W

    def get_volume_original(self,left_rgb_feat, right_rgb_feat,d_index,points_d,mask_d,f,D_all):
        device = left_rgb_feat.device
        dtype = left_rgb_feat.dtype
        # Quarter-resolution sampling.
        B,C,h_down_img,w_down_img=left_rgb_feat.shape
        if not hasattr(self, 'left_coord_down') or getattr(self, 'h_down_img', 0) != h_down_img or getattr(self, 'w_down_img', 0) != w_down_img or getattr(self, 'D_all_down', 0) != D_all:
            self.h_down_img = h_down_img
            self.w_down_img = w_down_img
            self.D_all_down = D_all
            x = torch.arange(h_down_img)  # Generate indices from 0 to h_down_img.
            y = torch.arange(w_down_img)  # Generate indices from 0 to w_down_img.
            z = torch.arange(D_all)  # Generate indices from 0 to D_all.
            # Generate a 3D coordinate grid with meshgrid.
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
            self.left_coord_down = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=-1).unsqueeze(0)

        # Stack x, y, and z coordinates into a tensor with shape (N, 3).
        left_coord_down = self.left_coord_down.expand(B, -1, -1).to(device)
        d_down_index = torch.floor(d_index.clone() / self.args.down_factor).long()  # (B, N, 2) H,W
        d_down_depth = torch.floor(points_d[:, :, 2].to(dtype) / self.args.z_max *  D_all).to(dtype)
        d_down_batch = torch.arange(B).unsqueeze(1).expand(-1, d_down_index.shape[1]).to(device)
        #     (B, self.in_hg_channels, H // self.args.down_factor, W // self.args.down_factor,  D_all), device=device,
        volume = left_rgb_feat.new_zeros((B, self.in_hg_channels, h_down_img, w_down_img, D_all), device=device, dtype=dtype)
        if('d' in self.args.get("volume_source","rgbd")):
            if self.args.depth_fusion=="hard_code":
                # Downsampled (u, v), voxel depth bin, and original point coordinates.
                depth_points_coord_down = torch.cat([d_down_index, d_down_depth.unsqueeze(-1), points_d], dim=-1)  # (B,N,6
                depth_points_coord_down = depth_points_coord_down[mask_d]  # (N,6) H,W,D,X,Y,Z
                voxel_coord_with_batch = torch.cat([d_down_batch[mask_d].unsqueeze(-1), depth_points_coord_down[:, 0:3]],
                                                   dim=-1)
                voxel_unique_indices, inverse_indices = torch.unique(voxel_coord_with_batch, return_inverse=True, dim=0)
                voxel_unique_indices = voxel_unique_indices.long()

                volume[voxel_unique_indices[:, 0], 2 * C:, voxel_unique_indices[:, 1],
                                voxel_unique_indices[:, 2],
                                voxel_unique_indices[:, 3]] = 1
            elif self.args.depth_fusion=="soft_code":
                volume=soft_code(volume, points_d, d_down_index, mask_d, self.args.z_max, D_all)
        if self.args.volume_fusion == "original":
            volume[:, 0:C] = left_rgb_feat.unsqueeze(-1).expand_as(volume[:, 0:C])
            # 128 64 D (H,W,D)
            left_coord_down_depth = left_coord_down.clone()
            left_coord_down_depth[:, :, 2] = left_coord_down_depth[:, :, 2] / D_all * self.args.z_max
            right_project_coord_down = project_left_to_right_with_crop_batch(left_coord_down_depth,  # (B,N,3)
                                                                             baseline=self.baseline,
                                                                             f=f,
                                                                             down_sample=True,
                                                                             down_factor=self.args.down_factor).long()  # (B,N,3)
            mask_right_valid = (right_project_coord_down[:, :, 1] >= 0) & (
                    right_project_coord_down[:, :, 1] < w_down_img)
            right_project_coord_down_valid = right_project_coord_down[mask_right_valid].long()  # H,W,D
            batch_index = mask_right_valid.nonzero()[:, 0]
            valid_left_coord = left_coord_down[mask_right_valid].long()  # stores H, W, and D indices
            volume[batch_index, C:2 * C,
                    valid_left_coord[:, 0],
                    valid_left_coord[:, 1],
                    valid_left_coord[:, 2]] = right_rgb_feat[batch_index, :, right_project_coord_down_valid[:, 0],
                                              right_project_coord_down_valid[:, 1]]
        elif self.args.volume_fusion == "corr":
            volume = torch.zeros((B, self.num_groups + 1,h_down_img, w_down_img,  D_all),
                                 device=device,
                                 dtype=dtype)

            left_coord_down_depth = left_coord_down.clone()
            left_coord_down_depth[:, :, 2] = left_coord_down_depth[:, :, 2] /  D_all * self.args.z_max
            right_project_coord_down = project_left_to_right_with_crop_batch(left_coord_down_depth,  # (B,N,3)
                                                                             baseline=self.args.baseline,
                                                                             f=f,
                                                                             down_sample=True,
                                                                             down_factor=self.args.down_factor).long()  # (B,N,3)
            mask_right_valid = (right_project_coord_down[:, :, 1] >= 0) & (
                    right_project_coord_down[:, :, 1] < w_down_img)
            right_project_coord_down_valid = right_project_coord_down[mask_right_valid].long()  # H,W,D
            batch_index = mask_right_valid.nonzero()[:, 0]
            valid_left_coord = left_coord_down[mask_right_valid].long()  # stores H, W, and D indices

            # Handle boundary points on the left image border.
            volume[batch_index, :-1,
                valid_left_coord[:, 0],
                valid_left_coord[:, 1],
                valid_left_coord[:, 2]] = groupwise_correlation(
                                                left_rgb_feat[batch_index, :, valid_left_coord[:, 0],
                                                valid_left_coord[:, 1]],
                                                right_rgb_feat[batch_index, :,
                                                right_project_coord_down_valid[:, 0],
                                                right_project_coord_down_valid[:, 1]], num_groups=self.num_groups)
        volume = volume.permute(0, 1, 4, 2, 3).contiguous()
        return volume

    def forward(self, batch,split="train"):
        left_norm, right_norm = batch['left_rgb'], batch['right_rgb']
        device = left_norm.device
        depth = batch['d']  # (B,1,H,W)
        P = batch['P']
        B, _, h_img, w_img = left_norm.shape
        d_index = batch['d_index'].long().to(device) if isinstance(batch['d_index'], torch.Tensor) else torch.tensor(batch['d_index']).long().to(device)  # (B, N, 2)
        points_d = batch['points_d'].to(device) if isinstance(batch['points_d'], torch.Tensor) else torch.tensor(batch['points_d']).to(device)  # (B, N, 3)
        if self.timer:
            self.timer.reset_timings()
        if (split!="test_completion"):
            mask_d = batch['mask_d'].bool().squeeze(-1)  # (B, N, 1)
        else:
            mask_d = torch.ones_like(points_d[:, :, 0]).bool()

        with ((autocast(enabled=self.args.mixed_precision,
                      dtype=torch.float16 if self.args.mixed_precision else torch.float32))):
            dtype = torch.float16 if self.args.mixed_precision else torch.float32
            # Random depth masking.
            if self.args.training_depth_mask_out_rate > 0 and split == "train":
                dep_original = depth.clone()
                keep_prob = torch.empty(B).uniform_(0,
                                                    1).cuda()  # for each sample in the minibatch, we randomly mask out 0% ~ 100% of the depth pixels

                do_masking = (torch.empty(B).uniform_(0,
                                                      1).cuda() < self.args.training_depth_mask_out_rate).float()  # for some samples, we don't do mask out
                keep_prob = (keep_prob * do_masking) + (
                            1.0 - do_masking)  # for the samples w/o do_masking, keep_prob=1.0. Else keep_prob is sampled value.

                # sample a H*W binary mask for each sample
                keep_mask = keep_prob.reshape(B, 1, 1, 1).expand(dep_original.shape)  # B x 1 x H x W
                keep_mask = torch.bernoulli(keep_mask)  # binary
                depth = dep_original * keep_mask
            #Apply training-time image noise.
            if self.args.add_noise and split == "train":
                if self.args.noise_type == "gaussian":
                    # Assume left_norm and right_norm are normalized to [0, 1].
                    # Noise standard deviation: self.args.noise_std.
                    noise_left = torch.randn_like(left_norm) * self.args.noise_std
                    noise_right = torch.randn_like(right_norm) * self.args.noise_std
                    left_norm = left_norm + noise_left
                    right_norm = right_norm + noise_right
                    # Clamp the images to [0, 1] if bounded intensities are required.

                elif self.args.noise_type == "salt_pepper":
                    # Salt-and-pepper noise sets pixels to 0 or 1 with probability p.
                    p = self.args.noise_sp_prob
                    # Sample a uniform random matrix u ~ U(0, 1) with the same shape.
                    u_left = torch.rand_like(left_norm)
                    u_right = torch.rand_like(right_norm)

                    # Set pixels to 0 for u < p/2, to 1 for p/2 <= u < p, and otherwise keep them unchanged.
                    # The three intervals correspond to salt, pepper, and unchanged pixels.
                    left_sp = left_norm.clone()
                    right_sp = right_norm.clone()

                    # Left view.
                    mask_salt = (u_left >= 0) & (u_left < p / 2)
                    mask_pepper = (u_left >= p / 2) & (u_left < p)
                    left_sp[mask_salt] = 1.0
                    left_sp[mask_pepper] = 0.0
                    # Right view.
                    mask_salt_r = (u_right >= 0) & (u_right < p / 2)
                    mask_pepper_r = (u_right >= p / 2) & (u_right < p)
                    right_sp[mask_salt_r] = 1.0
                    right_sp[mask_pepper_r] = 0.0

                    left_norm = left_sp
                    right_norm = right_sp

                else:
                    raise ValueError(f"Unsupported noise_type: {self.args.noise_type}")
            features_left = self.shared_feature_extractor(left_norm)
            features_right = self.shared_feature_extractor(right_norm)
            stem_2x = self.stem_2(left_norm)
            stem_4x = self.stem_4(stem_2x)
            stem_2y = self.stem_2(right_norm)
            stem_4y = self.stem_4(stem_2y)
            features_left[0] = torch.cat((features_left[0], stem_4x), 1)
            features_right[0] = torch.cat((features_right[0], stem_4y), 1)

            depth_pyramid = self.depth_generator(depth)
            if self.args.get("volume_att",False):
                features_left_d = self.sd_fpnet(features_left,depth_pyramid)

            match_left = self.desc(self.conv(features_left[0]))
            match_right = self.desc(self.conv(features_right[0]))

            left_rgb_feat = self.conv_feat(features_left[0]).to(dtype)
            right_rgb_feat = self.conv_feat(features_right[0]).to(dtype)

            _, _, h_down_img, w_down_img = left_rgb_feat.shape


            if self.args.update_with== "igevplusplus":
                D_all = self.args.D
                near0_factor = 4
                near1_factor = 2
                volume_near0 = self.get_volume(points_d, d_index, mask_d, left_rgb_feat, right_rgb_feat,
                                               self.args.baseline, f=P[:, 0, 0],
                                               down_factor=self.args.down_factor,
                                               z_max=self.args.z_max // near0_factor, D_max=D_all)
                volume_near1 = self.get_volume(points_d, d_index, mask_d, left_rgb_feat, right_rgb_feat,
                                               self.args.baseline, f=P[:, 0, 0],
                                               down_factor=self.args.down_factor,
                                               z_max=self.args.z_max // near1_factor, D_max=D_all)
                volume_all = self.get_volume(points_d, d_index, mask_d, left_rgb_feat, right_rgb_feat,
                                             self.args.baseline, f=P[:, 0, 0],
                                             down_factor=self.args.down_factor,
                                             z_max=self.args.z_max, D_max=D_all)
                # 25mcorresponds to D=50 with interval=0.5m
                agg_depth_near0, geo_encoding_volume_near0 = self.agg_near0(volume_near0,
                                                                            h_down_img, w_down_img,
                                                                            max_depth=self.args.z_max // near0_factor,
                                                                            interval=D_all / (self.args.z_max * near0_factor) * 4)
                #50mcorresponds to D=50 with interval=1m
                agg_depth_near1, geo_encoding_volume_near1 = self.agg_near1(volume_near1,
                                                                            h_down_img, w_down_img,
                                                                            max_depth=self.args.z_max // near1_factor,
                                                                            interval=D_all / (self.args.z_max * near1_factor) * 4)
                # 100mcorresponds to D=50 with interval=2m
                init_depth_all, geo_encoding_volume_all = self.agg_all(volume_all,
                                                                       h_down_img, w_down_img,
                                                                       max_depth=self.args.z_max,
                                                                       interval=D_all / (self.args.z_max) * 4)

                #                                                   init_depth_all], dim=1))

                init_depth = init_depth_all

            else:
                # # Enable this block with vis_project for shift visualization.
                D_all=self.args.D
                # New volume construction uses get_volume.
                # Legacy volume construction uses get_volume_original and may introduce small coordinate-rounding shifts.
                volume=self.get_volume_original(left_rgb_feat,
                                                       right_rgb_feat,
                                                       d_index, points_d,
                                                       mask_d,
                                                       f=P[:, 0, 0],D_all=D_all)
                # vis_project(volume_ori,volume,2)
                if self.args.get("volume_att", False):
                    volume_out = self.asr_net(volume, features=features_left_d)
                    out1 = self.classifier(volume_out)
                else:
                    volume_out = self.asr_net(volume)
                    out1 = self.classifier(volume_out)  # [B, 1, 1/4D, 1/4H, 1/4W]
                if self.args.get("initial_softmax", True):
                    prob = F.softmax(
                    F.interpolate(out1, size=(self.args.z_max, h_down_img, w_down_img), mode='trilinear').squeeze(
                        dim=1),
                    dim=1)
                else:
                    prob =F.interpolate(out1, size=(self.args.z_max, h_down_img, w_down_img), mode='trilinear').squeeze(
                        dim=1)

                init_depth = depth_regression(prob, max_depth=self.args.z_max, interval=1)

                # del prob, volume
            depth_pred_up_iter = []
            disp_pred_up_iter = []
            xspx = self.spx_4(features_left[0])
            xspx = self.spx_2(xspx, stem_2x)
            spx_pred = self.spx(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            if not "i" in self.args.train_strategy:  # Non-iterative inference.
                depth_pred_up = context_upsample(init_depth,
                                                 spx_pred.float()).unsqueeze(1)
                depth_pred_up_iter.append(depth_pred_up)
                if 'disp' in self.args.loss_source:
                    disp_pred_up_iter.append(
                        (self.baseline * P[:, 0, 0]).reshape(-1, 1, 1, 1) / (depth_pred_up + 1e-6))

        if "i" in self.args.train_strategy:

            with autocast(enabled=self.args.mixed_precision,
                          dtype=torch.float16 if self.args.mixed_precision else torch.float32):
                if not self.args.update_with == "igevplusplus":
                    depth_pred_up = context_upsample(init_depth, spx_pred.float()).unsqueeze(1)
                    depth_pred_up_iter.append(depth_pred_up)
                if self.args.update_mode == "rgbd":
                    if self.args.convolutional_layer_encoding != 'std':
                        if self.args.convolutional_layer_encoding == 'z':
                            depth_geo_feat = self.cal_geo_feat(P, batch["position"], depth)
                        cnet_list = self.context_extractor(left_norm, depth=depth, depth_geo_feat=depth_geo_feat)
                    # elif self.args.update_with == "selective" and self.args.attention_active:
                    else:
                        cnet_list = self.context_extractor(left_norm, depth=depth)

                else:
                    cnet_list = self.context_extractor(left_norm)
                net_list = [torch.tanh(x[0]) for x in cnet_list]
                inp_list = [torch.relu(x[1]) for x in cnet_list]

                if self.args.update_with == "selective":
                    inp_list = [self.ihcfr_channel_attention(x) * x for x in inp_list]
                    att = [self.ihcfr_spatial_attention(x) for x in inp_list]

                elif self.args.update_with == "igev" or self.args.update_with == "igevplusplus":
                    inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in
                                zip(inp_list, self.ihcfr_context_projections)]

            geo_block = Hypothesis_guided_Cross_scale_Feature_Fusion

            if self.args.update_with == "igevplusplus":
                geo_fn = geo_block(self.args, match_left.float(),
                                   match_right.float(),
                                   depth_pyramid[:self.args.corr_levels],
                                   geo_volume_near0=geo_encoding_volume_near0,
                                     geo_volume_near1=geo_encoding_volume_near1,
                                     geo_volume_all=geo_encoding_volume_all)
            else:
                geo_encoding_volume = self.convout(volume_out)
                geo_fn = geo_block(self.args, match_left.float(), match_right.float(), depth_pyramid[:self.args.corr_levels],
                                   geo_volume_all=geo_encoding_volume.float())
            f = P[:, 0, 0].view(-1, 1, 1, 1).to(device)  # Ensure shape (B, 1, 1, 1).

            b, c, h, w = match_left.shape
            coords = torch.arange(w).float().to(match_left.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
            curr_f = f / self.args.down_factor
            curr_depth = init_depth.clone()
            # 17G
            # Approximate memory cost is 1 GB per iteration.
            base_steps = self.args.steps  # Initial step size.
            curr_steps = base_steps  # Initialize the current search step.
            for itr in range(self.args.num_iters):
                curr_depth = curr_depth.detach()
                if self.args.update_with == "igevplusplus":
                    (geo_feat_near0, geo_feat_near1,
                         geo_feat_all,
                         init_corr,depth_feat) = geo_fn(curr_depth, coords, curr_f, curr_steps,D_all)
                else:
                    if self.args.update_with == "selective" and self.args.attention_active:
                        corr_feat, depth_feat, geo_feat_all = geo_fn(curr_depth, coords, curr_f, curr_steps,D_all)
                        if('depth' in self.args.geo_fn_mode):
                            geo_feat = torch.cat((geo_feat_all, corr_feat, depth_feat), dim=1)
                        else:
                            geo_feat = torch.cat((geo_feat_all, corr_feat), dim=1)
                        # # Apply depth attention to generated features.
                        # # Fuse features with selective weighting.

                        #                        selective_weights[:, 1:2]*corr_feat,
                        #                        selective_weights[:, 2:3]*depth_feat], dim=1)
                        #           selective_weights[:, 1:2]*corr_feat+\
                        #           selective_weights[:, 2:3]*depth_feat)

                    else:
                         geo_feat = geo_fn(curr_depth, coords, curr_f, curr_steps,D_all)

                if getattr(self.args, "steps_decay", True):
                    curr_steps = base_steps * (self.decay ** itr)  # Decay the search range by iteration.
                else:
                    curr_steps = base_steps

                with (autocast(enabled=self.args.mixed_precision,
                              dtype=torch.float16 if self.args.mixed_precision else torch.float32)):
                    depth_arg = (
                        depth_pyramid[0] if ((self.args.update_mode == "rgbd" and
                                             self.args.convolutional_layer_encoding == 'std')
                                             or
                                             self.args.convolutional_layer_encoding == 'z')
                        else depth_geo_feat[2] if (self.args.convolutional_layer_encoding !='std')
                        else None
                    )
                    if self.args.update_with == "selective":
                        att_arg = att
                    else:
                        att_arg = None
                    if self.args.update_with == "igev":
                        net_list, mask_feat_4, pred_del = self.ihcfr(net_list, inp_list, corr=geo_feat,
                                                                            curr_depth=curr_depth,
                                                                            depth=depth_arg)
                    elif self.args.update_with == "selective":
                        net_list, mask_feat_4, pred_del = self.ihcfr(net_list, inp_list, corr=geo_feat,
                                                                            curr_depth=curr_depth, att=att_arg,
                                                                            depth=depth_arg)

                    elif self.args.update_with == "igevplusplus":
                        net_list, mask_feat_4, pred_del = self.ihcfr(net_list, inp_list, curr_depth=curr_depth,
                                                                            geo_feat_near0=geo_feat_near0,
                                                                            geo_feat_near1=geo_feat_near1,
                                                                            geo_feat_all=geo_feat_all,
                                                                            init_corr=init_corr,
                                                                            att=att_arg,
                                                                            depth_feat=depth_feat,
                                                                            depth=depth_arg)

                if self.args.predict_head=="linear_regression":
                    # Compute mapping parameters.
                    start = 1 - curr_steps * self.args.corr_radius
                    end = 1 + curr_steps * self.args.corr_radius
                    clamp_min=-self.args.corr_radius; clamp_max=self.args.corr_radius
                    clamped_pred = torch.clamp(pred_del, min=clamp_min, max=clamp_max)  # Clamp the update logits.
                    alpha = start + ( (clamped_pred - clamp_min) / (clamp_max - clamp_min+1e-6) ) * (end - start)
                    curr_depth = alpha * curr_depth

                elif self.args.predict_head=="classification":
                    multi_class=pred_del.float()
                    prob_map = torch.softmax(multi_class, dim=1).float()  # [B,9,H,W]

                    alpha = torch.linspace(1 - curr_steps * self.args.corr_radius,
                                           1 + curr_steps * self.args.corr_radius,
                                           2 * self.args.corr_radius + 1).to(device).float()

                    # Select the corresponding alpha value.
                    alpha = torch.einsum('bchw,c->bhw', prob_map, alpha)
                    alpha = alpha.unsqueeze(1)  # [B,1,H,W]
                    curr_depth = alpha * curr_depth
                elif self.args.predict_head == "log_regression":
                    # Log-space regression branch.
                    start = 1 - curr_steps * self.args.corr_radius
                    end = 1 + curr_steps * self.args.corr_radius

                    # Numerical stabilization for a valid interval.
                    log_start = math.log(start)
                    log_end = math.log(end)
                    log_span = log_end - log_start

                    # pred_del is constrained to [-1, 1] by tanh.
                    delta_log = log_start + (pred_del + 1) / 2 * log_span  # Linear mapping.

                    # Depth update.
                    _min_log = torch.log(torch.tensor(0.1, device=curr_depth.device))
                    _max_log = torch.log(torch.tensor(self.args.z_max, device=curr_depth.device))
                    log_depth = torch.clamp(torch.log(curr_depth), min=_min_log, max=_max_log)
                    # Apply the constrained update.
                    curr_depth = torch.exp(log_depth + delta_log)

                depth_pred_up = self.upsample_depth(curr_depth, mask_feat_4.float(),
                                                    stem_2x)
                if torch.isnan(depth_pred_up).any():
                    print("nan in depth_pred_up")
                depth_pred_up_iter.append(depth_pred_up)
                if 'disp' in self.args.loss_source:
                    disp_pred_up_iter.append((self.baseline * P[:, 0, 0]).reshape(-1,1,1,1) / (depth_pred_up+1e-6))
            # Report profiling results after inference.
        if self.timer and split == "test_completion":  # Only report during test-time completion.
            self.print_timings()
        if self.args.update_with == "igevplusplus":
            agg_depth_all = context_upsample(init_depth_all, spx_pred.float())
            agg_depth_near1 = context_upsample(agg_depth_near1, spx_pred.float())
            agg_depth_near0 = context_upsample(agg_depth_near0, spx_pred.float())
            return [agg_depth_near0,agg_depth_near1,agg_depth_all],depth_pred_up_iter,None,left_norm,right_norm,depth
        else:
            if 'disp' in self.args.loss_source:
                return depth_pred_up_iter ,disp_pred_up_iter,left_norm,right_norm,depth
            else:
                return depth_pred_up_iter, None,left_norm,right_norm,depth
