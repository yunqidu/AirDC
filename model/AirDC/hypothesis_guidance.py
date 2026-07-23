import torch
import torch.nn.functional as F
def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    assert torch.unique(ygrid).numel() == 1 and H == 1 # This is a stereo problem
    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        mask=mask.permute(0,1,3,2)
        return img * mask.float(), mask.float()
    return img


class Hypothesis_guided_Cross_scale_Feature_Fusion:
    def __init__(self, args,init_fmap1, init_fmap2,depth_pyramid,
                 geo_volume_near0=None,
                 geo_volume_near1=None,
                 geo_volume_all=None,
                  baseline=0.54):
        self.args = args
        self.geo_volume_all_pyramid = []
        self.init_corr_pyramid = []
        self.depth_pyramid = depth_pyramid
        self.depth_geo_pyramid = []
        self.baseline = baseline
        self.steps = args.steps

        # all pairs correlation
        # Compute left-right feature correlation.
        # init_fmap1:[b 96 1/4 1/4]
        init_corr = Hypothesis_guided_Cross_scale_Feature_Fusion.corr(init_fmap1, init_fmap2) #(b,h/4,w/4,w/4)

        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume_all.shape
        if self.args.update_with == "igevplusplus":
            self.geo_volume_near0 = geo_volume_near0.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, 1,
                                                                                    d)  # (b*h/4*w/4,8,1,d)
            self.geo_volume_near1 = geo_volume_near1.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, 1,
                                                                                    d)  # (b*h/4*w/4,8,1,d)
        geo_volume_all = geo_volume_all.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)#(b*h/4*w/4,8,1,d)
        init_corr = init_corr.reshape(b*h*w, 1, 1, w2) ##(b*h/4*w/4,1,1,w/4)
        self.geo_volume_all_pyramid.append(geo_volume_all)
        self.init_corr_pyramid.append(init_corr)
        for i in range(self.args.corr_levels-1):
            geo_volume_all = F.avg_pool2d(geo_volume_all, [1,2], stride=[1,2]) # #(b*h/4*w/4,8,1,d/2)
            self.geo_volume_all_pyramid.append(geo_volume_all)
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])#(b*h/4*w/4,8,1,w/8)
            self.init_corr_pyramid.append(init_corr)



    def __call__(self, depth, coords,curr_f,curr_steps,D_all):
        epsilon = 1e-6
        # curr_disp = curr_f * self.baseline / (init_depth + epsilon)
        # depth: [b, 1, h, w] current depth estimate with shape [B, 1, H, W].
        # coords: [B, H, W, 1], pixel coordinates for each image location.
        # depth: Current depth map [B,H,W]
        # curr_f: Current focal length.
        # r: Search radius.
        # self.baseline: Stereo baseline length.
        # self.steps: Discretized step-size factor.

        r = self.args.corr_radius
        b, _, h, w = depth.shape
        if self.args.update_with == "igevplusplus"\
                or (self.args.update_with == "selective" \
                    and self.args.attention_active):
            init_corr_pyramid = []
            geo_feat_all_pyramid = []
            depth_feat_pyramid = []
        else:
            out_pyramid = []
        depth = depth.permute(0, 2, 3, 1)  # [B,H,W,1,1]
        # Compute disparity-induced sampling offsets.
        # Generate alpha coefficients in [1 - steps*r, 1 + steps*r].
        alpha = torch.linspace(1 - curr_steps * r, 1 + curr_steps * r, 2 * r + 1)
        alpha = alpha.view(1, 1, 2 * r + 1, 1).to(depth.device)  # [1,1,2r+1,1]
        # Generate candidate depths by broadcasting.
        candidates_depth = depth * alpha.view(1, 1, 1, 2 * r + 1)  # [B,H,W,1] * [1,1,2r+1] -> [B,H,W,2r+1]
        # Compute disparity candidates (B,H,W,2r+1,1)
        # disp = self.baseline * curr_f.reshape(b, 1, 1, 1, 1) / (depth.unsqueeze(-2) + epsilon)  # Analytical form.
        # Compute horizontal offsets dx (B,H,W,2r+1,1)
        # dx = self.baseline * curr_f.reshape(b, 1, 1, 1, 1) * (1 - alpha) / (depth.unsqueeze(-2) * (2 - alpha) + epsilon)
        voxel_depth=depth/self.args.z_max * D_all

        if self.args.update_with == "igevplusplus":
            x0 = alpha * voxel_depth.reshape(b * h * w, 1, 1, 1) * 4 # (B*h*w,1,2r+1,1)
            y0 = torch.zeros_like(x0)  # (B*h*w,1,2r+1,1)
            d_lvl0 = torch.cat([x0, y0], dim=-1)  # (B*h*w,1,2r+1,2)  #stores horizontal and vertical offsets.
            geo_feat_near0 = bilinear_sampler(self.geo_volume_near0, d_lvl0, mask=True)[0]
            geo_feat_near0 = geo_feat_near0.view(b, h, w, -1)  # b,h,w,72

            x1 = alpha * voxel_depth.reshape(b * h * w, 1, 1, 1) * 2# (B*h*w,1,2r+1,1)
            y0 = torch.zeros_like(x1)  # (B*h*w,1,2r+1,1)
            d_lvl1 = torch.cat([x1, y0], dim=-1)  # (B*h*w,1,2r+1,2)  #stores horizontal and vertical offsets.
            geo_feat_near1 = bilinear_sampler(self.geo_volume_near1, d_lvl1, mask=True)[0]
            geo_feat_near1 = geo_feat_near1.view(b, h, w, -1)  # b,h,w,72

        for i in range(self.args.corr_levels):
            geo_volume_all = self.geo_volume_all_pyramid[i]   # (B*H*W, 8, 1, 50), each location has 8 channels on a 1 x 50 grid.
            x0 = alpha*voxel_depth.reshape(b*h*w,1,1,1)/2**i #(B*h*w,1,2r+1,1)
            y0 = torch.zeros_like(x0) #(B*h*w,1,2r+1,1)
            d_lvl = torch.cat([x0,y0], dim=-1) #(B*h*w,1,2r+1,2)  #stores horizontal and vertical offsets.
            geo_volume_all = bilinear_sampler(geo_volume_all, d_lvl,mask=True)[0]
            geo_volume_all = geo_volume_all.view(b, h, w, -1) #b,h,w,72

            #Process correlation features.
            init_corr = self.init_corr_pyramid[i]
            deltax=self.baseline*curr_f.reshape(b,1,1,1)/(candidates_depth*2**i+1e-6) #(B,H,W,2r+1)
            init_x0=coords.reshape(b*h*w, 1, 1, 1)/2**i - deltax.reshape(b*h*w,  1,-1, 1)
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            #Resample init_corr at init_coords_lvl with bilinear interpolation.
            init_corr = bilinear_sampler(init_corr, init_coords_lvl, mask=True)[0]
            init_corr = init_corr.view(b, h, w, -1)
            if('depth' in self.args.geo_fn_mode):
                sparse_depth = self.depth_pyramid[i]  # (B, 1, H, W)
                # Compute depth similarity where sparse_depth is valid.
                # Set channels to zero where sparse_depth is invalid.
                # Resize the current depth estimate to the current pyramid level.
                # Generate candidate depths (B, 5, curr_h, curr_w)
                curr_depth=depth * alpha.reshape(1, 1, 2 * r + 1) # [B,H,W,1] * [1,1,2r+1] -> [B,H,W,2r+1]
                curr_depth=curr_depth.permute(0,3,1,2)#[B,2r+1,H,W]
                curr_h, curr_w = h // 2**i, w // 2**i
                if i > 0:
                    curr_depth = F.interpolate(curr_depth, (curr_h, curr_w),
                                               mode='bilinear', align_corners=False)
                else:
                    curr_depth = curr_depth

                diff = torch.abs(curr_depth.reshape(b, 2*r+1, curr_h, curr_w) - sparse_depth)  # (B, 5, curr_h, curr_w)
                sigma = 0.1 * sparse_depth  # Adaptive standard deviation.
                similarity = torch.exp(-diff / (sigma + 1e-6))
                # Create the valid mask and align its dimensions.
                mask = (sparse_depth > 0).float()
                depth_feat = (similarity * mask)  # (B, 9,curr_h, curr_w)
                depth_feat = F.interpolate(depth_feat, (h, w), mode='bilinear', align_corners=False)
                depth_feat=depth_feat.permute(0, 2, 3, 1)
            # Each pyramid level produces geometry and correlation features.
            if self.args.update_with == "igevplusplus" \
                    or (self.args.update_with == "selective" \
                        and self.args.attention_active):
                geo_feat_all_pyramid.append(geo_volume_all)
                init_corr_pyramid.append(init_corr)
                if('depth' in self.args.geo_fn_mode):
                    depth_feat_pyramid.append(depth_feat)
            else:
                out_pyramid.append(geo_volume_all)
                out_pyramid.append(init_corr)
                if('depth' in self.args.geo_fn_mode):
                    out_pyramid.append(depth_feat)

        if self.args.update_with == "igevplusplus":
            init_corr=torch.cat(init_corr_pyramid,dim=-1).permute(0, 3, 1, 2).contiguous().float()
            depth_feat=torch.cat(depth_feat_pyramid,dim=-1).permute(0, 3, 1, 2).contiguous().float()
            geo_feat_all=torch.cat(geo_feat_all_pyramid,dim=-1).permute(0, 3, 1, 2).contiguous().float()
            geo_feat_near0=geo_feat_near0.permute(0, 3, 1, 2).contiguous().float()
            geo_feat_near1=geo_feat_near1.permute(0, 3, 1, 2).contiguous().float()
            return geo_feat_near0, geo_feat_near1,geo_feat_all,init_corr,depth_feat
        else:
            if self.args.update_with == "selective" and self.args.attention_active:
                corr_feat = torch.cat(init_corr_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous().float()
                if 'depth' in self.args.geo_fn_mode:
                    depth_feat = torch.cat(depth_feat_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous().float()
                else:
                    depth_feat = None
                geo_feat_all = torch.cat(geo_feat_all_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous().float()
                return corr_feat, depth_feat, geo_feat_all
            else:
                out = torch.cat(out_pyramid, dim=-1)
            # Concatenate all pyramid features along the channel dimension.
            out=out.permute(0, 3, 1, 2).contiguous().float()
            return out

    @staticmethod
    def corr(fmap1, fmap2):
        B, C, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, C, H, W1)
        fmap2 = fmap2.view(B, C, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr