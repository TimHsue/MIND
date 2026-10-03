"""Holoplane encoding, SDF decoding and displacement/property prediction."""


import numpy as np
import torch
import torch.nn as nn

class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=1, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels, conv, norm_shape, downsample=False, last_layer=False):
        super().__init__()
        self.norm0 = GroupNorm(num_channels=in_channels)
        self.conv1 = conv

        self.last_layer = last_layer
        if emb_channels and emb_channels > 0:
            self.affine = nn.Linear(in_features=emb_channels, out_features=out_channels*2)
            self.norm1 = nn.LayerNorm(norm_shape)

        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(norm_shape)


        self.downsample = nn.Identity()
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=conv.stride),
            )

    def forward(self, x, emb):
        identity = x

        x = self.conv1(nn.functional.silu(self.norm0(x)))

        if emb is not None:
            params = self.affine(emb).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(x.dtype)
            scale, shift = params.chunk(chunks=2, dim=1)
            x = nn.functional.silu(torch.addcmul(shift, self.norm1(x), scale + 1))

        x = self.conv2(x)
        x = self.norm2(x)
        x = nn.functional.silu(x)

        identity = self.downsample(identity)

        x += identity
        if not self.last_layer:
            x = nn.functional.silu(x)

        return x

class VoxelProjector(nn.Module):
    """Project a voxel SDF along one axis into a feature plane."""
    def __init__(self,
        resolution,
        target_dim,
        in_channels,
        emb_channels,
        out_channels,
        out_resolution,
        label_dim,
        label_scale,
    ):
        super().__init__()
        self.resolution = resolution
        self.target_dim = target_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_resolution = out_resolution
        self.depth = resolution.bit_length() - 1

        self.map_labels = torch.nn.ModuleList([
            LabelFourierEmbedding(label_dim=1, num_channels=emb_channels, scale=label_scale[i]) for i in range(len(label_scale))
        ])
        self.label_emb = nn.Linear(in_features=emb_channels*label_dim, out_features=emb_channels) if label_dim else None
        emb_channels = emb_channels if label_dim else 0

        self.conv_layers = nn.ModuleList()

        now_channels = in_channels
        now_resolution = resolution
        for now_depth in range(self.depth):
            next_channels = out_channels if now_depth == self.depth - 1 else (now_depth + 1) * 8

            if next_channels > out_channels:
                next_channels = out_channels

            kernel_other = 2 if now_resolution > out_resolution else 1
            now_resolution = int(now_resolution / kernel_other)
            if target_dim == 0:
                conv = nn.Conv3d(now_channels, next_channels, kernel_size=(2, kernel_other, kernel_other), stride=(2, kernel_other, kernel_other))
                norm_shape = [next_channels, resolution // 2**(now_depth + 1), now_resolution, now_resolution]
            elif target_dim == 1:
                conv = nn.Conv3d(now_channels, next_channels, kernel_size=(kernel_other, 2, kernel_other), stride=(kernel_other, 2, kernel_other))
                norm_shape = [next_channels, now_resolution, resolution // 2**(now_depth + 1), now_resolution]
            elif target_dim == 2:
                conv = nn.Conv3d(now_channels, next_channels, kernel_size=(kernel_other, kernel_other, 2), stride=(kernel_other, kernel_other, 2))
                norm_shape = [next_channels, now_resolution, now_resolution, resolution // 2**(now_depth + 1)]
            else:
                raise ValueError("dim must be 0, 1, or 2")

            downsample = conv.stride[0] > 1 or conv.stride[1] > 1 or conv.stride[2] > 1 or now_channels != next_channels

            self.conv_layers.append(
                ResidualBlock3D(in_channels=now_channels, out_channels=next_channels, emb_channels=emb_channels, conv=conv, norm_shape=norm_shape, downsample=downsample, last_layer=now_depth == self.depth - 1)
            )

            now_channels = next_channels


    def forward(self, x, labels):
        B, H, W, D = x.shape
        x = x.reshape(B, self.in_channels, H, W, D)

        if self.label_emb is not None:
            label_embeddings = torch.stack([map_label(labels[:, i:i+1]) for i, map_label in enumerate(self.map_labels)], dim=1)  # Shape: (batch_size, label_dim, noise_channels)
            label_embeddings = label_embeddings.reshape(label_embeddings.shape[0], label_embeddings.shape[1], 2, -1).flip(2).reshape(label_embeddings.shape[0], -1)
            emb_labels = self.label_emb(label_embeddings) * np.sqrt(len(self.map_labels))
        else:
            emb_labels = None

        for layer in self.conv_layers:
            x = layer(x, emb_labels)

        B, C, H, W, D = x.shape
        # x shape [b, out_channels, 1, r, r] / [b, out_channels, r, 1, r] / [b, out_channels, r, r, 1]
        x = x.reshape(B, C, self.out_resolution, self.out_resolution)
        # x shape [b, out_channels, r, r]

        return x

class LabelFourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, label_dim, scale=10):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(num_channels) * scale, requires_grad=False) # shape: [num_channels * 2]
        self.out = torch.nn.Linear(label_dim * num_channels * 2, num_channels)

    def forward(self, x):
        # x: [batch_size, label_dim]
        batch_size = x.shape[0]
        freqs = self.freqs  # [num_channels]
        x = x.unsqueeze(2)  # [batch_size, label_dim, 1]
        freqs = freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, num_channels]
        x = x * (2 * np.pi * freqs)  # Element-wise multiplication
        x = torch.cat([x.cos(), x.sin()], dim=2)  # [B, label_dim, 2*num_channels]
        x = x.view(batch_size, -1)  # Flatten to [batch_size, label_dim * num_channels]
        x = self.out(x)  # Final linear layer
        return x

class HoloplaneEncoder(nn.Module):
    """Encode voxel SDFs and physical conditions into [3,B,C,H,W] holoplanes."""
    def __init__(self,
        resolution,
        in_channels=1,
        out_channels=32,
        out_resolution=64,
        emb_channels=64,
        label_dim=0,
        label_scale=[],
        device='cpu'
    ):
        if label_dim != len(label_scale):
            raise ValueError("label_scale must match label_dim")
        if (resolution < 1 or out_resolution < 1 or resolution & (resolution - 1)
                or out_resolution & (out_resolution - 1) or out_resolution > resolution):
            raise ValueError("Input and output resolutions must be powers of two; output cannot exceed input")
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_resolution = out_resolution
        self.device = device

        # Projection along x, y, and z axes
        args = {
            'resolution': resolution,
            'target_dim': 0,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'out_resolution': out_resolution,
            'emb_channels': emb_channels,
            'label_dim': label_dim,
            'label_scale': label_scale,
        }
        self.projection_x = VoxelProjector(**args)
        args['target_dim'] = 1
        self.projection_y = VoxelProjector(**args)
        args['target_dim'] = 2
        self.projection_z = VoxelProjector(**args)

    def forward(self, x, labels):

        # Apply downsampling along each axis
        plane_yz = self.projection_x(x, labels)  # shape: [b, oc, or, or]
        plane_xz = self.projection_y(x, labels)
        plane_xy = self.projection_z(x, labels)

        # Stack the three feature planes; the decoder merges plane and channel axes.
        latent = [plane_yz, plane_xz, plane_xy]

        latent = torch.stack(latent, dim=0)  # shape: [3, b, oc, or, or]

        return latent


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 96)
        self.fc3 = nn.Linear(96, 96)
        self.fc4 = nn.Linear(96, 64)
        self.out = nn.Linear(64, output_dim)

        self.shortcut1 = nn.Linear(input_dim, 128) if input_dim != 128 else nn.Identity()
        self.shortcut2 = nn.Linear(128, 96)
        self.shortcut3 = nn.Linear(96, 64)

    def forward(self, x):
        identity = x # [b, I]
        out = self.fc1(x) # [b, 128]
        out = nn.functional.silu(out)

        identity = self.shortcut1(identity) # [b, 128]
        out = out + identity
        out = nn.functional.silu(out)

        identity = out # [b, 128]
        out = self.fc2(out) # [b, 96]
        out = nn.functional.silu(out)
        identity = self.shortcut2(identity) # [b, 96]
        out = out + identity
        out = nn.functional.silu(out)

        identity = out # [b, 96]
        out = self.fc3(out) # [b, 96]
        out = nn.functional.silu(out)
        out = out + identity
        out = nn.functional.silu(out)

        identity = out # [b, 128]
        out = self.fc4(out) # [b, 64]
        out = nn.functional.silu(out)
        identity = self.shortcut3(identity) # [b, 64]
        out = out + identity
        out = nn.functional.silu(out)

        out = self.out(out)
        return out

class HoloplaneDecoder(nn.Module):
    def __init__(
        self,
        channels,
        aggregate_fn='sum',
        use_tanh=False,
        out_channels=1,
        use_coord=False,
        device='cpu',
    ):
        super().__init__()

        if aggregate_fn != 'sum':
            raise ValueError('The MIND decoder uses sum aggregation')
        self.aggregate_fn = aggregate_fn

        self.channels = channels
        self.use_tanh = use_tanh
        self.use_coord = use_coord
        self.device = device

        MLP_in_channels = channels * 3

        if use_coord:
            self.coor_feature_decoder = LabelFourierEmbedding(label_dim=3, num_channels=MLP_in_channels, scale=10)
        self.MLPdecoder = ResidualMLP(MLP_in_channels, out_channels)

        # init parameters


    def sample_plane_vali(self, coords2d, plane, sym=48):
        # plane shape [b, c, r, r]
        # coords2d shape [b, p, 2]
        assert len(coords2d.shape) == 3, coords2d.shape

        # 生成平面的对称版本
        if sym == 8 or sym == 48:
            plane_flipped_x = torch.flip(plane, dims=[2])  # 沿 x 轴对称
            plane_flipped_y = torch.flip(plane, dims=[3])  # 沿 y 轴对称
            plane_flipped_xy = torch.flip(plane, dims=[2, 3])  # 沿 x 和 y 轴同时对称

        if sym == 48:
            plane_swap_x_y = plane.permute(0, 1, 3, 2)  # 交换 x 和 y 轴
            plane_flipped_x_swap_x_y = torch.flip(plane_swap_x_y, dims=[2])  # 沿 x 轴对称
            plane_flipped_y_swap_x_y = torch.flip(plane_swap_x_y, dims=[3])  # 沿 y 轴对称
            plane_flipped_xy_swap_x_y = torch.flip(plane_swap_x_y, dims=[2, 3])  # 沿 x 和 y 轴同时对称

        # 对平面进行求和取平均
        if sym == 1:
            symmetric_plane = plane
        if sym == 8:
            symmetric_plane = (plane + plane_flipped_x + plane_flipped_y + plane_flipped_xy) / 4
        elif sym == 48:
            symmetric_plane = (plane + plane_flipped_x + plane_flipped_y + plane_flipped_xy + plane_swap_x_y + plane_flipped_x_swap_x_y + plane_flipped_y_swap_x_y + plane_flipped_xy_swap_x_y) / 8

        # 使用对称后的 plane 进行采样
        sampled_features = torch.nn.functional.grid_sample(
            symmetric_plane,
            coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
            mode='bilinear', padding_mode='zeros', align_corners=True
        )

        # 获取采样结果的形状
        N, C, H, W = sampled_features.shape

        # 将采样结果展平为 [b, p, c]
        sampled_features = sampled_features.reshape(N, C, H * W).permute(0, 2, 1)

        return sampled_features


    def forward(self, holoplane_latents, coordinates, sym=48):

        # Input features: [3, B, C, R, R].


        holoplane_latents = holoplane_latents.permute(1, 0, 2, 3, 4) # [B, 3, C, R, R]
        holoplane_latents = holoplane_latents.reshape(holoplane_latents.shape[0], -1, holoplane_latents.shape[-2], holoplane_latents.shape[-1])

        # Use tanh to clamp holoplanes
        if self.use_tanh:
            holoplane_latents = torch.tanh(holoplane_latents)

        # Merged features: [B, 3*C, R, R].

        sample_plane = self.sample_plane_vali

        yz_embed = sample_plane(coordinates[..., 1:3], holoplane_latents, sym=sym)
        xz_embed = sample_plane(coordinates[..., [0, 2]], holoplane_latents, sym=sym)
        xy_embed = sample_plane(coordinates[..., 0:2], holoplane_latents, sym=sym)  # [B, P, 3*C]


        # Aggregate the three coordinate projections of the merged features.
        features = torch.sum(torch.stack([yz_embed, xz_embed, xy_embed]), dim=0)  # [B, P, 3*C]

        # Features: [B, P, 3*C].

        if self.use_coord:
            # coord shape [b, p, 3]
            batch_size, num_points, _ = coordinates.shape
            input_coor = coordinates.reshape(-1, 3)  / 2.0 + 0.5
            coor_features = self.coor_feature_decoder(input_coor) # [b * p, c]
            coor_features = coor_features.reshape(batch_size, num_points, -1)
            features = features + coor_features

        return self.MLPdecoder(features)

class PropertiesDecoder(nn.Module):
    def __init__(
            self,
            resolution,
            in_channels,
            out_channels,
            global_channels,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1), # 64
            GroupNorm(32),
            nn.SiLU(),

            nn.MaxPool3d(kernel_size=2, stride=2), # 32

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1), # 32
            GroupNorm(64),
            nn.SiLU(),

            nn.MaxPool3d(kernel_size=2, stride=2), # 16

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1), # 16
            GroupNorm(128),
            nn.SiLU(),

            nn.MaxPool3d(kernel_size=2, stride=2), # 8

            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1), # 8
            GroupNorm(128),
            nn.SiLU(),

            nn.AdaptiveAvgPool3d((1, 1, 1)), # 1
        )
        self.fc1 = nn.Linear(128 + global_channels, 96)
        self.fc2 = nn.Linear(96, 64)
        self.fc3 = nn.Linear(64, out_channels)


    def forward(self, displacement, global_features):

        x = self.downsample(displacement)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, global_features], dim=-1)
        x = nn.functional.silu(self.fc1(x))
        x = nn.functional.silu(self.fc2(x))
        x = self.fc3(x)
        return x

class HoloplaneGlobalFeature(nn.Module):
    '''
    HoloplaneGlobalFeature processes the input holoplane latents by applying downsampling and a global average pooling operation.

    - Parameters:
        - in_channels: the number of input channels
        - out_channels: the number of output channels
    '''
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(3 * in_channels, 96, kernel_size=3, stride=2, padding=1), # 32
            GroupNorm(96),
            nn.SiLU(inplace=True),

            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1), # 32
            GroupNorm(96),
            nn.SiLU(inplace=True),

            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1), # 8
            GroupNorm(96),
            nn.SiLU(inplace=True),

            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1), # 32
            GroupNorm(96),
            nn.SiLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)), # 1
        )

        self.out = nn.Linear(96, out_channels)

    def forward(self, holoplane_latents):
        # shape [3, b, c, r, r]
        holoplane_latents = holoplane_latents.permute(1, 0, 2, 3, 4) # [B, 3, C, R, R]
        holoplane_latents = holoplane_latents.reshape(holoplane_latents.shape[0], -1, holoplane_latents.shape[-2], holoplane_latents.shape[-1])

        x = self.downsample(holoplane_latents)  # [B, 96, 1, 1]
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x  # b, 128

class ElasticTensorDecoder(nn.Module):
    def __init__(
            self,
            resolution,
            in_channels,
            mid_channels,
            out_channels,
            global_channels=128,
            local_channels=64,
            aggregate_fn='sum',
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels


        self.global_feature_decoder = HoloplaneGlobalFeature(in_channels, global_channels)
        self.local_feature_decoder = HoloplaneDecoder(
            channels=in_channels,
            out_channels=local_channels,
            aggregate_fn=aggregate_fn,
            use_tanh=False,
            use_coord=True)

        self.pro_decoder = PropertiesDecoder(
            resolution=64,
            in_channels=mid_channels,
            out_channels=out_channels,
            global_channels=global_channels,
        )

        self.fc1 = nn.Linear(global_channels + local_channels, 128)
        self.fc2 = nn.Linear(128, 96)
        self.fc3 = nn.Linear(96, mid_channels)


    def forward(self, holoplane_latents, coordinates):
        global_features = self.global_feature_decoder(holoplane_latents) # b, 128
        global_features_node = global_features.unsqueeze(1).expand(-1, coordinates.shape[1], -1) # b, p, 128
        local_features = self.local_feature_decoder(holoplane_latents, coordinates, sym=1)  # [B, P, local_channels]
        u = torch.cat([global_features_node, local_features], dim=-1)
        u = nn.functional.silu(self.fc1(u))
        u = nn.functional.silu(self.fc2(u))
        u = self.fc3(u) # b, p, 18

        u_input = u.permute(0, 2, 1).reshape(u.shape[0], self.mid_channels, self.resolution, self.resolution, self.resolution)
        x = self.pro_decoder(u_input, global_features)

        return u, x
