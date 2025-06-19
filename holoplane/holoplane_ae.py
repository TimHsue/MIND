# inputs  [128, 128, 128]
# latent  [128, 128, 32, 3]
# outputs occupancy [num_p, 4]


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # self.norm1 = nn.LayerNorm(norm_shape) if use_norm else nn.Identity()
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
        # print("x shape", x.shape)
        identity = x

        x = self.conv1(nn.functional.silu(self.norm0(x)))

        if emb is not None:
            params = self.affine(emb).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(x.dtype)
            scale, shift = params.chunk(chunks=2, dim=1)
            x = nn.functional.silu(torch.addcmul(shift, self.norm1(x), scale + 1))

        x = self.conv2(x)
        # print("conv2 shape", out.shape)
        # print(self.norm1)
        x = self.norm2(x)
        x = nn.functional.silu(x)
        # out = self.act(out)
        
        identity = self.downsample(identity)
        # print("identity shape", identity.shape)
        
        x += identity
        # print("out shape", out.shape)
        if not self.last_layer:
            x = nn.functional.silu(x)
        
        return x

class VoxelProjector(nn.Module):
    '''
    VoxelProjection applies convolutions along the specified dimension to reduce its resolution by half.
    
    - Parameters:
        - dim: the dimension to downsample (0 for x, 1 for y, 2 for z)
        - in_channels: the number of input channels
        - depth: the number of downsampling operations to perform
    - Input shape: [b, in_channels, r, r, r]
    - Output shape: [b, out_chanels, r/2**depth, r, r]
    '''
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
            # residual = x
            x = layer(x, emb_labels)
            # x += residual
        
        B, C, H, W, D = x.shape
        # x shape [b, out_channels, 1, r, r] / [b, out_channels, r, 1, r] / [b, out_channels, r, r, 1]
        # print(x.shape)
        x = x.reshape(B, C, self.out_resolution, self.out_resolution)
        # x shape [b, out_channels, r, r]
            
        return x

class LabelFourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, label_dim, scale=10):
        super().__init__()
        # self.register_buffer('freqs', torch.randn(num_channels * 2) * scale)  # Remove register_buffer for simplicity
        self.freqs = nn.Parameter(torch.randn(num_channels) * scale, requires_grad=False) # shape: [num_channels * 2]
        self.out = torch.nn.Linear(label_dim * num_channels * 2, num_channels)

    def forward(self, x):
        # x: [batch_size, label_dim]
        batch_size = x.shape[0]
        freqs = self.freqs  # [num_channels // 2]
        x = x.unsqueeze(2)  # [batch_size, label_dim, 1]
        freqs = freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, num_channels // 2]
        x = x * (2 * np.pi * freqs)  # Element-wise multiplication
        x = torch.cat([x.cos(), x.sin()], dim=2)  # [batch_size, label_dim, num_channels]
        x = x.view(batch_size, -1)  # Flatten to [batch_size, label_dim * num_channels]
        x = self.out(x)  # Final linear layer
        return x

class TriplaneEncoder(nn.Module):
    '''
    TriplaneEncoder processes the input 3D volume by applying downsampling along each axis and concatenating
    the results into a unified latent representation.
    
    - Parameters:
        - resolution: the resolution of the input volume
        - depth: the depth of convolutions (i.e., how many times to halve the resolution along each axis)
        - device: the device to use
    '''
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
        assert label_dim == len(label_scale)
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
        
        # Concatenate along the first dimension (channel dimension)
        latent = [plane_yz, plane_xz, plane_xy]
        
        latent = torch.stack(latent, dim=0)  # shape: [3, b, oc, or, or]
        
        return latent

def first_layer_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            # if hasattr(m, 'weight'):
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init

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

class TriplaneDecoder(nn.Module):
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

        self.aggregate_fn = aggregate_fn
        # print(f'Using aggregate_fn {aggregate_fn}')

        self.channels = channels
        self.use_tanh = use_tanh
        self.use_coord = use_coord
        self.device = device

        MLP_in_channels = channels * 9 if aggregate_fn == 'cat' else channels * 3

        if use_coord:
            self.coor_feature_decoder = LabelFourierEmbedding(label_dim=3, num_channels=MLP_in_channels, scale=10)
        self.MLPdecoder = ResidualMLP(MLP_in_channels, out_channels)

        # init parameters
        # self.net.apply(frequency_init(30))
        # self.net[0].apply(first_layer_sine_init)
        
    def sample_plane_train(self, coords2d, plane):
        # plane shape [b, c, r, r]
        # coords2d shape [b, p, 2]
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features
     
    def sample_plane_vali(self, coords2d, plane, sym=48):
        assert len(coords2d.shape) == 3, coords2d.shape

        if sym == 8 or sym == 48:
            plane_flipped_x = torch.flip(plane, dims=[2]) 
            plane_flipped_y = torch.flip(plane, dims=[3]) 
            plane_flipped_xy = torch.flip(plane, dims=[2, 3])  

        if sym == 48:
            plane_swap_x_y = plane.permute(0, 1, 3, 2)  
            plane_flipped_x_swap_x_y = torch.flip(plane_swap_x_y, dims=[2])  
            plane_flipped_y_swap_x_y = torch.flip(plane_swap_x_y, dims=[3]) 
            plane_flipped_xy_swap_x_y = torch.flip(plane_swap_x_y, dims=[2, 3])  

        if sym == 1:
            symmetric_plane = plane
        if sym == 8:
            symmetric_plane = (plane + plane_flipped_x + plane_flipped_y + plane_flipped_xy) / 4
        elif sym == 48: 
            symmetric_plane = (plane + plane_flipped_x + plane_flipped_y + plane_flipped_xy + plane_swap_x_y + plane_flipped_x_swap_x_y + plane_flipped_y_swap_x_y + plane_flipped_xy_swap_x_y) / 8

        sampled_features = torch.nn.functional.grid_sample(
            symmetric_plane,
            coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
            mode='bilinear', padding_mode='zeros', align_corners=True
        )

        N, C, H, W = sampled_features.shape

        sampled_features = sampled_features.reshape(N, C, H * W).permute(0, 2, 1)

        return sampled_features


    def forward(self, triplane_latents, coordinates, sym=48):


        triplane_latents = triplane_latents.permute(1, 0, 2, 3, 4)
        triplane_latents = triplane_latents.reshape(triplane_latents.shape[0], -1, triplane_latents.shape[-2], triplane_latents.shape[-1])

        if self.use_tanh:
            for triplane in triplane_latents:
                triplane = torch.tanh(triplane)

        if self.training:
            sample_plane = self.sample_plane_vali
        else:
            sample_plane = self.sample_plane_vali

        yz_embed = sample_plane(coordinates[..., 1:3], triplane_latents, sym=sym)
        xz_embed = sample_plane(coordinates[..., [0, 2]], triplane_latents, sym=sym)
        xy_embed = sample_plane(coordinates[..., 0:2], triplane_latents, sym=sym)  
        
        if self.aggregate_fn == 'sum':
            features = torch.sum(torch.stack([yz_embed, xz_embed, xy_embed]), dim=0) # after stack [3, batch_size, P, 32] # after sum [batch_size, 20000, 64] # batch_size, h*w, channels
        elif self.aggregate_fn == 'prod':
            features = torch.prod(torch.stack([yz_embed, xz_embed, xy_embed]), dim=0)
        elif self.aggregate_fn == 'cat':
            features = torch.cat([yz_embed, xz_embed, xy_embed], dim=-1)

        if self.use_coord:
            batch_size, num_points, _ = coordinates.shape
            input_coor = coordinates.reshape(-1, 3)  / 2.0 + 0.5
            coor_features = self.coor_feature_decoder(input_coor)
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
            noise_level=0.02,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.noise_level = noise_level

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

    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_level
        return x + noise


    def forward(self, displacement, global_features):

        x = self.downsample(displacement)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, global_features], dim=-1)
        x = nn.functional.silu(self.fc1(x))
        x = nn.functional.silu(self.fc2(x))
        x = self.fc3(x)
        return x

class TriplaneGlobalFeature(nn.Module):
    '''
    TriplaneGlobalFeature processes the input triplane latents by applying downsampling and a global average pooling operation.

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
            nn.Conv2d(3 * in_channels, 96, kernel_size=3, stride=2, padding=1),
            GroupNorm(96),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            GroupNorm(96),
            nn.SiLU(inplace=True),

            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1),
            GroupNorm(96),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            GroupNorm(96),
            nn.SiLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)), # 1
        )

        self.out = nn.Linear(96, out_channels)

    def forward(self, triplane_latents):
        triplane_latents = triplane_latents.permute(1, 0, 2, 3, 4)
        triplane_latents = triplane_latents.reshape(triplane_latents.shape[0], -1, triplane_latents.shape[-2], triplane_latents.shape[-1])

        x = self.downsample(triplane_latents) 
        x = x.view(x.size(0), -1)
        x = self.out(x)     
        return x 

class ConstitutiveMatrixDecoder(nn.Module):
    def __init__(
            self,
            resolution,
            in_channels,
            mid_channels,
            out_channels,
            global_channels=128,
            local_channels=64,
            aggregate_fn='sum',
            noise_level=0.02,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.noise_level = noise_level


        self.global_feature_decoder = TriplaneGlobalFeature(in_channels, global_channels)
        self.local_feature_decoder = TriplaneDecoder(
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
            noise_level=noise_level,
        )

        self.fc1 = nn.Linear(global_channels + local_channels, 128)
        self.fc2 = nn.Linear(128, 96)
        self.fc3 = nn.Linear(96, mid_channels)

    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_level
        return x + noise

    def forward(self, triplane_latents, coordinates):
        global_features = self.global_feature_decoder(triplane_latents)
        global_features_node = global_features.unsqueeze(1).expand(-1, coordinates.shape[1], -1)
        local_features = self.local_feature_decoder(triplane_latents, coordinates, sym=1)
        u = torch.cat([global_features_node, local_features], dim=-1)
        u = nn.functional.silu(self.fc1(u))
        u = nn.functional.silu(self.fc2(u))
        u = self.fc3(u)
        
        u_input = u.permute(0, 2, 1).reshape(u.shape[0], self.mid_channels, self.resolution, self.resolution, self.resolution)
        x = self.pro_decoder(u_input, global_features)

        return u, x
