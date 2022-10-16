import enum
from math import sqrt
from turtle import forward, width
import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import time


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):

        # print('DEBUG in NeRF class')
        # print('Inspect network input shape')
        # print(x.shape)
        # print('self.input_ch ')
        # print(self.input_ch)
        # print('self.input_ch_views')
        # print(self.input_ch_views)

        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts

        # print('input_pts shape')
        # print(input_pts.shape)

        # print('input_views.shape')
        # print(input_views.shape)


        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        # print('h.shape')
        # print(h.shape)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        # print('NeRF output shape')
        # print(outputs.shape)    
        return outputs    

    # def load_weights_from_keras(self, weights):
    #     assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
    #     # Load pts_linears
    #     for i in range(self.D):
    #         idx_pts_linears = 2 * i
    #         self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
    #         self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
    #     # Load feature_linear
    #     idx_feature_linear = 2 * self.D
    #     self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
    #     self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

    #     # Load views_linears
    #     idx_views_linears = 2 * self.D + 2
    #     self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
    #     self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

    #     # Load rgb_linear
    #     idx_rbg_linear = 2 * self.D + 4
    #     self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
    #     self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

    #     # Load alpha_linear
    #     idx_alpha_linear = 2 * self.D + 6
    #     self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
    #     self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



class NeRFFormer(nn.Module):

    def __init__(self, depth = 8, input_dim = 90,  output_dim= 4, internal_dim = 64,
                heads=8, dim_head = 32, mlp_dim = 128):

        super(NeRFFormer, self).__init__()

        self.preprocessor = nn.Sequential(
            nn.Linear(input_dim, internal_dim),
            nn.ReLU()
        )
        self.transformer = Transformer(internal_dim, depth, heads, dim_head, mlp_dim, dropout = 0.)

        self.postprocessor = nn.Sequential(
            nn.Linear(internal_dim, output_dim)
        )
        return

    def forward(self, x):

        x = self.preprocessor(x)
        x = self.transformer(x)
        x = self.postprocessor(x)

        return x





class NeRFViT(nn.Module):
    def __init__(self, depth = 8,  input_dim = 90,  output_dim= 4, internal_dim = 64,
                heads=8, dim_head = 32, mlp_dim = 128, rays_seg_num =2, pts_seg_num = 2):
        super(NeRFViT, self).__init__()

        # patch_dim = input_dim * rays_seg_num * pts_seg_num

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('(p1 rays_seg_num) (p2 pts_seg_num) c -> p1 p2 (rays_seg_num pts_seg_num c)',
        #               rays_seg_num = rays_seg_num, pts_seg_num = pts_seg_num),
        #     nn.Linear(patch_dim, internal_dim * rays_seg_num, pts_seg_num)
        # )

        # self.pos_embedding = nn.Parameter(torch.randn(rays_num // rays_seg_num, pts_num // pts_seg_num, 
        #                                     patch_dim))

        self.rays_seg_num = rays_seg_num
        self.pts_seg_num = pts_seg_num
                
        self.preprocessor = nn.Sequential(
            nn.Linear(input_dim * rays_seg_num * pts_seg_num, internal_dim),
            nn.ReLU()
        )
        self.transformer = Transformer(internal_dim, depth, heads, dim_head, mlp_dim, dropout = 0.)

        self.postprocessor = nn.Sequential(
            nn.Linear( int(internal_dim / (rays_seg_num * pts_seg_num)), output_dim)
        )
        return

    def forward(self, x):

        # Patch embedding
        assert x.shape[0] % self.rays_seg_num == 0, 'ray dim must be divisible by rays_seg_num'
        assert x.shape[1] % self.pts_seg_num == 0, 'pts dim must be divisible by pts_seg_num'
        x = rearrange(x, '(r rs) (p ps) w -> r p (rs ps w)', rs = self.rays_seg_num, ps=self.pts_seg_num)

        x = self.preprocessor(x)
        x = self.transformer(x)

        # Reverse patch embedding
        x = rearrange(x, 'r p (rs ps w) -> (r rs) (p ps) w', rs = self.rays_seg_num, ps=self.pts_seg_num)
        
        x = self.postprocessor(x)
        
        return x





class NeRFFormerSplit(nn.Module):
    def __init__(self, depth = 8, alpha_depth = 4,  input_dim = 90,  output_dim= 4, internal_dim = 64,
                heads=8, dim_head = 32, mlp_dim = 128):
        
        super(NeRFFormerSplit, self).__init__()
        assert alpha_depth < depth

        self.preprocessor = nn.Sequential(
            nn.Linear(input_dim, internal_dim),
            nn.ReLU()
        )
        self.transformer1 = Transformer(internal_dim, alpha_depth, heads, dim_head, mlp_dim, dropout=0.)
        self.transformer2 = Transformer(internal_dim, depth - alpha_depth, heads, dim_head, mlp_dim, dropout = 0.)

        self.alpha_processor = nn.Sequential(
            nn.Linear(internal_dim, 1)
        )

        self.rgb_processor = nn.Sequential(
            nn.Linear(internal_dim, output_dim - 1)
        )
        return

    def forward(self, x):

        x = self.preprocessor(x)
        x = self.transformer1(x)
        
        alpha = self.alpha_processor(x)
        
        x = self.transformer2(x)
        rgb = self.rgb_processor(x)

        out = torch.cat([alpha, rgb], -1)
        return out





class NeRFConvNet1d(nn.Module):
    def __init__(self, depth = 8, input_dim = 90,  internal_dim = 256, output_dim= 4, 
                 kernel_size_pt = 3, padding_pts = 1, padding_mode='replicate') -> None:
        super(NeRFConvNet1d, self).__init__()
        self.pre_processor = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=internal_dim, 
                        kernel_size=kernel_size_pt, padding=padding_pts, bias=True, 
                        padding_mode=padding_mode),
            nn.ReLU()
        )
        
        self.conv_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels=internal_dim, out_channels=internal_dim, 
                        kernel_size=kernel_size_pt, padding=padding_pts, padding_mode=padding_mode),
                        nn.ReLU()
                ) for i in range(depth)
            ]
        )

        self.post_processor = nn.Sequential(
            nn.Conv1d(in_channels=internal_dim, out_channels=output_dim, 
                        kernel_size=kernel_size_pt, padding=padding_pts, padding_mode=padding_mode)
        )

    def forward(self, x):

        # Move Axis to coporate into Conv1d 
        x = torch.moveaxis(x, 2, 1)

        x = self.pre_processor(x)
        for _, module in enumerate(self.conv_list):
            x = module(x)
        
        x = self.post_processor(x)


        x = torch.moveaxis(x, 1, 2)

        return x

class NeRFConvNet2d(nn.Module):
    def __init__(self, depth = 8, input_dim = 90,  internal_dim = 256, output_dim= 4, 
                 kernel_size = 3, padding = 1, padding_mode='replicate') -> None:
        super(NeRFConvNet2d, self).__init__()
        self.pre_processor = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=internal_dim, 
                        kernel_size=kernel_size, padding=padding, bias=True, 
                        padding_mode=padding_mode),
            nn.ReLU()
        )
        
        self.conv_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels=internal_dim, out_channels=internal_dim, 
                        kernel_size=kernel_size, padding=padding, padding_mode=padding_mode),
                        nn.ReLU()
                ) for i in range(depth)
            ]
        )

        self.post_processor = nn.Sequential(
            nn.Conv2d(in_channels=internal_dim, out_channels=output_dim, 
                        kernel_size=kernel_size, padding=padding, padding_mode=padding_mode)
        )

    def forward(self, x):

        # Move Axis to coporate into Conv1d 
        x = torch.moveaxis(x, 2, 0)

        x = self.pre_processor(x)
        for _, module in enumerate(self.conv_list):
            x = module(x)
        
        x = self.post_processor(x)


        x = torch.moveaxis(x, 0, 2)

        return x


class NeRFResConvNet1d(nn.Module):
    def __init__(self, depth = 8, skip_interval = 2, input_dim = 90,  internal_dim = 256, output_dim= 4, 
                 kernel_size_pt = 3, padding_pts = 1, padding_mode='replicate') -> None:
        super(NeRFResConvNet1d, self).__init__()
        assert depth % skip_interval == 0, "depth must be divisible by skip_interval"
        self.pre_processor = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=internal_dim, 
                        kernel_size=kernel_size_pt, padding=padding_pts, bias=True, 
                        padding_mode=padding_mode),
            nn.ReLU()
        )
        self.skip_interval = skip_interval
        
        self.conv_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels=internal_dim, out_channels=internal_dim, 
                        kernel_size=kernel_size_pt, padding=padding_pts, padding_mode=padding_mode),
                        nn.ReLU()
                ) for i in range(depth)
            ]
        )

        self.post_processor = nn.Sequential(
            nn.Conv1d(in_channels=internal_dim, out_channels=output_dim, 
                        kernel_size=kernel_size_pt, padding=padding_pts, padding_mode=padding_mode)
        )
        return 

    def forward(self, x):

        # Move Axis to coporate into Conv1d 
        x = torch.moveaxis(x, 2, 1)

        x = self.pre_processor(x)
        for i, module in enumerate(self.conv_list):
            if i % self.skip_interval == 0:
                x = x + module(x)
            else:
                x = module(x)
        
        x = self.post_processor(x)


        x = torch.moveaxis(x, 1, 2)

        return x


class NeRFResConvNet2d(nn.Module):
    def __init__(self, depth = 8, skip_interval = 2, input_dim = 90,  internal_dim = 256, output_dim= 4, 
                 kernel_size = 3, padding = 1, padding_mode='replicate') -> None:
        super(NeRFResConvNet2d, self).__init__()
        assert depth % skip_interval == 0, "depth must be divisible by skip_interval"
        self.skip_interval = skip_interval
        self.pre_processor = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=internal_dim, 
                        kernel_size=kernel_size, padding=padding, bias=True, 
                        padding_mode=padding_mode),
            nn.ReLU()
        )
        
        self.conv_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels=internal_dim, out_channels=internal_dim, 
                        kernel_size=kernel_size, padding=padding, padding_mode=padding_mode),
                        nn.ReLU()
                ) for i in range(depth)
            ]
        )

        self.post_processor = nn.Sequential(
            nn.Conv2d(in_channels=internal_dim, out_channels=output_dim, 
                        kernel_size=kernel_size, padding=padding, padding_mode=padding_mode)
        )

    def forward(self, x):

        # Move Axis to coporate into Conv1d 
        x = torch.moveaxis(x, 2, 0)

        x = self.pre_processor(x)
        for i, module in enumerate(self.conv_list):
            if i % self.skip_interval == 0:
                x = x + module(x)
            else:
                x = module(x)
        
        x = self.post_processor(x)


        x = torch.moveaxis(x, 0, 2)

        return x




#### Mix Models


# MLP + Conv1d
class MLPConv(nn.Module):
    def __init__(self, depth = 8, input_dim = 90,  internal_dim = 256, output_dim= 4, 
                 kernel_size_pt = 3, padding_pts = 1, padding_mode='replicate') -> None:
        super(MLPConv, self).__init__()
        self.pre_processor = nn.Sequential(
            nn.Linear(input_dim, internal_dim),
            nn.ReLU()
        )
        
        self.mod_list = nn.ModuleList()
        for i in range(depth):
            if i % 2 == 0:
                self.mod_list.append(
                    nn.Sequential(
                        nn.Linear(internal_dim, internal_dim),
                        nn.ReLU()
                    )
                )
            else:
                self.mod_list.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels=internal_dim, out_channels=internal_dim, 
                                    kernel_size=kernel_size_pt, padding=padding_pts, padding_mode=padding_mode),
                        nn.ReLU()
                    )
                )


        self.post_processor = nn.Sequential(
            nn.Linear(internal_dim, output_dim)
        )
        return 

    def forward(self, x):

        x = self.pre_processor(x)
        for i, module in enumerate(self.mod_list):
            if i % 2 == 0:
                x = module(x)
            else:
                # Move axis for 1D Conv to work properly
                x = torch.moveaxis(x, 2, 1)
                x = module(x)
                x = torch.moveaxis(x, 1, 2)
        
        x = self.post_processor(x)

        return x

# Conv1d
class ResMLPConv(nn.Module):
    def __init__(self, depth = 8, skip_interval = 2, input_dim = 90,  internal_dim = 256, output_dim= 4, 
                 kernel_size_pt = 3, padding_pts = 1, padding_mode='replicate') -> None:
        super(ResMLPConv, self).__init__()
        assert depth % skip_interval == 0, "depth must be divisible by skip_interval"
        self.skip_interval = skip_interval
        self.pre_processor = nn.Sequential(
            nn.Linear(input_dim, internal_dim),
            nn.ReLU()
        )
        
        self.mod_list = nn.ModuleList()
        for i in range(depth):
            if i % skip_interval == 0:
                self.mod_list.append(
                    nn.Sequential(
                        nn.Linear(internal_dim, internal_dim),
                        nn.ReLU()
                    )
                )
            else:
                self.mod_list.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels=internal_dim, out_channels=internal_dim, 
                                    kernel_size=kernel_size_pt, padding=padding_pts, padding_mode=padding_mode),
                        nn.ReLU()
                    )
                )


        self.post_processor = nn.Sequential(
            nn.Linear(internal_dim, output_dim)
        )
        return 

    def forward(self, x):

        x = self.pre_processor(x)
        for i, module in enumerate(self.mod_list):
            if i % self.skip_interval == 0:
                x = x + module(x)
            else:
                # Move axis for 1D Conv to work properly
                x = torch.moveaxis(x, 2, 1)
                x = module(x)
                x = torch.moveaxis(x, 1, 2)
        
        x = self.post_processor(x)

        return x



class NeRFMLPFormer(nn.Module):

    def __init__(self, mix_mlp_depth = 4, mix_former_depth = 4, input_dim = 90,  output_dim= 4, internal_dim = 64,
                heads=8, dim_head = 32, mlp_dim = 128):

        super(NeRFMLPFormer, self).__init__()

        self.preprocessor = nn.Sequential(
            nn.Linear(input_dim, internal_dim),
            nn.ReLU()
        )

        self.mlp = nn.ModuleList()
        for _ in range(mix_mlp_depth):
            self.mlp.append(
                nn.Sequential(
                    nn.Linear(internal_dim, internal_dim),
                    nn.ReLU()
                )
            )


        self.transformer = Transformer(internal_dim, mix_former_depth, heads, dim_head, mlp_dim, dropout = 0.)

        self.postprocessor = nn.Sequential(
            nn.Linear(internal_dim, output_dim)
        )
        return

    def forward(self, x):

        x = self.preprocessor(x)
        for _, module in enumerate(self.mlp):
            x = module(x)
        x = self.transformer(x)
        x = self.postprocessor(x)

        return x



class NeRFConv1dFormer(nn.Module):
    def __init__(self, mix_conv1d_depth = 4, kernel_size_pt = 3, padding_pts = 1, padding_mode='replicate',
                 mix_former_depth = 4, input_dim = 90,  output_dim= 4, internal_dim = 64,
                 heads=8, dim_head = 32, mlp_dim = 128):

        super(NeRFConv1dFormer, self).__init__()

        self.preprocessor = nn.Sequential(
            nn.Linear(input_dim, internal_dim),
            nn.ReLU()
        )

        self.conv = nn.ModuleList()
        for _ in range(mix_conv1d_depth):
            self.conv.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=internal_dim, out_channels=internal_dim, 
                            kernel_size=kernel_size_pt, padding=padding_pts, padding_mode=padding_mode),
                    nn.ReLU()
                )
            )


        self.transformer = Transformer(internal_dim, mix_former_depth, heads, dim_head, mlp_dim, dropout = 0.)

        self.postprocessor = nn.Sequential(
            nn.Linear(internal_dim, output_dim)
        )
        return

    def forward(self, x):

        x = self.preprocessor(x)

        # Move axis for 1D Conv to work properly
        x = torch.moveaxis(x, 2, 1)
        for _, module in enumerate(self.conv):
            x = module(x)
        x = torch.moveaxis(x, 1, 2)
        
        x = self.transformer(x)
        x = self.postprocessor(x)

        return x




# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


# Parallel Card
# Gradient Aggregation
# Alpha RGB Split
# Synthetic Data
# Internal Dim Upto 256
# Test on training data
# 1D, 2D Convolution, small network