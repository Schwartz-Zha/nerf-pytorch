from email.policy import default
import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import logging
import shutil
from tqdm.contrib.logging import logging_redirect_tqdm # corporate logging with tqdm 

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data


##############################################

# For logging the rendering time of a whole image
import timeit
# For logging the model FLOPS
from fvcore.nn import FlopCountAnalysis
# ssim
from skimage.metrics import structural_similarity as ssim
# lpips
import lpips

##############################################



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)    
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def run_transformer_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk = 1024):
    """
    Prepares inputs and applies network 'fn'.
    Unlike the original NeRF implementation, we do not flatten here\
    """
    
    embedded = embed_fn(inputs)
    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        embedded_dirs = embeddirs_fn(input_dirs)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    output = batchify(fn, netchunk)(embedded)    

    return output



def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}

    # DEBUG INFO
    # print('chunk in batchify_rays')
    # print(chunk)
    # print('rays_flat.shape in batchify_rays')
    # print(rays_flat.shape)
    # print(type(rays_flat.shape[0]))

    if rays_flat.shape[0] % chunk == 0:
        for i in range(0, rays_flat.shape[0], chunk):
            ret = render_rays(rays_flat[i:i+chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret
    else:
        remainder = rays_flat.shape[0] % chunk
        for i in range(0, rays_flat.shape[0] - remainder, chunk):
            ret = render_rays(rays_flat[i:i+chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        # Extra operation for last batch
        last_batch = rays_flat[rays_flat.shape[0] - chunk : rays_flat.shape[0]]
        ret = render_rays(last_batch, **kwargs)
        # print(ret.keys())
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            # Discard the useless redundant part

            # DEBUG INFO
            # print('ret[k].shape before discarding')
            # print(ret[k].shape)
            ret[k] = ret[k][chunk - remainder:chunk]
            # print('ret[k].shape after discarding')
            # print(ret[k].shape)
            all_ret[k].append(ret[k])
            
        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret



def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)


    # DEBUG INFO 
    # print('rays.shape in render()')
    # print(rays.shape)    

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        logging.info(str(i)+' ' + str(time.time() - t))
        t = time.time()
        rgb, disp, _, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            logging.info(f'rgb.shape {rgb.shape}, disp.shape {disp.shape}')

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args, logging):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4

    if args.model_type == 'NeRF':
        model = NeRF(D=args.netdepth, W=args.netwidth,
                     input_ch=input_ch, output_ch=output_ch, skips=args.skips,
                     input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    elif args.model_type == 'ResNeRF':
        model = ResNeRF(D=args.netdepth, W=args.netwidth,skip_interval=args.skip_interval,
            input_ch=input_ch, output_ch=output_ch, skips=args.skips,
            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)

    elif args.model_type == 'NeRFFormer':
        model = NeRFFormer(depth=args.transformer_depth, input_dim=input_ch + input_ch_views, 
                        output_dim=4, internal_dim=args.internal_dim,
                        heads=args.heads, dim_head=args.dim_head, mlp_dim=args.mlp_dim).to(device)
    elif args.model_type == 'NeRFViT':
        model = NeRFViT(
                depth=args.transformer_depth, input_dim=input_ch + input_ch_views, 
                output_dim=4, internal_dim=args.internal_dim, heads=args.heads, 
                dim_head=args.dim_head, mlp_dim=args.mlp_dim, rays_seg_num = args.rays_seg_num, 
                pts_seg_num = args.pts_seg_num
            ).to(device)    
    elif args.model_type == 'Conv1d':
        model = NeRFConvNet1d(
            depth=args.netdepth, internal_dim=args.netwidth, input_dim=input_ch+input_ch_views, 
            output_dim=4
        ).to(device)
    elif args.model_type == 'Conv2d':
        model = NeRFConvNet2d(
            depth=args.netdepth, internal_dim=args.netwidth, input_dim=input_ch+input_ch_views, 
            output_dim=4
        ).to(device)
    elif args.model_type == 'ResConv1d':
        model = NeRFResConvNet1d(
            depth=args.netdepth, skip_interval = args.skip_interval, internal_dim=args.netwidth, input_dim=input_ch+input_ch_views, 
            output_dim=4, kernel_size_pt=args.kernel_size, padding_pts=args.padding
        ).to(device)
    elif args.model_type == 'ResConv2d':
        model = NeRFResConvNet2d(
            depth=args.netdepth, skip_interval = args.skip_interval, internal_dim=args.netwidth, input_dim=input_ch+input_ch_views, 
            output_dim=4, kernel_size=args.kernel_size, padding=args.padding
        ).to(device)
    elif args.model_type == 'ResDeformConv2d':
        model = NeRFDeformConv2d(
            depth=args.netdepth, skip_interval = args.skip_interval, internal_dim=args.netwidth, input_dim=input_ch+input_ch_views, 
            output_dim=4, num_rays=1024, num_pts=64, kernel_size=args.kernel_size, padding=tuple(args.padding_2d)
        ).to(device)

    ### Mix Models
    elif args.model_type == 'MLPConv':
        model = MLPConv(
            depth=args.netdepth, internal_dim=args.netwidth, input_dim=input_ch+input_ch_views, 
            output_dim=4
        )
    elif args.model_type == 'ResMLPConv':
        model = ResMLPConv(
            depth=args.netdepth, skip_interval = args.skip_interval, internal_dim=args.netwidth, input_dim=input_ch+input_ch_views, 
            output_dim=4
        ).to(device)
    elif args.model_type == 'MLPFormer':
        model = NeRFMLPFormer(
            mix_mlp_depth = args.mix_mlp_depth, mix_former_depth = args.mix_former_depth, 
            input_dim=input_ch + input_ch_views, output_dim=4, internal_dim=args.internal_dim,
            heads=args.heads, dim_head=args.dim_head, mlp_dim=args.mlp_dim
        ).to(device)
    elif args.model_type == 'Conv1dFormer':
        model = NeRFConv1dFormer(
            mix_conv1d_depth = args.mix_conv1d_depth, kernel_size_pt=args.kernel_size, 
            padding_pts=args.padding, padding_mode=args.padding_mode,
            mix_former_depth=args.mix_former_depth, input_dim=input_ch + input_ch_views, 
            output_dim=4, internal_dim=args.internal_dim,
            heads=args.heads, dim_head=args.dim_head, mlp_dim=args.mlp_dim
        ).to(device)
    elif args.model_type == 'ViG':
        model = NeRFViG(
            k = args.vig_k, blocks = args.vig_blocks, channels = args.vig_channels
        ).to(device)
    elif args.model_type == 'ResConv1dMLP':
        model = ResConvMLP(
            conv_depth=args.conv_depth, mlp_depth = args.mlp_depth, skip_interval = args.skip_interval, internal_dim=args.netwidth, input_dim=input_ch+input_ch_views, 
            output_dim=4, kernel_size_pt=args.kernel_size, padding_pts=args.padding
        ).to(device)
    else:
        sys.exit('model_type not supprted 1, shound be in [NeRF, NeRFFormer, Conv1d, ResConv1d]')
    grad_vars = list(model.parameters())

    # Inspect model parameter size
    
    n_parameter = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'model parameter number: {n_parameter}')


    model_fine = None
    if args.N_importance > 0:
        if args.model_type == 'NeRF':
            model_fine = NeRF(D=args.netdepth, W=args.netwidth,
                        input_ch=input_ch, output_ch=output_ch, skips=args.skips,
                        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        elif args.model_type == 'ResNeRF':
            model_fine = ResNeRF(D=args.netdepth, W=args.netwidth, skip_interval = args.skip_interval,
                        input_ch=input_ch, output_ch=output_ch, skips=args.skips,
                        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)

        elif args.model_type == 'NeRFFormer':
            model_fine = NeRFFormer(depth=args.transformer_depth, input_dim=input_ch + input_ch_views, 
                            output_dim=4, internal_dim=args.internal_dim,
                            heads=args.heads, dim_head=args.dim_head, mlp_dim=args.mlp_dim).to(device)
        elif args.model_type == 'NeRFViT':
            model_fine = NeRFViT(
                depth=args.transformer_depth, input_dim=input_ch + input_ch_views, 
                output_dim=4, internal_dim=args.internal_dim, heads=args.heads, 
                dim_head=args.dim_head, mlp_dim=args.mlp_dim, rays_seg_num = args.rays_seg_num, 
                pts_seg_num = args.pts_seg_num
            ).to(device)    
        elif args.model_type == 'Conv1d':
            model_fine = NeRFConvNet1d(
                depth=args.netdepth, internal_dim=args.netwidth, input_dim=input_ch+input_ch_views, 
                output_dim=4
            ).to(device)
        elif args.model_type == 'Conv2d':
            model_fine = NeRFConvNet2d(
                depth=args.netdepth, internal_dim=args.netwidth, input_dim=input_ch+input_ch_views, 
                output_dim=4
            ).to(device)
        elif args.model_type == 'ResConv1d':
            model_fine = NeRFResConvNet1d(
                depth=args.netdepth, skip_interval = args.skip_interval, internal_dim=args.netwidth, input_dim=input_ch+input_ch_views, 
                output_dim=4, kernel_size_pt=args.kernel_size, padding_pts=args.padding
            ).to(device)
        elif args.model_type == 'ResConv2d':
            model_fine = NeRFResConvNet2d(
                depth=args.netdepth, skip_interval = args.skip_interval, internal_dim=args.netwidth, input_dim=input_ch+input_ch_views, 
                output_dim=4, kernel_size=args.kernel_size, padding=args.padding
            ).to(device)
        elif args.model_type == 'ResDeformConv2d':
            model_fine = NeRFDeformConv2d(
                depth=args.netdepth, skip_interval = args.skip_interval, internal_dim=args.netwidth, input_dim=input_ch+input_ch_views, 
                output_dim=4, num_rays=1024, num_pts=192, kernel_size=args.kernel_size, padding=tuple(args.padding_2d)
            ).to(device)
        elif args.model_type == 'ViG':
            model_fine = NeRFViG(
                k = args.vig_k, blocks = args.vig_blocks, channels = args.vig_channels
            ).to(device) 
        
        # Mix models
        elif args.model_type == 'MLPConv':
            model_fine = MLPConv(
                depth=args.netdepth, internal_dim=args.netwidth, input_dim=input_ch+input_ch_views, 
                output_dim=4
            ).to(device)
        elif args.model_type == 'ResMLPConv':
            model_fine = ResMLPConv(
                depth=args.netdepth, skip_interval = args.skip_interval, internal_dim=args.netwidth, input_dim=input_ch+input_ch_views, 
                output_dim=4
            ).to(device)
        elif args.model_type == 'MLPFormer':
            model_fine = NeRFMLPFormer(
                mix_mlp_depth = args.mix_mlp_depth, mix_former_depth = args.mix_former_depth, 
                input_dim=input_ch + input_ch_views, output_dim=4, internal_dim=args.internal_dim,
                heads=args.heads, dim_head=args.dim_head, mlp_dim=args.mlp_dim
            ).to(device)
        elif args.model_type == 'Conv1dFormer':
            model_fine = NeRFConv1dFormer(
                mix_conv1d_depth = args.mix_conv1d_depth, kernel_size_pt=args.kernel_size, 
                padding_pts=args.padding, padding_mode=args.padding_mode,
                mix_former_depth=args.mix_former_depth, input_dim=input_ch + input_ch_views, 
                output_dim=4, internal_dim=args.internal_dim,
                heads=args.heads, dim_head=args.dim_head, mlp_dim=args.mlp_dim
            ).to(device)
        elif args.model_type == 'ResConv1dMLP':
            model_fine = ResConvMLP(
            conv_depth=args.conv_depth, mlp_depth = args.mlp_depth, skip_interval = args.skip_interval, internal_dim=args.netwidth, input_dim=input_ch+input_ch_views, 
            output_dim=4, kernel_size_pt=args.kernel_size, padding_pts=args.padding
        ).to(device)
        else:
            sys.exit('model_type not supprted 2, shound be in [NeRF, NeRFFormer, Conv1d, ResConv1d]')
        
        grad_vars += list(model_fine.parameters())

    
    if args.model_type == 'NeRF':
        network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)
    elif args.model_type == 'ResNeRF':
        network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)
    elif args.model_type == 'ResConv1dMLP':
        network_query_fn = lambda inputs, viewdirs, network_fn : run_transformer_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=embed_fn,
                                                                    embeddirs_fn=embeddirs_fn,
                                                                    netchunk=args.netchunk)
    elif args.model_type == 'NeRFFormer':                                                            
        network_query_fn = lambda inputs, viewdirs, network_fn : run_transformer_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=embed_fn,
                                                                    embeddirs_fn=embeddirs_fn,
                                                                    netchunk=args.netchunk)
    elif args.model_type == 'NeRFViT':
        network_query_fn = lambda inputs, viewdirs, network_fn : run_transformer_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=embed_fn,
                                                                    embeddirs_fn=embeddirs_fn,
                                                                    netchunk=args.netchunk)
    elif args.model_type == 'Conv1d':
        network_query_fn = lambda inputs, viewdirs, network_fn : run_transformer_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=embed_fn,
                                                                    embeddirs_fn=embeddirs_fn,
                                                                    netchunk=args.netchunk)
    elif args.model_type == 'Conv2d':
        network_query_fn = lambda inputs, viewdirs, network_fn : run_transformer_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=embed_fn,
                                                                    embeddirs_fn=embeddirs_fn,
                                                                    netchunk=args.netchunk)
    elif args.model_type == 'ResConv1d':
        network_query_fn = lambda inputs, viewdirs, network_fn : run_transformer_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=embed_fn,
                                                                    embeddirs_fn=embeddirs_fn,
                                                                    netchunk=args.netchunk)
    elif args.model_type == 'ResConv2d':
        network_query_fn = lambda inputs, viewdirs, network_fn : run_transformer_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=embed_fn,
                                                                    embeddirs_fn=embeddirs_fn,
                                                                    netchunk=args.netchunk)
    elif args.model_type == 'ResDeformConv2d':
        network_query_fn = lambda inputs, viewdirs, network_fn : run_transformer_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=embed_fn,
                                                                    embeddirs_fn=embeddirs_fn,
                                                                    netchunk=args.netchunk)
    elif args.model_type == 'MLPConv':
         network_query_fn = lambda inputs, viewdirs, network_fn : run_transformer_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=embed_fn,
                                                                    embeddirs_fn=embeddirs_fn,
                                                                    netchunk=args.netchunk)
    elif args.model_type == 'ResMLPConv':
        network_query_fn = lambda inputs, viewdirs, network_fn : run_transformer_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=embed_fn,
                                                                    embeddirs_fn=embeddirs_fn,
                                                                    netchunk=args.netchunk)
    elif args.model_type == 'MLPFormer':
        network_query_fn = lambda inputs, viewdirs, network_fn : run_transformer_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=embed_fn,
                                                                    embeddirs_fn=embeddirs_fn,
                                                                    netchunk=args.netchunk)
    elif args.model_type == 'Conv1dFormer':
        network_query_fn = lambda inputs, viewdirs, network_fn : run_transformer_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=embed_fn,
                                                                    embeddirs_fn=embeddirs_fn,
                                                                    netchunk=args.netchunk)
    elif args.model_type == 'ViG':
        network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    else:
        sys.exit('model_type not supprted 3, shound be in [NeRF, NeRFFormer, NeRFViT, Conv1d, ResConv1d, MLPConv, ResMLPConv]')

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    # Optimizer change to AdamW

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    logging.info('Found ckpts' + str(ckpts))
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        logging.info(f'Reloading from {ckpt_path}')
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        logging.info('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.


    ############################### Logging model, model_fine flops

    # macs, params = get_model_complexity_info(model, (1024, 64, 90), as_strings=True, 
    #                     print_per_layer_stat=False, verbose=False)
    # logging.info('{:<30}  {:<8}'.format('model Computational complexity: ', macs))
    # logging.info('{:<30}  {:<8}'.format('model Number of parameters: ', params))

    # macs, params = get_model_complexity_info(model_fine, (1024, 192, 90), as_strings=True, 
    #                     print_per_layer_stat=False, verbose=False)
    # logging.info('{:<30}  {:<8}'.format('model_fine Computational complexity: ', macs))
    # logging.info('{:<30}  {:<8}'.format('model_fine Number of parameters: ', params))
    with torch.no_grad():
        model_flops =  FlopCountAnalysis(model, torch.randn(1024, 64, 90)) #(65536, 90)
        model_fine_flops = FlopCountAnalysis(model_fine, torch.randn(1024, 192, 90)) 
        logging.info(f'model_flops : {model_flops.total()}')
        logging.info(f'model_fine_flops : {model_fine_flops.total()}')

    ############################### 


    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            logging.info(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training option to select model type
    parser.add_argument('--model_type', type=str, default='NeRF',
                        help='model type for implicit density leaning')

    # Training options for sub-batching and gradient aggregation
    parser.add_argument('--aggre_num', type=int, default=1, 
                        help='number of aggregated iteration')

    # Options for resconv1d + MLP
    parser.add_argument('--conv_depth', type=int, default=6, help='num of conv layers')
    parser.add_argument('--mlp_depth', type=int, default=2, help='num of mlp layers')

    # Random Rays Options
    parser.add_argument('--continuous_rays', action='store_true')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument('--skips', nargs='+', type=int, default=[4], 
                        help='skip layer numbers')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    
    # Optional Config keywords for Residual Conv and Residual NeRFFormer
    parser.add_argument("--skip_interval", type=int, default=2, 
                        help='layer number in a residual block')
    
    
    # training options for transformer
    parser.add_argument('--transformer_depth', type=int, default=3, 
                        help='number of tranformer blocks')
    parser.add_argument('--internal_dim', type=int, default=64,
                        help='internal dimension in transformer')
    parser.add_argument('--heads', type=int, default=8, 
                        help='number of heads in transformer')
    parser.add_argument('--dim_head', type=int, default=32, 
                        help='dimension on each attention head')
    parser.add_argument('--mlp_dim', type=int, default=128, 
                        help='dimension size for mlp in the transformer')

    # Additional options for NeRFViT
    parser.add_argument('--rays_seg_num', type=int, default=2)
    parser.add_argument('--pts_seg_num', type=int, default=2)

    # Config for Convolution
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--padding', type=int, default=1)
    parser.add_argument('--padding_mode', type=str, default='replicate')
    parser.add_argument('--padding_2d', nargs='+', type=int, default=[1,1])

    # Config for ViG
    parser.add_argument('--vig_k', type=int, default=3)
    parser.add_argument('--vig_blocks', nargs='+', type=int, default=[1,1,1,1])
    parser.add_argument('--vig_channels', nargs='+', type=int, default=[90,90,90,90])

    # Additional options for Mix models
    parser.add_argument('--mix_mlp_depth', type=int, default=4)
    parser.add_argument('--mix_former_depth', type=int, default=4)
    parser.add_argument('--mix_conv1d_depth', type=int, default=4)
    

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 
    parser.add_argument("--train_iter",type=int, default=200000, 
    help='Total number of training iterations')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Create logger
    basedir = args.basedir
    expname = args.expname
    if not args.render_only:
        os.makedirs(os.path.join(basedir, expname, 'code'), exist_ok=True)
    if args.render_only:

        # Delete previous log
        if os.path.exists(os.path.join(basedir, expname, 'log_render_only.txt')):
            os.remove(os.path.join(basedir, expname, 'log_render_only.txt'))

        logging.basicConfig(filename=os.path.join(basedir, expname, 'log_render_only.txt'), level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    else:
        logging.basicConfig(filename=os.path.join(basedir, expname, 'log.txt'), level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


    # Logging Code Files
    

    if not args.render_only:
        shutil.copyfile('run_nerf.py', os.path.join(basedir, expname, 'code', 'run_nerf.py'))
        shutil.copyfile('run_nerf_helpers.py', os.path.join(basedir, expname, 'code', 'run_nerf_helpers.py'))
        shutil.copyfile('load_blender.py', os.path.join(basedir, expname, 'code','load_blender.py'))
        shutil.copyfile('load_deepvoxels.py', os.path.join(basedir, expname, 'code', 'load_deepvoxels.py'))
        shutil.copyfile('load_LINEMOD.py', os.path.join(basedir, expname, 'code', 'load_LINEMOD.py'))
        shutil.copyfile('load_llff.py', os.path.join(basedir, expname, 'code','load_llff.py'))

    

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        logging.info(f'Loaded llff images.shape{images.shape}, render_poses.shape{render_poses.shape}, \
                     hwf{hwf}, args.datadir{args.datadir}')
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            logging.info(f'Auto LLFF holdout: {args.llffhold}')
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        logging.info('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        logging.info('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        logging.info(f'Loaded blender images.shape{images.shape}, render_poses.shape{render_poses.shape}, \
                     hwf{hwf}, args.datadir{args.datadir}')
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        logging.info(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        logging.info(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        logging.info('Loaded deepvoxelsimages.shape{images.shape}, render_poses.shape{render_poses.shape}, \
                     hwf{hwf}, args.datadir{args.datadir}')
        
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        logging.info(f'Unknown dataset type {args.dataset_type} exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    
    if not args.render_only:
        f = os.path.join(basedir, expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None and not args.render_only:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    # model is actually saved in render_kwargs_train
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args, logging)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    # render_poses.shape = (120, 3, 5)
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        logging.info('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = torch.Tensor(images[i_test])
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            logging.info(f'test poses shape: {render_poses.shape}')

            start_time = timeit.default_timer()
            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            end_time = timeit.default_timer()
            
            
            
            

            if args.render_test:
                # Compute the average mse and psnr on the testset
                assert images.shape[0] == render_poses.shape[0], 'images and render_poses num must match!'
                assert images.shape[0] == rgbs.shape[0], 'images and rgbs num must match!'

                mse_total = 0.
                psnr_total = 0.
                ssim_total = 0.
                lpips_alex_total = 0.
                lpips_vgg_total = 0.

                for i in range(images.shape[0]):
                    rgb = torch.Tensor(rgbs[i])
                    image = images[i]

                    # # DEBUG INFO
                    
                    # print(f'rgb max {torch.max(rgb)}')
                    # print(f'rgb min {torch.min(rgb)}')
                    # print(f'image max {torch.max(image)}')
                    # print(f'image min {torch.min(image)}')
                    # exit(1)


                    # mse psnr ssim lpips 
                    mse = img2mse(rgb, image)
                    psnr = mse2psnr(mse)

                    # ssim
                    loss_ssim = ssim(rgb.cpu().numpy(), image.cpu().numpy(), multichannel=True)

                    # for lpips, normalize to [-1,1] first
                    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
                    loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
                    
                    # DEBUG INFO
                    # print(f'rgb shape {(rgb.shape)}')
                    # print(f'image shape {(image.shape)}')
                    # print(f'rgb shape {(2*(rgb-0.5)).shape}')
                    # print(f'image shape {(2*(image-0.5)).shape}')

                    lpips_alex = loss_fn_alex(2*(torch.moveaxis(rgb, 2, 0) - 0.5), 
                                            2*(torch.moveaxis(image, 2, 0) - 0.5))
                    lpips_vgg = loss_fn_vgg(2*(torch.moveaxis(rgb, 2, 0) - 0.5), 
                                            2*(torch.moveaxis(image, 2, 0) - 0.5))




                    logging.info(f'Rendered Image {i} has MSE: {mse.item()}, \
                        PSNR: {psnr.item()}, SSIM: {loss_ssim.item()}, \
                        lpips_alex: {lpips_alex.item()}, lpips_vgg: {lpips_vgg.item()}')
                    mse_total += mse.item()
                    psnr_total += psnr.item()
                    ssim_total += loss_ssim.item()
                    lpips_alex_total += lpips_alex.item()
                    lpips_vgg_total += lpips_vgg.item()
            
                mse_avg = mse_total / images.shape[0]
                psnr_avg = psnr_total / images.shape[0]
                ssim_avg = ssim_total / images.shape[0]
                lpips_alex_avg = lpips_alex_total / images.shape[0]
                lpips_vgg_avg = lpips_vgg_total / images.shape[0]

                logging.info('Metric calculation completed over the test set.')
                logging.info(f'Over the test set, avg MSE: {mse_avg}, avg PSNR: {psnr_avg}, \
                    avg SSIM: {ssim_avg}, avg alex LPIPS: {lpips_alex_avg}, avg vgg LPIPS: {lpips_vgg_avg}')


                logging.info(f'The lapsed time for rendering is {end_time - start_time}')
                logging.info(f'Avarage rendering time for each image in {render_poses.shape[0]} images is: {(end_time - start_time) / render_poses.shape[0]}')
            
            logging.info(f'Done rendering: {testsavedir}')
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand // args.aggre_num

    # Ignore use_batching option for distributed training
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching 
        logging.info('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        logging.info('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        logging.info('shuffle rays')
        np.random.shuffle(rays_rgb)

        logging.info('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = args.train_iter + 1
    logging.info('Begin')
    logging.info('TRAIN views are '  + str(i_train))
    logging.info('TEST views are ' + str(i_test))
    logging.info('VAL views are ' + str(i_val))

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    for i in trange(start, N_iters):
        with logging_redirect_tqdm():
            time0 = time.time()

            # Sample random ray batch
            if use_batching: # WE do not use batching
                
                '''
                Original Code will cause last batch in an epoch to have different of data
                '''

                # # Random over all images
                # batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
                # batch = torch.transpose(batch, 0, 1)
                # batch_rays, target_s = batch[:2], batch[2]

                # i_batch += N_rand
                # if i_batch >= rays_rgb.shape[0]:
                #     print("Shuffle data after an epoch!")
                #     rand_idx = torch.randperm(rays_rgb.shape[0])
                #     rays_rgb = rays_rgb[rand_idx]
                #     i_batch = 0

                if i_batch + N_rand >= rays_rgb.shape[0]:
                    logging.info("Shuffle data after an epoch!")
                    rand_idx = torch.randperm(rays_rgb.shape[0])
                    rays_rgb = rays_rgb[rand_idx]
                    i_batch = 0
                
                batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
                batch = torch.transpose(batch, 0, 1)
                batch_rays, target_s = batch[:2], batch[2]
                i_batch += N_rand

            else:
                # Random from one image
                img_i = np.random.choice(i_train)
                target = images[img_i]
                target = torch.Tensor(target).to(device)
                pose = poses[img_i, :3,:4]

                if N_rand is not None:
                    rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                    if i < args.precrop_iters:
                        dH = int(H//2 * args.precrop_frac)
                        dW = int(W//2 * args.precrop_frac)
                        coords = torch.stack(
                            torch.meshgrid(
                                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                            ), -1)
                        if i == start:
                            logging.info(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                    else:
                        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                    select_inds = None
                    if args.continuous_rays:
                        ind_start = np.random.choice(coords.shape[0])
                        if (N_rand + ind_start) < coords.shape[0]:
                            select_inds = np.arange(ind_start, ind_start+N_rand, 1)
                        else:
                            select_inds = np.arange(ind_start, coords.shape[0], 1)
                            np.append(select_inds, np.arange(0, N_rand - select_inds.size))
                    else:    
                        select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                    select_coords = coords[select_inds].long()  # (N_rand, 2)
                    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    batch_rays = torch.stack([rays_o, rays_d], 0)
                    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            #####  Core optimization loop  #####
            rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                    verbose=i < 10, retraw=True,
                                                    **render_kwargs_train)


            # optimizer.zero_grad()
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][...,-1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)

            loss.backward()
            # optimizer.step()

            # if i % args.aggre_num == 0: # Cancel this for logging learning rate
            optimizer.step()
            optimizer.zero_grad()
            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            # Iter 200000
            # Dataset 100

            
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000  #// 250 * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            ################################

            

            # dt = time.time()-time0
            # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
            #####           end            #####

            # Rest is logging
            if i%args.i_weights==0:

                path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                logging.info(f'Saved checkpoints at {path}')

            if i%args.i_video==0 and i > 0:

                # Turn on testing mode
                with torch.no_grad():
                    start_time = timeit.default_timer()
                    rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
                    end_time = timeit.default_timer()
                    logging.info(f'The lapsed time for rendering is {end_time - start_time}')
                logging.info(f'Done, saving rgbs.shape{rgbs.shape}, disps.shape{disps.shape}')
                moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
                imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
                imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

                # if args.use_viewdirs:
                #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
                #     with torch.no_grad():
                #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
                #     render_kwargs_test['c2w_staticcam'] = None
                #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

            if i%args.i_testset==0 and i > 0:
                testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
                os.makedirs(testsavedir, exist_ok=True)
                logging.info(f'test poses shape {poses[i_test].shape}')
                with torch.no_grad():
                    render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
                logging.info('Saved test set')


        
            if i%args.i_print==0:
                logging.info(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} Learning Rate: {new_lrate}")
            """
                print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
                print('iter time {:.05f}'.format(dt))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                    tf.contrib.summary.scalar('loss', loss)
                    tf.contrib.summary.scalar('psnr', psnr)
                    tf.contrib.summary.histogram('tran', trans)
                    if args.N_importance > 0:
                        tf.contrib.summary.scalar('psnr0', psnr0)


                if i%args.i_img==0:

                    # Log a rendered validation view to Tensorboard
                    img_i=np.random.choice(i_val)
                    target = images[img_i]
                    pose = poses[img_i, :3,:4]
                    with torch.no_grad():
                        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                            **render_kwargs_test)

                    psnr = mse2psnr(img2mse(rgb, target))

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                        tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                        tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                        tf.contrib.summary.scalar('psnr_holdout', psnr)
                        tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                    if args.N_importance > 0:

                        with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                            tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                            tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                            tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
            """

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
