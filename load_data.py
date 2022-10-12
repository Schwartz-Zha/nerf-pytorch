import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data
import sys

 # Load data
def load_date(agrs, logging):
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

    return images, poses, render_poses, hwf, K, near, far, i_train, i_val, i_test



class SyntheticDataset(Dataset):
    def __init__(args, logging):
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
            sys.exit(1)

        
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

        

        # After the processing
        self.images = images
        self.poses = poses
        self.render_poses = render_poses
        self.hwf = hwf
        self.i_train = i_train
        self.i_val = i_val
        self.i_test = i_test
        self.K = K
