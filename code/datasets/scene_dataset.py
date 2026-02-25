import os
import torch
import numpy as np
from PIL import Image

import utils.general as utils
from utils import rend_util

class SceneDataset(torch.utils.data.Dataset):
    """
    Scene dataset supporting single-frame (original) or multi-frame mode.
    
    Multi-frame mode: each image folder contains frame_XXXXX subfolders.
    """

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 obj_pose_path=None,
                 frame_skip=1,
                 ):

        self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.frame_skip = frame_skip

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        self.sampling_size = -1

        image_dir = '{0}/image'.format(self.instance_dir)
        
        # Detect multi-frame vs single-frame structure
        frame_dirs = sorted([d for d in os.listdir(image_dir) 
                            if os.path.isdir(os.path.join(image_dir, d)) and d.startswith('frame_')])
        
        if len(frame_dirs) > 0:
            # Multi-frame mode
            self.multi_frame = True
            self.frame_dirs = frame_dirs[::frame_skip]
            self.n_frames = len(self.frame_dirs)
            
            # Get cameras from first frame
            first_frame = os.path.join(image_dir, self.frame_dirs[0])
            camera_images = sorted(utils.glob_imgs(first_frame))
            self.n_cameras = len(camera_images)
            self.n_images = self.n_frames * self.n_cameras
            
            print(f"Multi-frame mode: {self.n_frames} frames × {self.n_cameras} cameras")
        else:
            # Single-frame mode (original)
            self.multi_frame = False
            self.n_frames = 1
            image_paths = sorted(utils.glob_imgs(image_dir))
            self.n_cameras = len(image_paths)
            self.n_images = self.n_cameras

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_cameras)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_cameras)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        # Load images
        self.rgb_images = []
        self.frame_indices = []  # Track embedding index (0 to n_frames-1)
        self.actual_frame_numbers = []  # Track actual frame numbers from filenames
        
        if self.multi_frame:
            for embed_idx, frame_dir in enumerate(self.frame_dirs):
                frame_path = os.path.join(image_dir, frame_dir)
                image_paths = sorted(utils.glob_imgs(frame_path))
                actual_frame = int(frame_dir.replace('frame_', ''))
                for path in image_paths:
                    rgb = rend_util.load_rgb(path)
                    rgb = rgb.reshape(3, -1).transpose(1, 0)
                    self.rgb_images.append(torch.from_numpy(rgb).float())
                    # Use embedding index (0, 1, 2, ...) not actual frame number
                    self.frame_indices.append(embed_idx)
                    self.actual_frame_numbers.append(actual_frame)
        else:
            image_paths = sorted(utils.glob_imgs(image_dir))
            for path in image_paths:
                rgb = rend_util.load_rgb(path)
                rgb = rgb.reshape(3, -1).transpose(1, 0)
                self.rgb_images.append(torch.from_numpy(rgb).float())
                self.frame_indices.append(0)
                self.actual_frame_numbers.append(0)
        
        # Load object poses if provided
        self.obj_poses = None
        if obj_pose_path and os.path.exists(obj_pose_path):
            self.obj_poses = np.load(obj_pose_path)
            print(f"Loaded object poses: {self.obj_poses.shape}")

        # Load human masks - pixels to EXCLUDE from sampling
        human_mask_dir = '{0}/human_mask'.format(self.instance_dir)
        self.human_mask_indices = []  # For each image, indices of pixels that are NOT human
        if os.path.exists(human_mask_dir):
            if self.multi_frame:
                # Multi-frame: load masks for each frame
                for frame_dir in self.frame_dirs:
                    mask_frame_dir = os.path.join(human_mask_dir, frame_dir)
                    if os.path.exists(mask_frame_dir):
                        mask_paths = sorted(utils.glob_imgs(mask_frame_dir))
                        for path in mask_paths:
                            mask = np.array(Image.open(path).convert('L'))
                            if mask.shape != (self.img_res[0], self.img_res[1]):
                                mask = np.array(Image.fromarray(mask).resize(
                                    (self.img_res[1], self.img_res[0]), Image.NEAREST))
                            mask_flat = mask.reshape(-1)
                            valid_indices = np.where(mask_flat < 127)[0]
                            self.human_mask_indices.append(torch.from_numpy(valid_indices).long())
            else:
                # Single-frame mode
                mask_paths = sorted(utils.glob_imgs(human_mask_dir))
                if len(mask_paths) == self.n_images:
                    print(f'Loading {len(mask_paths)} human masks from {human_mask_dir}')
                    for path in mask_paths:
                        mask = np.array(Image.open(path).convert('L'))
                        if mask.shape != (self.img_res[0], self.img_res[1]):
                            mask = np.array(Image.fromarray(mask).resize(
                                (self.img_res[1], self.img_res[0]), Image.NEAREST))
                        mask_flat = mask.reshape(-1)
                        valid_indices = np.where(mask_flat < 127)[0]
                        self.human_mask_indices.append(torch.from_numpy(valid_indices).long())
            if len(self.human_mask_indices) > 0:
                avg_valid = np.mean([len(m) for m in self.human_mask_indices])
                print(f'Human masks loaded. Sampling from {avg_valid:.0f} pixels per image ({100*avg_valid/self.total_pixels:.1f}%)')
        else:
            print(f'No human_mask directory at {human_mask_dir}, sampling from all pixels')

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        # Get camera index (for multi-frame: idx % n_cameras)
        cam_idx = idx % self.n_cameras if self.multi_frame else idx
        frame_idx = self.frame_indices[idx]

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[cam_idx],
            "pose": self.pose_all[cam_idx],
            "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        if self.sampling_idx is not None:
            # If human masks loaded, sample per-image excluding human pixels
            if len(self.human_mask_indices) > 0 and idx < len(self.human_mask_indices):
                valid_indices = self.human_mask_indices[idx]
                sampling_size = self.sampling_size
                if len(valid_indices) >= sampling_size:
                    perm = torch.randperm(len(valid_indices))[:sampling_size]
                    sampling_idx = valid_indices[perm]
                else:
                    perm = torch.randint(0, len(valid_indices), (sampling_size,))
                    sampling_idx = valid_indices[perm]
            else:
                sampling_idx = self.sampling_idx
            
            ground_truth["rgb"] = self.rgb_images[idx][sampling_idx, :]
            sample["uv"] = uv[sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
            self.sampling_size = -1
        else:
            self.sampling_size = sampling_size
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']
    
    def get_frame_indices(self):
        """Get unique embedding indices used in dataset."""
        return sorted(list(set(self.frame_indices)))
    
    def get_obj_poses_for_frames(self):
        """Get object poses only for frames used in this dataset."""
        if self.obj_poses is None:
            return None
        # Use actual frame numbers to index into obj_poses
        actual_frames = sorted(list(set(self.actual_frame_numbers)))
        return self.obj_poses[actual_frames]
