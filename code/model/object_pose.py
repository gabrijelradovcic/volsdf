"""Learnable object pose parameters for multi-frame VolSDF using pytorch3d."""
import torch
import torch.nn as nn
from pytorch3d.transforms import se3_exp_map, se3_log_map


class ObjectPoseParams(nn.Module):
    """Learnable SE3 object poses using pytorch3d."""
    
    def __init__(self, num_frames, init_poses=None):
        super().__init__()
        self.num_frames = num_frames
        # 6D log representation for SE3
        self.pose_deltas = nn.Embedding(num_frames, 6)
        nn.init.zeros_(self.pose_deltas.weight)
        
        if init_poses is not None:
            self._init_from_poses(init_poses)
    
    def _init_from_poses(self, poses):
        """Initialize from 4x4 pose matrices."""
        poses = torch.as_tensor(poses, dtype=torch.float32)
        self.register_buffer('pose_0', poses[0:1].clone())
        self.register_buffer('init_poses', poses.clone())
        
        # Compute relative poses to frame 0
        pose_0_inv = torch.linalg.inv(self.pose_0)
        relative = poses @ pose_0_inv
        
        # Convert to SE3 log representation using pytorch3d
        # pytorch3d expects (N, 4, 4) and returns (N, 6)
        log_rep = se3_log_map(relative.transpose(-1, -2))
        
        # Store initial log representation for regularization (don't learn this)
        self.register_buffer('init_log_rep', log_rep.clone())
        
        with torch.no_grad():
            self.pose_deltas.weight.copy_(log_rep)
    
    def get_pose(self, frame_ids):
        """Get RELATIVE 4x4 pose for given frame indices (relative to frame 0).
        
        Convention (matching combine_human_object_meshes.py):
          - Frame 0 → identity (canonical space = frame-0 world space)
          - Frame N → pose_n @ inv(pose_0) (relative transform)
        
        VolSDF applies inv(get_pose) to world points → canonical:
          - Frame 0: identity → points stay in world (= canonical)
          - Frame N: inv(T_rel) → maps to frame-0 world (= canonical)
        """
        log_rep = self.pose_deltas(frame_ids)
        # Fix frame 0: replace its log_rep with zeros → se3_exp_map gives identity
        is_frame_0 = (frame_ids == 0).unsqueeze(-1)
        log_rep = torch.where(is_frame_0, torch.zeros_like(log_rep), log_rep)
        # pytorch3d se3_exp_map: (N, 6) -> (N, 4, 4) in column-major, transpose back
        rel_pose = se3_exp_map(log_rep).transpose(-1, -2)
        return rel_pose
    
    def reg_loss(self):
        """Regularization: penalize deviation from initial poses (frame 0 excluded).
        
        This keeps poses close to their initialization while still allowing
        small corrections during optimization.
        """
        # Deviation from initial log representation (not from zero!)
        delta = self.pose_deltas.weight[1:] - self.init_log_rep[1:]
        return (delta ** 2).mean()
