import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
import numpy as np
from tqdm import tqdm

import utils.general as utils
import utils.plots as plt
from utils import rend_util

class VolSDFTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_int('dataset.scan_id', default=-1)
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)

        self.ds_len = len(self.train_dataset)
        print('Finish loading data. Data-set size: {0}'.format(self.ds_len))
        if scan_id < 24 and scan_id > 0: # BlendedMVS, running for 200k iterations
            self.nepochs = int(200000 / self.ds_len)
            print('RUNNING FOR {0}'.format(self.nepochs))

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        conf_model = self.conf.get_config('model')
        
        # Get multi-frame pose info
        num_frames = getattr(self.train_dataset, 'n_frames', 1)
        init_poses = None
        if hasattr(self.train_dataset, 'get_obj_poses_for_frames'):
            init_poses = self.train_dataset.get_obj_poses_for_frames()
        
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(
            conf=conf_model, num_frames=num_frames, init_poses=init_poses
        )
        if torch.cuda.is_available():
            self.model.cuda()

        # Setup single log file for tracking object poses over training
        self.pose_log_path = os.path.join(self.expdir, self.timestamp, 'object_poses.log')
        if hasattr(self.train_dataset, 'actual_frame_numbers'):
            self.actual_frame_numbers = sorted(list(set(self.train_dataset.actual_frame_numbers)))
        else:
            self.actual_frame_numbers = None
        # Write header and initial poses
        self._init_pose_log()
        self._save_object_poses(epoch=-1)

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        self.lr = self.conf.get_float('train.learning_rate')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.nepochs * len(self.train_dataset)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))

        self.do_vis = kwargs['do_vis']

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')

        # Freeze object poses for the first N epochs
        self.pose_freeze_epochs = self.conf.get_int('train.pose_freeze_epochs', default=0)
        if self.pose_freeze_epochs > 0 and self.model.object_pose is not None:
            self.model.object_pose.requires_grad_(False)
            print(f'Object poses FROZEN for the first {self.pose_freeze_epochs} epochs')

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

    def _init_pose_log(self):
        """Write header to the pose log file."""
        if self.model.object_pose is None:
            return
        obj_pose_module = self.model.object_pose
        n_frames = obj_pose_module.num_frames
        frame_labels = self.actual_frame_numbers if self.actual_frame_numbers else list(range(n_frames))
        
        with open(self.pose_log_path, 'w') as f:
            f.write(f"# Object pose log - {n_frames} frames\n")
            f.write(f"# Actual frame numbers: {frame_labels}\n")
            f.write(f"# Format: epoch | frame | tx ty tz | r1 r2 r3 r4 r5 r6 r7 r8 r9 | log_rep(6)\n")
            f.write(f"# {'='*100}\n")
            
            # Write initial (reference) poses
            if hasattr(obj_pose_module, 'init_poses'):
                init_poses = obj_pose_module.init_poses.cpu().numpy()
                f.write(f"# Initial poses from file:\n")
                for i in range(n_frames):
                    t = init_poses[i, :3, 3]
                    R = init_poses[i, :3, :3].flatten()
                    f.write(f"# init | frame {frame_labels[i]:>5d} | t=[{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}] | R=[{' '.join(f'{v:.6f}' for v in R)}]\n")
                f.write(f"# {'='*100}\n")

    def _save_object_poses(self, epoch):
        """Append current object poses for all frames to a single log file."""
        if self.model.object_pose is None:
            return
        
        obj_pose_module = self.model.object_pose
        n_frames = obj_pose_module.num_frames
        frame_labels = self.actual_frame_numbers if self.actual_frame_numbers else list(range(n_frames))
        
        with torch.no_grad():
            frame_ids = torch.arange(n_frames).cuda()
            current_poses = obj_pose_module.get_pose(frame_ids).cpu().numpy()  # (n_frames, 4, 4)
            log_reps = obj_pose_module.pose_deltas(frame_ids).cpu().numpy()    # (n_frames, 6)
        
        with open(self.pose_log_path, 'a') as f:
            for i in range(n_frames):
                t = current_poses[i, :3, 3]
                R = current_poses[i, :3, :3].flatten()
                lr = log_reps[i]
                f.write(f"{epoch:>6d} | frame {frame_labels[i]:>5d} | "
                        f"t=[{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}] | "
                        f"R=[{' '.join(f'{v:.6f}' for v in R)}] | "
                        f"log=[{' '.join(f'{v:.6f}' for v in lr)}]\n")
        
        # Print translation drift summary to terminal
        if hasattr(obj_pose_module, 'init_poses'):
            init_t = obj_pose_module.init_poses[:, :3, 3].cpu().numpy()
            curr_t = current_poses[:, :3, 3]
            drift = np.linalg.norm(curr_t - init_t, axis=-1)
            print(f'  [Pose] epoch={epoch} | translation drift per frame: {np.array2string(drift, precision=4)}')

    def run(self):
        print("training...")

        for epoch in range(self.start_epoch, self.nepochs + 1):

            # Unfreeze object poses after freeze period
            if (epoch == self.pose_freeze_epochs 
                    and self.model.object_pose is not None 
                    and self.pose_freeze_epochs > 0):
                self.model.object_pose.requires_grad_(True)
                # Frame 0 stays fixed (handled in get_pose), but its embedding
                # gradient won't affect anything. Keep it clean:
                print(f'\n*** Epoch {epoch}: Object poses UNFROZEN ***\n')

            if epoch % self.checkpoint_freq == 0:
                self.save_checkpoints(epoch)

            # Save object poses every epoch
            self._save_object_poses(epoch)

            if self.do_vis and epoch % self.plot_freq == 0:
                self.model.eval()

                self.train_dataset.change_sampling_idx(-1)
                indices, model_input, ground_truth = next(iter(self.plot_dataloader))

                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                if 'frame_idx' in model_input:
                    model_input['frame_idx'] = model_input['frame_idx'].cuda()

                split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                res = []
                for s in tqdm(split):
                    out = self.model(s)
                    d = {'rgb_values': out['rgb_values'].detach(),
                         'normal_map': out['normal_map'].detach()}
                    res.append(d)

                batch_size = ground_truth['rgb'].shape[0]
                model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                plot_data = self.get_plot_data(model_outputs, model_input['pose'], ground_truth['rgb'])

                plt.plot(self.model.implicit_network,
                         indices,
                         plot_data,
                         self.plots_dir,
                         epoch,
                         self.img_res,
                         **self.plot_conf
                         )

                self.model.train()

            self.train_dataset.change_sampling_idx(self.num_pixels)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                if 'frame_idx' in model_input:
                    model_input['frame_idx'] = model_input['frame_idx'].cuda()

                model_outputs = self.model(model_input)
                loss_output = self.loss(model_outputs, ground_truth)

                loss = loss_output['loss']

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                          ground_truth['rgb'].cuda().reshape(-1,3))
                pose_reg = loss_output.get('pose_reg_loss', torch.tensor(0.0))
                print(
                    '{0}_{1} [{2}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7}, pose_reg = {8}, psnr = {9}'
                        .format(self.expname, self.timestamp, epoch, data_index, self.n_batches, loss.item(),
                                loss_output['rgb_loss'].item(),
                                loss_output['eikonal_loss'].item(),
                                pose_reg.item(),
                                psnr.item()))

                self.train_dataset.change_sampling_idx(self.num_pixels)
                self.scheduler.step()

        self.save_checkpoints(epoch)

    def get_plot_data(self, model_outputs, pose, rgb_gt):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.

        plot_data = {
            'rgb_gt': rgb_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
        }

        return plot_data
