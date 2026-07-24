import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffuser_actor.utils.layers import (
    FFWRelativeSelfAttentionModule,
    FFWRelativeCrossAttentionModule,
    FFWRelativeSelfCrossAttentionModule
)
from diffuser_actor.utils.encoder import Encoder
from diffuser_actor.utils.layers import ParallelAttention
from diffuser_actor.utils.position_encodings import (
    RotaryPositionEncoding3D,
    SinusoidalPosEmb
)
from diffuser_actor.utils.utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat,
    matrix_to_quaternion,
    quaternion_to_matrix
)


def resolve_token_groups(spec, dim):
    """Normalize a token-group spec into a list of group sizes summing to `dim`.

    Mirrors the semantics of EC-Diffuser's `pint.py`:
      'default' / None -> None, i.e. keep the single-token baseline path
      'per_dim'        -> [1] * dim   (EC-Diffuser's "uniform action")
      [3, 6], [2,1,...]-> used as-is, must sum to dim
    """
    if spec is None or spec == 'default':
        return None
    if spec == 'per_dim':
        return [1] * dim
    if isinstance(spec, str):
        spec = [int(x) for x in spec.replace(' ', '').strip('[]').split(',') if x]
    groups = [int(g) for g in spec]
    if sum(groups) != dim:
        raise ValueError(
            f"token groups {groups} sum to {sum(groups)}, expected {dim}"
        )
    if any(g <= 0 for g in groups):
        raise ValueError(f"token groups must be positive, got {groups}")
    return groups


class DiffuserActor(nn.Module):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 num_vis_ins_attn_layers=2,
                 use_instruction=False,
                 fps_subsampling_factor=5,
                 gripper_loc_bounds=None,
                 rotation_parametrization='6D',
                 quaternion_format='xyzw',
                 diffusion_timesteps=100,
                 nhist=3,
                 relative=False,
                 lang_enhanced=False,
                 action_token_groups='default',
                 proprio_token_groups='default',
                 no_proprio=False,
                 diffuse_gripper=False,
                 n_tasks=1):
        super().__init__()
        self._rotation_parametrization = rotation_parametrization
        self._quaternion_format = quaternion_format
        self._relative = relative
        self.no_proprio = no_proprio
        # Option B: fold gripper openness into the diffused action (action_dim
        # 9 -> 10) so it becomes a real token, instead of a separate head. Lets
        # [3,6,1] semantic tokenization include a gripper token.
        self.diffuse_gripper = diffuse_gripper
        self.use_instruction = use_instruction
        self.encoder = Encoder(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_sampling_level=1,
            nhist=nhist,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor
        )
        self.prediction_head = DiffusionHead(
            embedding_dim=embedding_dim,
            use_instruction=use_instruction,
            rotation_parametrization=rotation_parametrization,
            nhist=nhist,
            lang_enhanced=lang_enhanced,
            action_token_groups=action_token_groups,
            proprio_token_groups=proprio_token_groups,
            diffuse_gripper=diffuse_gripper,
            n_tasks=n_tasks
        )
        self.position_noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
        self.rotation_noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )
        self.n_steps = diffusion_timesteps
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds)

    def encode_inputs(self, visible_rgb, visible_pcd, instruction,
                      curr_gripper, task_id=None):
        # Compute visual features/positional embeddings at different scales
        rgb_feats_pyramid, pcd_pyramid = self.encoder.encode_images(
            visible_rgb, visible_pcd
        )
        # Keep only low-res scale
        context_feats = einops.rearrange(
            rgb_feats_pyramid[0],
            "b ncam c h w -> b (ncam h w) c"
        )
        context = pcd_pyramid[0]

        # Encode instruction (B, 53, F)
        instr_feats = None
        if self.use_instruction:
            instr_feats, _ = self.encoder.encode_instruction(instruction)

        # Cross-attention vision to language
        if self.use_instruction:
            # Attention from vision to language
            context_feats = self.encoder.vision_language_attention(
                context_feats, instr_feats
            )

        # Encode gripper history (B, nhist, F)
        adaln_gripper_feats, _ = self.encoder.encode_curr_gripper(
            curr_gripper, context_feats, context
        )

        # FPS on visual features (N, B, F) and (B, N, F, 2)
        fps_feats, fps_pos = self.encoder.run_fps(
            context_feats.transpose(0, 1),
            self.encoder.relative_pe_layer(context)
        )
        return (
            context_feats, context,  # contextualized visual features
            instr_feats,  # language features
            adaln_gripper_feats,  # gripper history features
            fps_feats, fps_pos,  # sampled visual features
            curr_gripper,  # raw proprio dims, for proprio token grouping
            task_id  # multitask conditioning
        )

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        # Parse inputs
        (
            context_feats,
            context,
            instr_feats,
            adaln_gripper_feats,
            fps_feats,
            fps_pos,
            curr_gripper_raw,
            task_id
        ) = fixed_inputs

        return self.prediction_head(
            trajectory,
            timestep,
            context_feats=context_feats,
            context=context,
            instr_feats=instr_feats,
            adaln_gripper_feats=adaln_gripper_feats,
            fps_feats=fps_feats,
            fps_pos=fps_pos,
            curr_gripper_raw=curr_gripper_raw,
            task_id=task_id
        )

    def conditional_sample(self, condition_data, condition_mask, fixed_inputs):
        self.position_noise_scheduler.set_timesteps(self.n_steps)
        self.rotation_noise_scheduler.set_timesteps(self.n_steps)

        # Random trajectory, conditioned on start-end
        noise = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device
        )
        # Noisy condition data
        noise_t = torch.ones(
            (len(condition_data),), device=condition_data.device
        ).long().mul(self.position_noise_scheduler.timesteps[0])
        noise_pos = self.position_noise_scheduler.add_noise(
            condition_data[..., :3], noise[..., :3], noise_t
        )
        noise_rot = self.rotation_noise_scheduler.add_noise(
            condition_data[..., 3:9], noise[..., 3:9], noise_t
        )
        cond_parts = [noise_pos, noise_rot]
        if self.diffuse_gripper:
            cond_parts.append(self.position_noise_scheduler.add_noise(
                condition_data[..., 9:10], noise[..., 9:10], noise_t
            ))
        noisy_condition_data = torch.cat(cond_parts, -1)
        trajectory = torch.where(
            condition_mask, noisy_condition_data, noise
        )

        # Iterative denoising
        timesteps = self.position_noise_scheduler.timesteps
        for t in timesteps:
            out = self.policy_forward_pass(
                trajectory,
                t * torch.ones(len(trajectory)).to(trajectory.device).long(),
                fixed_inputs
            )
            out = out[-1]  # keep only last layer's output
            pos = self.position_noise_scheduler.step(
                out[..., :3], t, trajectory[..., :3]
            ).prev_sample
            rot = self.rotation_noise_scheduler.step(
                out[..., 3:9], t, trajectory[..., 3:9]
            ).prev_sample
            step_parts = [pos, rot]
            if self.diffuse_gripper:
                step_parts.append(self.position_noise_scheduler.step(
                    out[..., 9:10], t, trajectory[..., 9:10]
                ).prev_sample)
            trajectory = torch.cat(step_parts, -1)

        if not self.diffuse_gripper:
            # Option A: append the head-predicted openness (not diffused).
            trajectory = torch.cat((trajectory, out[..., 9:]), -1)

        return trajectory

    def compute_trajectory(
        self,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        task_id=None
    ):
        # Normalize all pos
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])
        curr_gripper = self.convert_rot(curr_gripper)

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, instruction, curr_gripper, task_id
        )

        # Condition on start-end pose
        B, nhist, D = curr_gripper.shape
        cond_data = torch.zeros(
            (B, trajectory_mask.size(1), D),
            device=rgb_obs.device
        )
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample
        trajectory = self.conditional_sample(
            cond_data,
            cond_mask,
            fixed_inputs
        )

        # Normalize quaternion
        if self._rotation_parametrization != '6D':
            trajectory[:, :, 3:7] = normalise_quat(trajectory[:, :, 3:7])
        # Back to quaternion
        trajectory = self.unconvert_rot(trajectory)
        # unnormalize position
        trajectory[:, :, :3] = self.unnormalize_pos(trajectory[:, :, :3])
        # Convert gripper status to probaility
        if trajectory.shape[-1] > 7:
            if self.diffuse_gripper:
                # Diffused openness lives in [-1,1]; map back to [0,1].
                trajectory[..., 7] = ((trajectory[..., 7] + 1) / 2).clamp(0, 1)
            else:
                trajectory[..., 7] = trajectory[..., 7].sigmoid()

        return trajectory

    def normalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def unnormalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min

    def convert_rot(self, signal):
        signal[..., 3:7] = normalise_quat(signal[..., 3:7])
        if self._rotation_parametrization == '6D':
            # The following code expects wxyz quaternion format!
            if self._quaternion_format == 'xyzw':
                signal[..., 3:7] = signal[..., (6, 3, 4, 5)]
            rot = quaternion_to_matrix(signal[..., 3:7])
            res = signal[..., 7:] if signal.size(-1) > 7 else None
            if len(rot.shape) == 4:
                B, L, D1, D2 = rot.shape
                rot = rot.reshape(B * L, D1, D2)
                rot_6d = get_ortho6d_from_rotation_matrix(rot)
                rot_6d = rot_6d.reshape(B, L, 6)
            else:
                rot_6d = get_ortho6d_from_rotation_matrix(rot)
            signal = torch.cat([signal[..., :3], rot_6d], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
        return signal

    def unconvert_rot(self, signal):
        if self._rotation_parametrization == '6D':
            res = signal[..., 9:] if signal.size(-1) > 9 else None
            if len(signal.shape) == 3:
                B, L, _ = signal.shape
                rot = signal[..., 3:9].reshape(B * L, 6)
                mat = compute_rotation_matrix_from_ortho6d(rot)
                quat = matrix_to_quaternion(mat)
                quat = quat.reshape(B, L, 4)
            else:
                rot = signal[..., 3:9]
                mat = compute_rotation_matrix_from_ortho6d(rot)
                quat = matrix_to_quaternion(mat)
            signal = torch.cat([signal[..., :3], quat], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
            # The above code handled wxyz quaternion format!
            if self._quaternion_format == 'xyzw':
                signal[..., 3:7] = signal[..., (4, 5, 6, 3)]
        return signal

    def convert2rel(self, pcd, curr_gripper):
        """Convert coordinate system relaative to current gripper."""
        center = curr_gripper[:, -1, :3]  # (batch_size, 3)
        bs = center.shape[0]
        pcd = pcd - center.view(bs, 1, 3, 1, 1)
        curr_gripper = curr_gripper.clone()
        curr_gripper[..., :3] = curr_gripper[..., :3] - center.view(bs, 1, 3)
        return pcd, curr_gripper

    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        run_inference=False,
        task_id=None
    ):
        """
        Arguments:
            gt_trajectory: (B, trajectory_length, 3+4+X)
            trajectory_mask: (B, trajectory_length)
            timestep: (B, 1)
            rgb_obs: (B, num_cameras, 3, H, W) in [0, 1]
            pcd_obs: (B, num_cameras, 3, H, W) in world coordinates
            instruction: (B, max_instruction_length, 512)
            curr_gripper: (B, nhist, 3+4+X)

        Note:
            Regardless of rotation parametrization, the input rotation
            is ALWAYS expressed as a quaternion form.
            The model converts it to 6D internally if needed.
        """
        if self._relative:
            pcd_obs, curr_gripper = self.convert2rel(pcd_obs, curr_gripper)
        if gt_trajectory is not None:
            gt_openess = gt_trajectory[..., 7:]
            if self.diffuse_gripper:
                # Keep openness in the diffused trajectory; map {0,1}->{-1,1} to
                # match the [-1,1] range the diffusion operates in.
                gt_trajectory = torch.cat(
                    [gt_trajectory[..., :7], gt_openess * 2 - 1], dim=-1
                )
            else:
                gt_trajectory = gt_trajectory[..., :7]
        if self.diffuse_gripper:
            # Proprio keeps openness too, so a dedicated proprio gripper group
            # (e.g. the '1' in [3,6,1]) is well-defined.
            cg_open = (curr_gripper[..., 7:8] if curr_gripper.size(-1) > 7
                       else torch.ones_like(curr_gripper[..., :1]))
            curr_gripper = torch.cat(
                [curr_gripper[..., :7], cg_open * 2 - 1], dim=-1
            )
        else:
            curr_gripper = curr_gripper[..., :7]
        if self.no_proprio:
            # Ablation (EC-Diffuser "no proprio"): strip all proprioceptive
            # information by replacing the gripper history with a constant
            # canonical pose, so the adaln/timestep gripper conditioning carries
            # no per-sample signal. Pairs with absolute actions -- relative mode
            # would leak the current pose back in through convert2rel.
            curr_gripper = torch.zeros_like(curr_gripper)
            curr_gripper[..., 3] = 1.0  # identity quaternion (wxyz)

        # gt_trajectory is expected to be in the quaternion format
        if run_inference:
            return self.compute_trajectory(
                trajectory_mask,
                rgb_obs,
                pcd_obs,
                instruction,
                curr_gripper,
                task_id
            )
        # Normalize all pos
        gt_trajectory = gt_trajectory.clone()
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        gt_trajectory[:, :, :3] = self.normalize_pos(gt_trajectory[:, :, :3])
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])

        # Convert rotation parametrization
        gt_trajectory = self.convert_rot(gt_trajectory)
        curr_gripper = self.convert_rot(curr_gripper)

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, instruction, curr_gripper, task_id
        )

        # Condition on start-end pose
        cond_data = torch.zeros_like(gt_trajectory)
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample noise
        noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)

        # Sample a random timestep
        timesteps = torch.randint(
            0,
            self.position_noise_scheduler.config.num_train_timesteps,
            (len(noise),), device=noise.device
        ).long()

        # Add noise to the clean trajectories
        pos = self.position_noise_scheduler.add_noise(
            gt_trajectory[..., :3], noise[..., :3],
            timesteps
        )
        rot = self.rotation_noise_scheduler.add_noise(
            gt_trajectory[..., 3:9], noise[..., 3:9],
            timesteps
        )
        noisy_parts = [pos, rot]
        if self.diffuse_gripper:
            grip = self.position_noise_scheduler.add_noise(
                gt_trajectory[..., 9:10], noise[..., 9:10], timesteps
            )
            noisy_parts.append(grip)
        noisy_trajectory = torch.cat(noisy_parts, -1)
        noisy_trajectory[cond_mask] = cond_data[cond_mask]  # condition
        assert not cond_mask.any()

        # Predict the noise residual
        pred = self.policy_forward_pass(
            noisy_trajectory, timesteps, fixed_inputs
        )

        # Compute loss
        total_loss = 0
        for layer_pred in pred:
            trans = layer_pred[..., :3]
            rot = layer_pred[..., 3:9]
            loss = (
                30 * F.l1_loss(trans, noise[..., :3], reduction='mean')
                + 10 * F.l1_loss(rot, noise[..., 3:9], reduction='mean')
            )
            if self.diffuse_gripper:
                # Openness is diffused like position: supervise its noise
                # residual with L1 (no separate BCE head).
                loss = loss + 10 * F.l1_loss(
                    layer_pred[..., 9:10], noise[..., 9:10], reduction='mean'
                )
            elif torch.numel(gt_openess) > 0:
                openess = layer_pred[..., 9:]
                loss += F.binary_cross_entropy_with_logits(openess, gt_openess)
            total_loss = total_loss + loss
        return total_loss


class DiffusionHead(nn.Module):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 use_instruction=False,
                 rotation_parametrization='quat',
                 nhist=3,
                 lang_enhanced=False,
                 action_token_groups='default',
                 proprio_token_groups='default',
                 diffuse_gripper=False,
                 n_tasks=1):
        super().__init__()
        self.use_instruction = use_instruction
        self.lang_enhanced = lang_enhanced
        if '6D' in rotation_parametrization:
            rotation_dim = 6  # continuous 6D
        else:
            rotation_dim = 4  # quaternion

        # --- Action tokenization -------------------------------------------
        # Baseline: the diffused action vector is [pos(3), rot(rotation_dim)];
        # openness is predicted from position features rather than diffused, so
        # grouping partitions these dims only (Option A).
        # Option B (diffuse_gripper): openness becomes the diffused (action_dim)
        # th dim so [3,6,1]-style grouping can give it a dedicated token.
        #
        # 'default' keeps the original single-token-per-timestep encoder so the
        # vanilla baseline is bit-for-bit the upstream architecture.
        self.diffuse_gripper = diffuse_gripper
        self.action_dim = 3 + rotation_dim + (1 if diffuse_gripper else 0)
        self.action_token_groups = resolve_token_groups(
            action_token_groups, self.action_dim
        )
        if self.action_token_groups is None and self.diffuse_gripper:
            # The baseline single-token path emits pos+rot only; a diffused
            # gripper needs the per-group predictors, so cover the whole action
            # with one group.
            self.action_token_groups = [self.action_dim]
        if self.action_token_groups is None:
            self.traj_encoder = nn.Linear(self.action_dim, embedding_dim)
        else:
            groups = self.action_token_groups
            self.group_offsets = np.cumsum([0] + list(groups)).tolist()
            self.traj_group_encoders = nn.ModuleList([
                nn.Linear(g, embedding_dim) for g in groups
            ])
            # lets the model tell otherwise-identical sub-tokens apart
            self.group_embed = nn.Embedding(len(groups), embedding_dim)
            # Each group reads BOTH branches, so 3DDA's position/rotation
            # specialization survives groups that straddle the pos/rot boundary.
            self.group_predictors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * embedding_dim, embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, g)
                ) for g in groups
            ])
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # --- Proprio tokenization ------------------------------------------
        # EC-Diffuser splits proprio into sub-tokens that feed the AdaLN
        # conditioning; the analogue here is to encode dim-groups of the raw
        # gripper history separately before they reach the timestep embedding.
        self.proprio_token_groups = resolve_token_groups(
            proprio_token_groups, self.action_dim
        )
        if self.proprio_token_groups is not None:
            pg = self.proprio_token_groups
            self.proprio_offsets = np.cumsum([0] + list(pg)).tolist()
            self.proprio_group_encoders = nn.ModuleList([
                nn.Linear(g, embedding_dim) for g in pg
            ])
            self.proprio_group_embed = nn.Embedding(len(pg), embedding_dim)
            self.proprio_emb = nn.Sequential(
                nn.Linear(embedding_dim * nhist * len(pg), embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )

        # Multitask conditioning: a learned task-ID embedding added to the
        # diffusion timestep embedding, mirroring EC-Diffuser's pint.py. No
        # language is used. A no-op when n_tasks <= 1.
        self.n_tasks = int(n_tasks)
        if self.n_tasks > 1:
            self.task_embedding = nn.Embedding(self.n_tasks, embedding_dim)
            nn.init.normal_(self.task_embedding.weight, std=0.02)
        else:
            self.task_embedding = None
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)

        # Attention from trajectory queries to language
        self.traj_lang_attention = nn.ModuleList([
            ParallelAttention(
                num_layers=1,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=False, apply_ffn=False
            )
        ])

        # Estimate attends to context (no subsampling)
        self.cross_attn = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=2, use_adaln=True
        )

        # Shared attention layers
        if not self.lang_enhanced:
            self.self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, num_layers=4, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads,
                num_self_attn_layers=4,
                num_cross_attn_layers=3,
                use_adaln=True
            )

        # Specific (non-shared) Output layers:
        # 1. Rotation
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.rotation_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.rotation_self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rotation_dim)
        )

        # 2. Position
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.position_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.position_self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)
        )

        # 3. Openess
        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, trajectory, timestep,
                context_feats, context, instr_feats, adaln_gripper_feats,
                fps_feats, fps_pos, curr_gripper_raw=None, task_id=None):
        """
        Arguments:
            trajectory: (B, trajectory_length, 3+6+X)
            timestep: (B, 1)
            context_feats: (B, N, F)
            context: (B, N, F, 2)
            instr_feats: (B, max_instruction_length, F)
            adaln_gripper_feats: (B, nhist, F)
            fps_feats: (N, B, F), N < context_feats.size(1)
            fps_pos: (B, N, F, 2)
        """
        # Trajectory features
        n_groups = 1
        if self.action_token_groups is None:
            traj_feats = self.traj_encoder(trajectory)  # (B, L, F)
            traj_steps = traj_feats.size(1)
        else:
            # One sub-token per group per timestep, laid out as
            # [t0g0, t0g1, ..., t0gG, t1g0, ...] so a timestep's sub-tokens stay
            # contiguous and reshape(L, G) recovers the grouping.
            n_groups = len(self.action_token_groups)
            traj_steps = trajectory.size(1)
            per_group = [
                enc(trajectory[..., self.group_offsets[i]:self.group_offsets[i + 1]])
                for i, enc in enumerate(self.traj_group_encoders)
            ]
            traj_feats = torch.stack(per_group, dim=2)  # (B, L, G, F)
            gid = torch.arange(n_groups, device=trajectory.device)
            traj_feats = traj_feats + self.group_embed(gid)[None, None]
            traj_feats = traj_feats.flatten(1, 2)  # (B, L*G, F)

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_steps, device=traj_feats.device)
        )[None].repeat(len(traj_feats), 1, 1)
        if n_groups > 1:
            # every sub-token of a timestep shares that timestep's time code
            traj_time_pos = traj_time_pos.repeat_interleave(n_groups, dim=1)
        if self.use_instruction:
            traj_feats, _ = self.traj_lang_attention[0](
                seq1=traj_feats, seq1_key_padding_mask=None,
                seq2=instr_feats, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
            )
        traj_feats = traj_feats + traj_time_pos

        # Predict position, rotation, opening
        traj_feats = einops.rearrange(traj_feats, 'b l c -> l b c')
        context_feats = einops.rearrange(context_feats, 'b l c -> l b c')
        adaln_gripper_feats = einops.rearrange(
            adaln_gripper_feats, 'b l c -> l b c'
        )
        # 3D positions for the rotary PE: sub-tokens of a timestep sit at that
        # timestep's position.
        traj_xyz = trajectory[..., :3]
        if n_groups > 1:
            traj_xyz = traj_xyz.repeat_interleave(n_groups, dim=1)

        pos_pred, rot_pred, openess_pred = self.prediction_head(
            traj_xyz, traj_feats,
            context[..., :3], context_feats,
            timestep, adaln_gripper_feats,
            fps_feats, fps_pos,
            instr_feats,
            n_groups=n_groups,
            curr_gripper_raw=curr_gripper_raw,
            task_id=task_id
        )
        return [torch.cat((pos_pred, rot_pred, openess_pred), -1)]

    def prediction_head(self,
                        gripper_pcd, gripper_features,
                        context_pcd, context_features,
                        timesteps, curr_gripper_features,
                        sampled_context_features, sampled_rel_context_pos,
                        instr_feats, n_groups=1, curr_gripper_raw=None,
                        task_id=None):
        """
        Compute the predicted action (position, rotation, opening).

        Args:
            gripper_pcd: A tensor of shape (B, N, 3)
            gripper_features: A tensor of shape (N, B, F)
            context_pcd: A tensor of shape (B, N, 3)
            context_features: A tensor of shape (N, B, F)
            timesteps: A tensor of shape (B,) indicating the diffusion step
            curr_gripper_features: A tensor of shape (M, B, F)
            sampled_context_features: A tensor of shape (K, B, F)
            sampled_rel_context_pos: A tensor of shape (B, K, F, 2)
            instr_feats: (B, max_instruction_length, F)
        """
        # Diffusion timestep
        time_embs = self.encode_denoising_timestep(
            timesteps, curr_gripper_features, curr_gripper_raw, task_id
        )

        # Positional embeddings
        rel_gripper_pos = self.relative_pe_layer(gripper_pcd)
        rel_context_pos = self.relative_pe_layer(context_pcd)

        # Cross attention from gripper to full context
        gripper_features = self.cross_attn(
            query=gripper_features,
            value=context_features,
            query_pos=rel_gripper_pos,
            value_pos=rel_context_pos,
            diff_ts=time_embs
        )[-1]

        # Self attention among gripper and sampled context
        features = torch.cat([gripper_features, sampled_context_features], 0)
        rel_pos = torch.cat([rel_gripper_pos, sampled_rel_context_pos], 1)
        features = self.self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]

        num_gripper = gripper_features.shape[0]

        # Rotation head
        rotation, rotation_features = self.predict_rot(
            features, rel_pos, time_embs, num_gripper, instr_feats
        )

        # Position head
        position, position_features = self.predict_pos(
            features, rel_pos, time_embs, num_gripper, instr_feats
        )

        grip = None
        if n_groups > 1 or self.diffuse_gripper:
            # Re-derive the action from per-group heads instead of the two
            # baseline predictors. Each group reads both branches so grouping
            # across the pos/rot boundary is well defined. diffuse_gripper forces
            # this path so the (grouped) gripper dim is predicted here too.
            joint = torch.cat([position_features, rotation_features], dim=-1)
            bs, ntok, _ = joint.shape
            joint = joint.view(bs, ntok // n_groups, n_groups, -1)
            action = torch.cat(
                [head(joint[:, :, i]) for i, head in enumerate(self.group_predictors)],
                dim=-1
            )  # (B, L, action_dim), groups concatenated back in dim order
            if self.diffuse_gripper:
                position = action[..., :3]
                rotation = action[..., 3:9]
                grip = action[..., 9:10]
            else:
                position, rotation = action[..., :3], action[..., 3:]
            # openness comes from the timestep's pooled position features
            position_features = position_features.view(
                bs, ntok // n_groups, n_groups, -1
            ).mean(2)

        if self.diffuse_gripper:
            # Option B: openness is the diffused gripper dim, not a head.
            openess = grip
        else:
            # Option A: openness head from position head.
            openess = self.openess_predictor(position_features)

        return position, rotation, openess

    def encode_denoising_timestep(self, timestep, curr_gripper_features,
                                  curr_gripper_raw=None, task_id=None):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)
            - curr_gripper_raw: (B, nhist, action_dim), for proprio grouping
            - task_id: (B,) long, for multitask conditioning

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)

        curr_gripper_features = einops.rearrange(
            curr_gripper_features, "npts b c -> b npts c"
        )
        curr_gripper_features = curr_gripper_features.flatten(1)
        curr_gripper_feats = self.curr_gripper_emb(curr_gripper_features)
        out = time_feats + curr_gripper_feats

        if self.proprio_token_groups is not None and curr_gripper_raw is not None:
            per_group = [
                enc(curr_gripper_raw[..., self.proprio_offsets[i]:self.proprio_offsets[i + 1]])
                for i, enc in enumerate(self.proprio_group_encoders)
            ]
            feats = torch.stack(per_group, dim=2)  # (B, nhist, G, F)
            gid = torch.arange(feats.size(2), device=feats.device)
            feats = feats + self.proprio_group_embed(gid)[None, None]
            out = out + self.proprio_emb(feats.flatten(1))

        if self.task_embedding is not None and task_id is not None:
            task_id = task_id.to(dtype=torch.long, device=out.device).view(-1)
            out = out + self.task_embedding(task_id)

        return out

    def predict_pos(self, features, rel_pos, time_embs, num_gripper,
                    instr_feats):
        position_features = self.position_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        position_features = einops.rearrange(
            position_features[:num_gripper], "npts b c -> b npts c"
        )
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features

    def predict_rot(self, features, rel_pos, time_embs, num_gripper,
                    instr_feats):
        rotation_features = self.rotation_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        rotation_features = einops.rearrange(
            rotation_features[:num_gripper], "npts b c -> b npts c"
        )
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation, rotation_features
