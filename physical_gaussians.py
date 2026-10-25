"""Thermal3D-GS geometry with independent thermal radiance and RGB SH fields."""
import torch
from torch import nn

from scene.gaussian_model import GaussianModel
from utils.general_utils import build_rotation


class PhysicalGaussians(GaussianModel):
    extra_fields = ('source', 'reflectivity')

    def __init__(self, sh_degree, source_degree=0):
        super().__init__(sh_degree)
        self.source_degree = source_degree
        self.active_source_degree = 0
        if source_degree:
            self.extra_fields = self.extra_fields + ('source_sh',)

    def oneupSHdegree(self):
        super().oneupSHdegree()
        self.active_source_degree = min(self.active_source_degree + 1, self.source_degree)

    def initialize_thermal(self, source_values, reflectivity=0.05):
        self.source = nn.Parameter(torch.log(torch.expm1(source_values.clamp_min(1e-4))))
        self.reflectivity = nn.Parameter(torch.full_like(
            source_values, float(torch.logit(torch.tensor(reflectivity)))))
        if self.source_degree:
            self.source_sh = nn.Parameter(source_values.new_zeros(
                len(source_values), (self.source_degree + 1) ** 2 - 1))

    def enable_source_spline(self):
        self.source_spline = nn.Parameter(self.source.new_zeros(len(self.source), 4))
        self.extra_fields = self.extra_fields + ('source_spline',)

    def enable_source_detail(self):
        self.source_detail = nn.Parameter(self.source.new_zeros(len(self.source), 4))
        self.extra_fields = self.extra_fields + ('source_detail',)

    def enable_source_fine(self):
        self.source_fine = nn.Parameter(self.source.new_zeros(len(self.source), 8))
        self.extra_fields = self.extra_fields + ('source_fine',)

    def enable_source_finer(self):
        self.source_finer = nn.Parameter(self.source.new_zeros(len(self.source), 16))
        self.extra_fields = self.extra_fields + ('source_finer',)

    def training_setup(self, training_args):
        super().training_setup(training_args)
        self.optimizer.add_param_group(dict(params=[self.source],
                                            lr=training_args.feature_lr, name='source'))
        self.optimizer.add_param_group(dict(params=[self.reflectivity],
                                            lr=training_args.feature_lr / 10, name='reflectivity'))
        if self.source_degree:
            self.optimizer.add_param_group(dict(params=[self.source_sh],
                                                lr=training_args.feature_lr / 20, name='source_sh'))

        if hasattr(self, 'source_spline'):
            self.optimizer.add_param_group(dict(params=[self.source_spline],
                                                lr=training_args.feature_lr, name='source_spline'))
        if hasattr(self, 'source_detail'):
            self.optimizer.add_param_group(dict(params=[self.source_detail],
                                                lr=training_args.feature_lr, name='source_detail'))

        if hasattr(self, 'source_fine'):
            self.optimizer.add_param_group(dict(params=[self.source_fine],
                                                lr=training_args.feature_lr, name='source_fine'))

        if hasattr(self, 'source_finer'):
            self.optimizer.add_param_group(dict(params=[self.source_finer],
                                                lr=training_args.feature_lr, name='source_finer'))

    def _prune_optimizer(self, mask):
        tensors = super()._prune_optimizer(mask)
        for name in self.extra_fields:
            setattr(self, name, tensors[name])
        return tensors

    def append_children(self, selected, xyz, scaling, repeats=1):
        additions = dict(
            xyz=xyz, scaling=scaling,
            f_dc=self._features_dc[selected].repeat(repeats, 1, 1),
            f_rest=self._features_rest[selected].repeat(repeats, 1, 1),
            opacity=self._opacity[selected].repeat(repeats, 1),
            rotation=self._rotation[selected].repeat(repeats, 1),
        )
        for name in self.extra_fields:
            additions[name] = getattr(self, name)[selected].repeat(repeats, 1)
        tensors = self.cat_tensors_to_optimizer(additions)
        for attr, group in [('_xyz', 'xyz'), ('_features_dc', 'f_dc'),
                            ('_features_rest', 'f_rest'), ('_opacity', 'opacity'),
                            ('_scaling', 'scaling'), ('_rotation', 'rotation')]:
            setattr(self, attr, tensors[group])
        for name in self.extra_fields:
            setattr(self, name, tensors[name])
        count = len(self._xyz)
        self.xyz_gradient_accum = self._xyz.new_zeros(count, 1)
        self.denom = self._xyz.new_zeros(count, 1)
        self.max_radii2D = self._xyz.new_zeros(count)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, screen_split=None):
        selected = (torch.norm(grads, dim=-1) >= grad_threshold) & (
            self.get_scaling.max(dim=1).values <= self.percent_dense * scene_extent)
        if screen_split is not None:
            selected &= ~screen_split
        self.append_children(selected, self._xyz[selected], self._scaling[selected])

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, screen_split=None):
        padded = self._xyz.new_zeros(len(self._xyz))
        padded[:len(grads)] = grads.reshape(-1)
        large = self.get_scaling.max(dim=1).values > self.percent_dense * scene_extent
        if screen_split is not None:
            large[:len(screen_split)] |= screen_split
        selected = (padded >= grad_threshold) & large
        stds = self.get_scaling[selected].repeat(N, 1)
        samples = torch.normal(torch.zeros_like(stds), stds)
        rotations = build_rotation(self._rotation[selected]).repeat(N, 1, 1)
        xyz = torch.bmm(rotations, samples.unsqueeze(-1)).squeeze(-1)
        xyz = xyz + self._xyz[selected].repeat(N, 1)
        scaling = self.scaling_inverse_activation(stds / (0.8 * N))
        self.append_children(selected, xyz, scaling, repeats=N)
        remove = torch.cat((selected, torch.zeros(len(xyz), dtype=torch.bool, device='cuda')))
        self.prune_points(remove)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size,
                          screen_split_pixels=None):
        if screen_split_pixels is None:
            return super().densify_and_prune(max_grad, min_opacity, extent, max_screen_size)
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        screen_split = self.max_radii2D > screen_split_pixels
        small_world = self.get_scaling.max(dim=1).values <= self.percent_dense * extent
        selected = grads[:, 0] >= max_grad
        stats = dict(points_before=len(grads),
                     gradient_selected=int(selected.sum()),
                     screen_large=int(screen_split.sum()),
                     redirected_clone_to_split=int((selected & small_world & screen_split).sum()))
        self.densify_and_clone(grads, max_grad, extent, screen_split=screen_split)
        self.densify_and_split(grads, max_grad, extent, screen_split=screen_split)
        # Keep native pruning unchanged; this experiment changes clone vs split only.
        prune = (self.get_opacity < min_opacity).squeeze(-1)
        if max_screen_size:
            prune |= (self.max_radii2D > max_screen_size)
            prune |= (self.get_scaling.max(dim=1).values > 0.1 * extent)
        self.prune_points(prune)
        torch.cuda.empty_cache()
        stats['points_after'] = len(self._xyz)
        return stats

    def state(self):
        names = ['_xyz', '_features_dc', '_features_rest', '_scaling',
                 '_rotation', '_opacity'] + list(self.extra_fields)
        return dict(fields={name: getattr(self, name).detach() for name in names},
                    active_sh_degree=self.active_sh_degree, max_sh_degree=self.max_sh_degree,
                    source_degree=self.source_degree, active_source_degree=self.active_source_degree)

    @classmethod
    def from_state(cls, state):
        # v1/v2 checkpoints predate the directional-source ablation.
        model = cls(state['max_sh_degree'], state.get('source_degree', 0))
        if 'source_spline' in state['fields']:
            model.extra_fields = model.extra_fields + ('source_spline',)
        if 'source_detail' in state['fields']:
            model.extra_fields = model.extra_fields + ('source_detail',)
        if 'source_fine' in state['fields']:
            model.extra_fields = model.extra_fields + ('source_fine',)
        if 'source_finer' in state['fields']:
            model.extra_fields = model.extra_fields + ('source_finer',)
        for name, value in state['fields'].items():
            setattr(model, name, nn.Parameter(value.cuda()))
        model.active_sh_degree = state['active_sh_degree']
        model.active_source_degree = state.get('active_source_degree', 0)
        model.max_radii2D = model._xyz.new_zeros(len(model._xyz))
        return model
