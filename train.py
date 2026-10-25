"""Per-scene optimization of the Thermal3D-GS physical rendering branch."""
import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import lpips
from tqdm import tqdm

from arguments import OptimizationParams
from dataset import Dataset
from measurement import Measurement
from physical_gaussians import PhysicalGaussians
from renderer import RadiativeField, render
from rgb_response import pose_records
from deformation_pose import pose_records as deformation_pose_records
from utils.loss_utils import l1_loss, ssim, corners_loss
from utils.general_utils import get_expon_lr_func


def image_loss(prediction, target, iteration, opt, display=True, rmse_mix=0.5):
    if not display:
        return l1_loss(prediction, target)
    corner_weight = 0.2 * max(1 - iteration / 5000, 0)
    photometric = l1_loss(prediction, target)
    if rmse_mix:
        photometric = .5 * photometric + .5 * ((prediction - target).square().mean() + 1e-8).sqrt()
    loss = (1 - opt.lambda_dssim - 0.2) * photometric
    loss = loss + opt.lambda_dssim * (1 - ssim(prediction, target))
    if corner_weight:
        loss = loss + corner_weight * corners_loss(prediction, target)
    return loss


def validation_scores(dataset, gaussians, field, measurement, iteration):
    """Development views only, in the same quantized display domain as export."""
    totals = {}
    with torch.no_grad():
        for view in dataset.cameras['validation']:
            view.load2device()
            result = render(view, gaussians, field, measurement, dataset.branch == 'rgbt', iteration)
            pairs = [('thermal', result['render'], view.original_image)]
            if dataset.branch == 'rgbt':
                pairs.append(('rgb', result['rgb'], view.rgb_image))
            for name, prediction, target in pairs:
                if dataset.profile['domain'] in ('display_gray', 'display_palette') or name == 'rgb':
                    prediction = (prediction * 255 + .5).clamp(0, 255).floor() / 255
                    target = (target * 255 + .5).clamp(0, 255).floor() / 255
                    row = dict(PSNR=float(-10 * (prediction - target).square().mean().log10()),
                               SSIM=float(ssim(prediction, target)))
                else:
                    row = dict(MAE=float((prediction - target).abs().mean()),
                               RMSE=float((prediction - target).square().mean().sqrt()))
                totals.setdefault(name, []).append(row)
            view.load2device('cpu')
            del result
    return {name: {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
            for name, rows in totals.items()}


def create_sensor_optimizer(field, measurement, opt, measurement_lr):
    parameters = [value for name, value in field.named_parameters()
                  if not name.startswith(('source_network.', 'view_deformation.')) and name != 'rgb_log_gain']
    groups = [dict(params=parameters + list(measurement.parameters()),
                   lr=measurement_lr, name='sensor')]
    if field.temporal_source:
        groups.append(dict(params=list(field.source_network.parameters()),
                           lr=opt.position_lr_init * 5, eps=1e-15, name='source_network'))
    if field.rgb_response_anchors is not None:
        groups.append(dict(params=[field.rgb_log_gain], lr=1e-3, name='rgb_response'))
    if field.view_deformation is not None:
        groups.append(dict(params=list(field.view_deformation.parameters()),
                           lr=opt.position_lr_init * 5, eps=1e-15, name='view_deformation'))
    return torch.optim.Adam(groups)


def restore_optimization(saved, gaussians, field, measurement, optimizer, cameras):
    state = saved['training']
    gaussians.optimizer.load_state_dict(state['gaussian_optimizer'])
    for name, value in state['density_buffers'].items():
        setattr(gaussians, name, value)
    field.load_state_dict(saved['radiative_field'])
    measurement.load_state_dict(saved['measurement'])
    # An optional group can have been introduced after another optional group.
    groups = {group['name']: group for group in optimizer.param_groups}
    optimizer.param_groups[:] = [groups[group['name']] for group in state['sensor_optimizer']['param_groups']]
    optimizer.load_state_dict(state['sensor_optimizer'])
    random.setstate(state['random'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state(state['cuda'].cpu())
    by_uid = {view.uid: view for view in cameras}
    return ([by_uid[uid] for uid in state['train_order']],
            [by_uid[uid] for uid in state['stack']], state['moving_loss'], saved['iteration'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    optimization = OptimizationParams(parser)
    parser.add_argument("--recipe", choices=["ti", "rgbt", "rgbt-perceptual"], required=True)
    parser.add_argument('-s', '--source', required=True)
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('--branch', choices=['auto', 'ir', 'rgbt'], default='auto')
    parser.add_argument('--measurement-profile')
    parser.add_argument('--validation-stride', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--atmosphere', action='store_true')
    parser.add_argument('--reflection', action='store_true')
    parser.add_argument('--temporal-source', action='store_true')
    parser.add_argument('--near-mode', choices=['native', 'rescaled-native', 'camera'], default='native')
    parser.add_argument('--screen-split-fraction', type=float, default=None)
    parser.add_argument('--rgb-background', type=float, default=0.)
    parser.add_argument('--source-time-interpolation', action='store_true')
    parser.add_argument('--rgb-scalar-response', action='store_true')
    parser.add_argument('--rgb-response-interpolation', choices=['time', 'pose'], default=None)
    parser.add_argument('--thermal-view-deformation', action='store_true')
    parser.add_argument('--deformation-pose-interpolation', action='store_true')
    parser.add_argument('--opacity-reset-stop-iteration', type=int, default=None)
    parser.add_argument('--topology-stop-iteration', type=int, default=None)
    parser.add_argument('--geometry-stop-iteration', type=int, default=None)
    parser.add_argument('--temporal-frequencies', type=int, default=10)
    parser.add_argument('--temporal-active-bands', type=int,
                        help='Mask higher time bands without changing network dimensions or initialization.')
    parser.add_argument('--local-source-spline', action='store_true')
    parser.add_argument('--local-source-detail', action='store_true')
    parser.add_argument('--local-source-fine', action='store_true')
    parser.add_argument('--local-source-finer', action='store_true')
    parser.add_argument('--source-sh-degree', type=int, choices=[0, 1, 2, 3], default=0,
                        help='Directional relative-radiance capacity for uncalibrated inputs.')
    parser.add_argument('--psf-sigma', type=float, default=0.)
    parser.add_argument('--digital-sigma', type=float, default=0.)
    parser.add_argument('--response-knots', type=int, default=8)
    parser.add_argument('--measurement-lr', type=float, default=1e-3)
    parser.add_argument('--clip-display', action='store_true')
    parser.add_argument('--checkpoint-interval', type=int, default=2000)
    parser.add_argument('--validation-iterations', nargs='+', type=int, default=[1000, 2000, 4000])
    parser.add_argument('--resume', help='Resume optimizer/RNG/densification state; only model path and final iteration change.')
    parser.set_defaults(iterations=4000)
    args = parser.parse_args()
    recipe = args.recipe
    resumed = None
    if args.resume:
        resumed = torch.load(args.resume, map_location='cuda')
        continuation = args
        args = argparse.Namespace(**resumed['config'])
        args.model = continuation.model
        args.iterations = continuation.iterations
        args.resume = continuation.resume
        if continuation.local_source_spline:
            args.local_source_spline = True
        if continuation.local_source_detail:
            args.local_source_detail = True
        if continuation.local_source_fine:
            args.local_source_fine = True
        if continuation.local_source_finer:
            args.local_source_finer = True
        if continuation.rgb_scalar_response:
            args.rgb_scalar_response = True
        if continuation.rgb_response_interpolation is not None:
            args.rgb_response_interpolation = continuation.rgb_response_interpolation
        if continuation.source_time_interpolation:
            args.source_time_interpolation = True
        if continuation.thermal_view_deformation:
            args.thermal_view_deformation = True
        if continuation.deformation_pose_interpolation:
            args.deformation_pose_interpolation = True
        for name in ('topology_stop_iteration', 'geometry_stop_iteration', 'opacity_reset_stop_iteration'):
            value = getattr(continuation, name)
            if value is not None:
                setattr(args, name, value)
    args.recipe = recipe
    args.perceptual_weight = 0. if recipe == "rgbt" else 0.01
    args.photometric_rmse_mix = 0.5 if recipe == "ti" else 0.
    opt = optimization.extract(args)
    output = Path(args.model).resolve()
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    dataset = Dataset(args.source, args.branch, args.measurement_profile,
                      validation_stride=args.validation_stride)
    if getattr(args, 'temporal_active_bands', None) is not None:
        dataset.profile['temporal_active_bands'] = args.temporal_active_bands
    if getattr(args, 'near_mode', 'native') != 'native':
        dataset.profile['near_mode'] = args.near_mode
    if getattr(args, 'rgb_background', 0.) != 0.:
        dataset.profile['rgb_background'] = args.rgb_background
    if getattr(args, 'source_time_interpolation', False):
        dataset.profile['source_time_anchors'] = sorted(set(float(view.fid) for view in dataset.cameras['train']))
    if getattr(args, 'rgb_scalar_response', False) and dataset.branch == 'rgbt':
        dataset.profile['rgb_response_anchors'] = sorted(set(float(view.fid) for view in dataset.cameras['train']))
        if getattr(args, 'rgb_response_interpolation', None) == 'pose':
            dataset.profile['rgb_response_poses'] = pose_records(dataset.cameras['train'], dataset.profile['rgb_response_anchors'])
    if getattr(args, 'thermal_view_deformation', False):
        dataset.profile['thermal_view_deformation'] = True
        dataset.profile['deformation_warmup'] = 3000
    if getattr(args, 'deformation_pose_interpolation', False):
        dataset.profile['deformation_pose_anchors'] = deformation_pose_records(dataset.cameras['train'])
    dataset.save_inputs(output)
    args.branch = dataset.branch
    args.source = str(dataset.source)
    args.model = str(output)
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
    calibrated = dataset.profile['domain'] in ('brightness_temperature', 'flir_counts')
    source_degree = 0 if calibrated else args.source_sh_degree
    if resumed:
        gaussians = PhysicalGaussians.from_state(resumed['gaussians'])
    else:
        gaussians = PhysicalGaussians(3 if dataset.branch == 'rgbt' else 0, source_degree)
        gaussians.create_from_pcd(dataset.point_cloud, dataset.extent)
        if calibrated:
            source = torch.full((len(gaussians.get_xyz), 1), 2.99, device='cuda')
        elif dataset.branch == 'ir' and dataset.profile['domain'] == 'display_gray':
            source = torch.tensor(dataset.point_cloud.colors.mean(1, keepdims=True), device='cuda', dtype=torch.float32)
        else:
            training_mean = torch.stack([c.original_image[:, ::16, ::16].mean()
                                         for c in dataset.cameras['train']]).mean()
            source = torch.full((len(gaussians.get_xyz), 1), float(training_mean), device='cuda')
        gaussians.initialize_thermal(source)
    gaussians.training_setup(opt)
    screen_fraction = getattr(args, 'screen_split_fraction', None)
    screen_pixels = None if screen_fraction is None else screen_fraction * min(
        min(view.image_width, view.image_height) for view in dataset.cameras['train'])
    field_profile = dict(dataset.profile)
    introducing_rgb_response = resumed is not None and 'rgb_response_anchors' in dataset.profile and 'rgb_log_gain' not in resumed['radiative_field']
    if introducing_rgb_response:
        field_profile.pop('rgb_response_anchors')
    introducing_deformation = resumed is not None and dataset.profile.get('thermal_view_deformation', False) and not any(
        name.startswith('view_deformation.') for name in resumed['radiative_field'])
    if introducing_deformation:
        field_profile.pop('thermal_view_deformation')
    field = RadiativeField(field_profile, dataset.extent, args.atmosphere, args.reflection,
                           args.temporal_source, args.temporal_frequencies).cuda()
    measurement = Measurement(dataset.profile, args.response_knots, args.psf_sigma,
                              args.digital_sigma, args.clip_display).cuda()
    optimizer = create_sensor_optimizer(field, measurement, opt, args.measurement_lr)
    source_schedule = get_expon_lr_func(lr_init=opt.position_lr_init * 5,
                                        lr_final=opt.position_lr_final,
                                        lr_delay_mult=opt.position_lr_delay_mult,
                                        max_steps=opt.ATF_lr_max_steps)
    deformation_schedule = get_expon_lr_func(lr_init=opt.position_lr_init * 5,
                                            lr_final=opt.position_lr_final,
                                            lr_delay_mult=opt.position_lr_delay_mult,
                                            max_steps=opt.position_lr_max_steps)
    torch.cuda.synchronize()
    initialized = time.perf_counter()
    train_views = dataset.cameras['train'].copy()
    random.shuffle(train_views)
    stack = []
    moving_loss = 0.
    first_iteration = 0
    perceptual_metric = None
    if args.perceptual_weight:
        feature_init_cpu_rng = torch.get_rng_state()
        feature_init_cuda_rng = torch.cuda.get_rng_state()
        perceptual_metric = lpips.LPIPS(net='vgg').cuda().eval().requires_grad_(False)
        if recipe == 'ti':
            torch.set_rng_state(feature_init_cpu_rng)
            torch.cuda.set_rng_state(feature_init_cuda_rng)
    if resumed:
        train_views, stack, moving_loss, first_iteration = restore_optimization(
            resumed, gaussians, field, measurement, optimizer, dataset.cameras['train'])
        del resumed
        torch.cuda.synchronize()
        initialized = time.perf_counter()

    if getattr(args, 'local_source_spline', False) and not hasattr(gaussians, 'source_spline'):
        gaussians.enable_source_spline()
        gaussians.optimizer.add_param_group(dict(params=[gaussians.source_spline],
                                                lr=opt.feature_lr, name='source_spline'))
    if getattr(args, 'local_source_detail', False) and not hasattr(gaussians, 'source_detail'):
        gaussians.enable_source_detail()
        gaussians.optimizer.add_param_group(dict(params=[gaussians.source_detail],
                                                lr=opt.feature_lr, name='source_detail'))
    if getattr(args, 'local_source_fine', False) and not hasattr(gaussians, 'source_fine'):
        gaussians.enable_source_fine()
        gaussians.optimizer.add_param_group(dict(params=[gaussians.source_fine],
                                                lr=opt.feature_lr, name='source_fine'))
    if getattr(args, 'local_source_finer', False) and not hasattr(gaussians, 'source_finer'):
        gaussians.enable_source_finer()
        gaussians.optimizer.add_param_group(dict(params=[gaussians.source_finer],
                                                lr=opt.feature_lr, name='source_finer'))
    if introducing_rgb_response:
        field.enable_rgb_response(dataset.profile['rgb_response_anchors'])
        optimizer.add_param_group(dict(params=[field.rgb_log_gain], lr=1e-3, name='rgb_response'))
    if introducing_deformation:
        field.enable_view_deformation()
        optimizer.add_param_group(dict(params=list(field.view_deformation.parameters()),
                                       lr=deformation_schedule(first_iteration), eps=1e-15,
                                       name='view_deformation'))

    def scene_state(iteration):
        return dict(iteration=iteration, config=vars(args), profile=dataset.profile,
                    extent=dataset.extent, gaussians=gaussians.state(),
                    radiative_field=field.state_dict(), measurement=measurement.state_dict())

    def save_resume(iteration):
        state = scene_state(iteration)
        state['training'] = dict(
            gaussian_optimizer=gaussians.optimizer.state_dict(),
            sensor_optimizer=optimizer.state_dict(),
            density_buffers={name: getattr(gaussians, name) for name in
                             ('max_radii2D', 'xyz_gradient_accum', 'denom')},
            random=random.getstate(), numpy=np.random.get_state(),
            torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
            train_order=[view.uid for view in train_views], stack=[view.uid for view in stack],
            moving_loss=moving_loss)
        temporary = output / 'resume_next.pt'
        torch.save(state, temporary)
        temporary.replace(output / 'resume.pt')

    validation_seconds = 0.
    checkpoint_seconds = 0.
    topology_stop = getattr(args, 'topology_stop_iteration', None)
    geometry_stop = getattr(args, 'geometry_stop_iteration', None)
    geometry_frozen = False
    display = dataset.profile['domain'] in ('display_gray', 'display_palette')
    print(json.dumps(dict(branch=dataset.branch, domain=dataset.profile['domain'],
                          train=len(train_views), validation=len(dataset.cameras['validation']),
                          test=len(dataset.cameras['test']), initial_gaussians=len(gaussians.get_xyz))), flush=True)
    with tqdm(total=opt.iterations, initial=first_iteration, desc='Physical Thermal3D-GS') as progress:
        for iteration in range(first_iteration + 1, opt.iterations + 1):
            if geometry_stop is not None and iteration > geometry_stop and not geometry_frozen:
                for name in ('_xyz', '_scaling', '_rotation', '_opacity'):
                    getattr(gaussians, name).requires_grad_(False)
                geometry_frozen = True
                print(json.dumps(dict(event='geometry_frozen', iteration=iteration,
                                      fields=['xyz', 'scaling', 'rotation', 'opacity'])), flush=True)
            density_active = (iteration < opt.densify_until_iter and not geometry_frozen
                              and (topology_stop is None or iteration <= topology_stop))
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()
            if not stack:
                stack = train_views.copy()
            view = stack.pop(random.randint(0, len(stack) - 1))
            view.load2device()
            package = render(view, gaussians, field, measurement, dataset.branch == 'rgbt', iteration)
            loss = image_loss(package['render'], view.original_image, iteration, opt, display, args.photometric_rmse_mix)
            if display and perceptual_metric is not None:
                loss = loss + args.perceptual_weight * perceptual_metric(
                    package['render'][None].clamp(0, 1), view.original_image[None], normalize=True).mean()
            if dataset.branch == 'rgbt':
                loss = loss + image_loss(package['rgb'], view.rgb_image, iteration, opt, rmse_mix=args.photometric_rmse_mix)
                if perceptual_metric is not None:
                    loss = loss + args.perceptual_weight * perceptual_metric(
                        package['rgb'][None].clamp(0, 1), view.rgb_image[None], normalize=True).mean()
            loss = loss + field.regularization(gaussians) + measurement.regularization()
            loss.backward()
            with torch.no_grad():
                moving_loss = 0.4 * loss.item() + 0.6 * moving_loss
                for part in package['density_packages']:
                    visible = part['visibility_filter']
                    gaussians.max_radii2D[visible] = torch.maximum(
                        gaussians.max_radii2D[visible], part['radii'][visible])
                    if density_active:
                        gaussians.add_densification_stats(part['viewspace_points'], visible)
                if iteration < opt.densify_until_iter and not geometry_frozen:
                    if density_active and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        density_stats = gaussians.densify_and_prune(
                            opt.densify_grad_threshold, 0.005, dataset.extent, size_threshold,
                            screen_split_pixels=screen_pixels)
                        if density_stats is not None:
                            density_stats.update(iteration=iteration, screen_split_pixels=screen_pixels)
                            with (output / 'densification.jsonl').open('a') as density_log:
                                density_log.write(json.dumps(density_stats) + '\n')
                    reset_stop = getattr(args, 'opacity_reset_stop_iteration', None)
                    if iteration % opt.opacity_reset_interval == 0 and (reset_stop is None or iteration <= reset_stop):
                        gaussians.reset_opacity()
                # Every numbered iteration includes its optimizer update,
                # including the fixed 30k endpoint.
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                optimizer.step()
                for group in optimizer.param_groups:
                    if group['name'] == 'source_network':
                        group['lr'] = source_schedule(iteration)
                    if group['name'] == 'view_deformation':
                        group['lr'] = deformation_schedule(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)
                optimizer.zero_grad(set_to_none=True)
            view.load2device('cpu')
            if iteration % 100 == 0 or iteration == opt.iterations:
                progress.update(iteration - progress.n)
                progress.set_postfix(loss=round(moving_loss, 6), points=len(gaussians.get_xyz))
            del package, loss
            if iteration in args.validation_iterations or iteration == opt.iterations:
                torch.cuda.synchronize()
                validation_started = time.perf_counter()
                metrics = validation_scores(dataset, gaussians, field, measurement, iteration)
                torch.cuda.synchronize()
                validation_seconds += time.perf_counter() - validation_started
                row = dict(iteration=iteration, split='validation', modalities=metrics,
                           gaussian_count=len(gaussians.get_xyz))
                with (output / 'validation_curve.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
            if iteration % args.checkpoint_interval == 0 or iteration == opt.iterations:
                checkpoint_started = time.perf_counter()
                save_resume(iteration)
                checkpoint_seconds += time.perf_counter() - checkpoint_started
    torch.cuda.synchronize()
    optimized = time.perf_counter()
    checkpoint = scene_state(opt.iterations)
    checkpoint_path = output / ('checkpoint_%d.pt' % opt.iterations)
    torch.save(checkpoint, checkpoint_path)
    report = dict(completed=True, iteration=opt.iterations,
                  segment_start_iteration=first_iteration,
                  schedule_iterations=opt.position_lr_max_steps,
                  full_schedule_completed=opt.iterations == opt.position_lr_max_steps,
                  initialization_seconds=initialized - started,
                  optimization_seconds=optimized - initialized - validation_seconds - checkpoint_seconds,
                  validation_seconds=validation_seconds, rolling_checkpoint_seconds=checkpoint_seconds,
                  elapsed_seconds=time.perf_counter() - started,
                  peak_allocated_gpu_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                  peak_reserved_gpu_gib=torch.cuda.max_memory_reserved() / 2 ** 30,
                  gaussian_count=len(gaussians.get_xyz), model_bytes=checkpoint_path.stat().st_size,
                  gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  torch=torch.__version__, cuda=torch.version.cuda, seed=args.seed,
                  scope='This training segment only. Optimization includes transfers and densification, excludes validation and rolling checkpoints. Total includes initialization and final export; previous segments are in training_profile_<iteration>.json.')
    (output / 'training_profile.json').write_text(json.dumps(report, indent=2))
    (output / ('training_profile_%d.json' % opt.iterations)).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
