"""Train from scratch, render iteration 30000, and evaluate the selected scenes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shlex
import subprocess
import sys

from aggregate import SCENES, aggregate


ROOT = Path(__file__).resolve().parent


def commands(dataset, scene, data_root, output, eval_python):
    source = data_root / scene
    destination = output / scene
    common = ['--branch', 'ir' if dataset == 'tinsd' else 'rgbt',
              '--validation-stride', '0', '--seed', '0', '--warm_up', '0',
              '--scaling_lr', '0.001', '--position_lr_max_steps', '30000',
              '--TCM_lr_max_steps', '30000', '--ATF_lr_max_steps', '30000',
              '--atmosphere', '--reflection', '--temporal-source',
              '--source-sh-degree', '3', '--topology-stop-iteration', '4000',
              '--opacity-reset-stop-iteration', '8000']
    if dataset == 'tinsd':
        stages = [
            ('prefix8k', 8000, 'ti', common + ['--source-time-interpolation']),
            ('prefix12k', 12000, 'ti', ['--local-source-spline']),
            ('model30k', 30000, 'ti', ['--local-source-detail', '--local-source-fine', '--local-source-finer']),
        ]
    else:
        deformation = ['--thermal-view-deformation', '--source-time-interpolation']
        if scene != 'Building':
            deformation += ['--deformation-pose-interpolation']
        stages = [
            ('prefix4k', 4000, 'rgbt', common + ['--digital-sigma', '0.3']),
            ('deformation8k', 8000, 'rgbt', deformation),
            ('common10k', 10000, 'rgbt', ['--rgb-scalar-response', '--rgb-response-interpolation',
                                        'pose', '--deformation-pose-interpolation']),
            ('model30k', 30000, 'rgbt-perceptual', []),
        ]
    jobs = [('prepare', [sys.executable, str(ROOT / 'prepare.py'), str(source)])]
    previous = None
    for name, endpoint, recipe, flags in stages:
        model = destination / name
        command = [sys.executable, str(ROOT / 'train.py'), '-s', str(source), '-m', str(model),
                   '--recipe', recipe, '--iterations', str(endpoint)]
        if previous is not None:
            command += ['--resume', str(previous / 'resume.pt')]
        jobs.append((name, command + flags))
        previous = model
    jobs += [
        ('render', [sys.executable, str(ROOT / 'render.py'), '-m', str(previous),
                    '--iteration', '30000', '--split', 'test']),
        ('evaluate', [eval_python, str(ROOT / 'evaluate.py'), '--model', str(previous),
                      '--iteration', '30000', '--split', 'test']),
    ]
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=list(SCENES), required=True)
    parser.add_argument('--data-root', type=Path, required=True, help='TI-NSD or RGBT-Scenes directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenes', nargs='+', help='Default: all scenes in the dataset')
    parser.add_argument('--gpus', nargs='+', default=['0'], help='One sequential worker per GPU')
    parser.add_argument('--eval-python', default=sys.executable)
    parser.add_argument('--dry-run', action='store_true', help='Print commands without running or writing outputs')
    args = parser.parse_args()
    scenes = args.scenes or SCENES[args.dataset]
    data_root, output = args.data_root.resolve(), args.output.resolve()

    def worker(gpu, assigned):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        for scene in assigned:
            for stage, command in commands(args.dataset, scene, data_root, output, args.eval_python):
                print('GPU %s | %s | %s\n%s' % (gpu, scene, stage,
                      ' '.join(shlex.quote(part) for part in command)), flush=True)
                if not args.dry_run:
                    log = output / scene / (stage + '.log')
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open('w') as stream:
                        subprocess.run(command, env=environment, cwd=str(ROOT), stdout=stream,
                                       stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker, gpu, scenes[index::len(args.gpus)])
                   for index, gpu in enumerate(args.gpus)]
        for future in futures:
            future.result()
    if not args.dry_run:
        aggregate(args.dataset, output, scenes)


if __name__ == '__main__':
    main()
