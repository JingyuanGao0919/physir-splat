"""Average views within each scene, then average scenes with equal weights."""
import argparse
import json
from pathlib import Path


TI_GROUPS = {
    'Indoor': ['apples', 'corridor', 'heated', 'merge', 'sitting', 'standing', 'wall'],
    'Outdoor': ['bicycle', 'car', 'bridge', 'chair', 'basketball_court', 'soccer_goal', 'tall_building'],
    'UAV': ['UAV1', 'UAV2', 'UAV3', 'UAV4', 'UAV5', 'UAV6'],
}
SCENES = {
    'tinsd': [scene for scenes in TI_GROUPS.values() for scene in scenes],
    'rgbt': ['Building', 'Dimsum', 'Ebike', 'DailyStuff'],
}


def aggregate(dataset, output, scenes):
    per_scene = {}
    for scene in scenes:
        report = json.loads((output / scene / 'model30k/benchmark_metrics.json').read_text())
        per_scene[scene] = {modality: row['mean'] for modality, row in report['modalities'].items()}

    def mean(names):
        first = per_scene[names[0]]
        return {modality: {metric: sum(per_scene[name][modality][metric] for name in names) / len(names)
                           for metric in first[modality]} for modality in first}

    result = dict(dataset=dataset, scenes=len(scenes), aggregation='scene_equal',
                  overall=mean(scenes), per_scene=per_scene)
    if dataset == 'tinsd':
        result['categories'] = {}
        for category, members in TI_GROUPS.items():
            selected = [name for name in members if name in per_scene]
            if selected:
                result['categories'][category] = dict(scenes=len(selected), mean=mean(selected))
    (output / 'scene_equal.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != 'per_scene'}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=list(SCENES), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenes', nargs='+')
    args = parser.parse_args()
    aggregate(args.dataset, args.output, args.scenes or SCENES[args.dataset])
