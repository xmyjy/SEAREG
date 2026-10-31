"""Unpack and verify the local tutorial inputs."""
from pathlib import Path
import argparse
import json
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'tutorials'))
    from seareg_tutorial_data import prepare_tutorial_file
    datasets = json.loads((root / 'data/manifest.json').read_text())['datasets']
    names = [item['dataset_id'] for item in datasets]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['all', *names], default='all')
    args = parser.parse_args()
    for dataset in datasets:
        if args.dataset != 'all' and dataset['dataset_id'] != args.dataset:
            continue
        for entry in dataset['files']:
            path = prepare_tutorial_file(root, dataset['dataset_id'], Path(entry['path']).name)
            print(path.relative_to(root))


if __name__ == '__main__':
    main()
