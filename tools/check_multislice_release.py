#!/usr/bin/env python3
"""Check the published multislice sources/assets without scientific imports."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re


OLD_FILES = (
    'seareg/multislice/selective_model.py',
    'seareg/multislice/selective_geometry.py',
    'tutorials/configs/Embryo_multislice.json',
    'tutorials/configs/Embryo_multislice_coherent_boundary.json',
)
OLD_RESULT_DIRS = (
    'result/Embryo_multislice/coherent_boundary',
    'result/Embryo_multislice/training',
)
OLD_REFERENCES = (
    'Embryo_multislice_coherent_boundary.json',
    'Embryo_multislice.json',
    'result/Embryo_multislice/coherent_boundary/',
    'result/Embryo_multislice/training/',
    'train_20261027T194615Z_94070e4c',
)


def check_multislice_release(root: Path) -> list[str]:
    """Return precise publication failures; do not execute notebooks or models."""
    root = Path(root).resolve()
    failures = []
    for name in OLD_FILES:
        if (root / name).exists():
            failures.append(f'{name}: obsolete multislice file remains')
    for name in OLD_RESULT_DIRS:
        directory = root / name
        if directory.exists() and any(path.is_file() for path in directory.rglob('*')):
            failures.append(f'{name}: obsolete result files remain')

    config_path = root / 'seareg/multislice/config.py'
    try:
        tree = ast.parse(config_path.read_text(encoding='utf-8-sig'))
        modes = [ast.literal_eval(node.value) for node in ast.walk(tree)
                 if isinstance(node, ast.AnnAssign) and
                 isinstance(node.target, ast.Name) and node.target.id == 'correction_mode']
        if modes != ['cs_matched03']:
            failures.append('seareg/multislice/config.py: default must be cs_matched03')
    except (OSError, SyntaxError, ValueError, TypeError) as error:
        failures.append(f'seareg/multislice/config.py: {error}')

    profile_path = root / 'tutorials/configs/Embryo_multislice_cs_matched03.json'
    try:
        profile = json.loads(profile_path.read_text(encoding='utf-8-sig'))
        training = profile['training']
        expected = {'correction_mode': 'cs_matched03', 'epochs': 500,
                    'gate_refresh_every': 50, 'checkpoint_every': 10,
                    'export_start_epoch': 300, 'export_every': 100,
                    'reshard_every': 100, 'negative_refresh': 5}
        for key, value in expected.items():
            if training.get(key) != value:
                failures.append(f'{profile_path.relative_to(root).as_posix()}: {key} must be {value!r}')
        if profile['analysis'].get('target_domains') != [16] or profile['analysis'].get('fixed_resolution') is not None:
            failures.append(f'{profile_path.relative_to(root).as_posix()}: target16 native search expected')
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(f'{profile_path.relative_to(root).as_posix()}: {error}')

    paths = list((root / 'docs').rglob('*.rst'))
    paths += [root / name for name in ('docs/conf.py', 'README.md', 'data/README.md',
                                      'result/README.md', 'tutorials/README.md',
                                      'data/Embryo_multislice/annotation_palette_manuscript_source.json')]
    notebook = root / 'tutorials/T6_Mouse_Brain_Development_MISAR_seq.ipynb'
    if notebook.is_file():
        paths.append(notebook)
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding='utf-8-sig')
        for old in OLD_REFERENCES:
            if old in text:
                failures.append(f'{path.relative_to(root).as_posix()}: obsolete reference {old}')

    for name, pattern in (
        ('pyproject.toml', r'(?m)^version\s*=\s*"1\.0\.0"\s*$'),
        ('seareg/__init__.py', r'__version__\s*=\s*[\'"]1\.0\.0[\'"]'),
        ('CITATION.cff', r'(?m)^version:\s*[\'"]?1\.0\.0[\'"]?\s*$'),
    ):
        path = root / name
        if not path.is_file() or not re.search(pattern, path.read_text(encoding='utf-8-sig')):
            failures.append(f'{name}: software version must remain1.0.0')
    return failures


if __name__ == '__main__':
    issues = check_multislice_release(Path(__file__).resolve().parents[1])
    for issue in issues:
        print(issue)
    print(f'Multislice release check: {len(issues)} issue(s)')
    raise SystemExit(bool(issues))
