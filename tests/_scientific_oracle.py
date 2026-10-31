"""Optional frozen scientific oracles configured outside the public repository.

Set SEAREG_SCIENTIFIC_ORACLE_MANIFEST for full private equivalence validation.
Only an unconfigured oracle skips; an invalid configured manifest fails loudly.
"""
import json
import os
from pathlib import Path


def scientific_oracle(test_case, section):
    location = os.environ.get('SEAREG_SCIENTIFIC_ORACLE_MANIFEST')
    if not location:
        test_case.skipTest('frozen scientific oracle manifest is not configured')
    manifest = json.loads(Path(location).read_text(encoding='utf-8'))
    if manifest['schema_version'] != 1:
        raise ValueError('unsupported scientific oracle manifest schema')
    configuration = manifest[section]
    for key in ('source_dir', 'source_file', 'selector_file'):
        if key in configuration and not Path(configuration[key]).exists():
            raise FileNotFoundError(configuration[key])
    return configuration
