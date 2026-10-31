"""Adaptive depth gate training over slice-local shards and shared peaks.

Public preparation and job submission are deliberately free of Torch imports.
"""
from .config import MultiSliceConfig
from .inputs import MultiSliceInputs, prepare_multislice_training
from .stages import (MultiSliceFilteredCounts, MultiSliceNetwork,
                     filter_multislice_counts, build_multislice_network)
from .jobs import start_multislice_training_job

__all__ = ['MultiSliceConfig', 'MultiSliceInputs', 'prepare_multislice_training',
           'MultiSliceFilteredCounts', 'MultiSliceNetwork',
           'filter_multislice_counts', 'build_multislice_network',
           'start_multislice_training_job']
