"""SEAREG public API. Imports are lazy; execution is always explicit."""

from importlib import import_module

__version__ = "1.0.0"

_EXPORTS = {'Workflow': ('pipeline', 'Workflow'),
 'MultiSliceConfig': ('multislice.config', 'MultiSliceConfig'),
 'MultiSliceInputs': ('multislice.inputs', 'MultiSliceInputs'),
 'MultiSliceFilteredCounts': ('multislice.stages', 'MultiSliceFilteredCounts'),
 'MultiSliceNetwork': ('multislice.stages', 'MultiSliceNetwork'),
 'filter_multislice_counts': ('multislice.stages', 'filter_multislice_counts'),
 'build_multislice_network': ('multislice.stages', 'build_multislice_network'),
 'prepare_multislice_training': ('multislice.inputs', 'prepare_multislice_training'),
 'start_multislice_training_job': ('multislice.jobs', 'start_multislice_training_job'),
 'run_workflow': ('pipeline', 'run_workflow'),
 'PreprocessConfig': ('config', 'PreprocessConfig'),
 'FilterConfig': ('config', 'FilterConfig'),
 'FilteredCounts': ('artifacts', 'FilteredCounts'),
 'filter_counts': ('preprocessing', 'filter_counts'),
 'SequenceConfig': ('config', 'SequenceConfig'),
 'PeakSequences': ('artifacts', 'PeakSequences'),
 'prepare_sequences': ('sequence', 'prepare_sequences'),
 'AnnotationConfig': ('config', 'AnnotationConfig'),
 'EdgeSelectionConfig': ('config', 'EdgeSelectionConfig'),
 'annotate_motifs': ('annotation', 'annotate_motifs'),
 'import_motif_sites': ('annotation', 'import_motif_sites'),
 'MotifAnnotation': ('annotation', 'MotifAnnotation'),
 'SelectedMotifEdges': ('edges', 'SelectedMotifEdges'),
 'select_motif_edges': ('edges', 'select_motif_edges'),
 'import_motif_edges': ('edges', 'import_motif_edges'),
 'NetworkConfig': ('config', 'NetworkConfig'),
 'ModelNetwork': ('artifacts', 'ModelNetwork'),
 'build_network': ('graph', 'build_network'),
 'TrainingInputs': ('artifacts', 'TrainingInputs'),
 'prepare_training': ('sampling', 'prepare_training'),
 'TrainConfig': ('config', 'TrainConfig'),
 'training_defaults': ('defaults', 'training_defaults'),
 'training_config': ('defaults', 'training_config'),
 'multislice_training_config': ('defaults', 'multislice_training_config'),
 'training_profile': ('config', 'training_profile'),
 'AnalysisConfig': ('config', 'AnalysisConfig'),
 'SEAREG': ('model', 'SEAREG'),
 'LightGCN': ('model', 'LightGCN'),
 'PreparedData': ('data', 'PreparedData'),
 'EmbeddingSet': ('data', 'EmbeddingSet'),
 'InputArtifact': ('data', 'InputArtifact'),
 'PreparedNetwork': ('data', 'PreparedNetwork'),
 'TrainingRun': ('data', 'TrainingRun'),
 'TrainingJob': ('jobs', 'TrainingJob'),
 'start_training_job': ('jobs', 'start_training_job'),
 'load_training_job': ('jobs', 'load_training_job'),
 'load_run': ('data', 'load_run'),
 'load_inputs': ('data', 'load_inputs'),
 'prepare_inputs': ('preprocessing', 'prepare_inputs'),
 'inspect_inputs': ('preprocessing', 'inspect_inputs'),
 'prepare_network': ('graph', 'prepare_network'),
 'load_network': ('graph', 'load_network'),
 'inspect_network': ('graph', 'inspect_network'),
 'scan_annotation': ('sequence', 'scan_annotation'),
 'scan_peak_motifs_legacy': ('sequence', 'scan_peak_motifs_legacy'),
 'annotate_peak_motifs': ('annotation', 'annotate_peak_motifs'),
 'import_fimo_annotation': ('annotation', 'import_fimo_annotation'),
 'export_motif_graph': ('annotation', 'export_motif_graph'),
 'AnnotationResult': ('annotation', 'AnnotationResult'),
 'train_SEAREG': ('train', 'train_SEAREG'),
 'analyze': ('analysis', 'analyze'),
 'spatial_scores': ('analysis', 'spatial_scores'),
 'AnalysisResult': ('analysis', 'AnalysisResult'),
 'identify_domains': ('domains', 'identify'),
 'DomainResult': ('domains', 'DomainResult'),
 'fit_spot_umap': ('embedding', 'fit_spot'),
 'fit_joint_umap': ('embedding', 'fit_joint'),
 'EmbeddingResult': ('embedding', 'EmbeddingResult')}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module("." + module, __name__), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
