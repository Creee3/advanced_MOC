"""Auditable schema repair for already recorded v2 initial-state actions.

The frozen collector attached a whole episode's two-node path to singleton
variants, causing their connected-path feature to differ from production.
No outcomes, costs, semantic features, or original records are rewritten.
"""
import copy
import numpy as np
from .bundles import GROUP_FEATURE_NAMES


CONNECTED_INDEX = GROUP_FEATURE_NAMES.index('group_is_connected_path')


def canonicalize_initial_features(row):
    """Return a derived row and field-level repair evidence; never mutate input."""
    value = copy.deepcopy(row)
    if value.get('left') != 'empty':
        raise ValueError('only initial-state features can be canonicalized')
    info = value['feature_info']
    sources, path = info['added_sources'], info['path']
    features = np.asarray(value['features'], dtype=float)
    if features.shape != (len(GROUP_FEATURE_NAMES),) or not np.isfinite(features).all():
        raise ValueError('invalid stored initial-state features')
    if info['selected_count'] != 0 or info['group_size'] != len(sources) or len(set(sources)) != len(sources):
        raise ValueError('invalid stored action membership')
    if info['group_size'] not in (1,2) or not set(sources) <= set(path):
        raise ValueError('unsupported stored action/path')
    if features[CONNECTED_INDEX] != float(info['is_connected_path']):
        raise ValueError('stored connected flag and feature disagree')
    changes = {}
    if len(sources) == 1:
        # A singleton's provenance path in evidence_groups contains itself only.
        if path != sources:
            changes['feature_info.path'] = {'before':path.copy(), 'after':sources.copy()}
            info['path'] = sources.copy()
        if info['is_connected_path']:
            changes['feature_info.is_connected_path'] = {'before':True, 'after':False}
            info['is_connected_path'] = False
            changes['features.group_is_connected_path'] = {'before':float(features[CONNECTED_INDEX]), 'after':0.}
            value['features'][CONNECTED_INDEX] = 0.
    elif len(path) != 2 or set(path) != set(sources) or not info['is_connected_path']:
        raise ValueError('stored pair is not a complete connected-path proposal')
    return value, changes
