import copy
import unittest
from moc_adaptive_sampling.bundles import GROUP_FEATURE_NAMES
from moc_adaptive_sampling.initial_features import CONNECTED_INDEX, canonicalize_initial_features


def recorded(sources, path):
    features = [float(i)/100 for i in range(len(GROUP_FEATURE_NAMES))]
    features[CONNECTED_INDEX] = float(len(path)>1)
    return dict(left='empty', right='a' if len(sources)==1 else 'ab', downstream_delta=1,
                local_delta=0, features=features,
                feature_info=dict(added_sources=sources, path=path, selected_count=0,
                                  group_size=len(sources), is_connected_path=len(path)>1,
                                  incremental_tokens=201, projected_tokens=201))


class InitialFeatureTests(unittest.TestCase):
    def test_singleton_repairs_only_path_flag_and_leaves_input_intact(self):
        row = recorded(['a'], ['a','b'])
        before = copy.deepcopy(row)
        fixed, changes = canonicalize_initial_features(row)
        self.assertEqual(row, before)
        self.assertEqual(set(changes), {'feature_info.path','feature_info.is_connected_path',
                                       'features.group_is_connected_path'})
        self.assertEqual(fixed['feature_info']['path'], ['a'])
        self.assertFalse(fixed['feature_info']['is_connected_path'])
        self.assertEqual([i for i,(a,b) in enumerate(zip(row['features'], fixed['features'])) if a!=b], [CONNECTED_INDEX])
        for key in ('downstream_delta','local_delta'):
            self.assertEqual(fixed[key], row[key])
        for key in ('incremental_tokens','projected_tokens','added_sources'):
            self.assertEqual(fixed['feature_info'][key], row['feature_info'][key])

    def test_pair_and_already_canonical_singleton_are_unchanged(self):
        for row in (recorded(['a','b'], ['b','a']), recorded(['a'], ['a'])):
            fixed, changes = canonicalize_initial_features(row)
            self.assertEqual(fixed, row)
            self.assertEqual(changes, {})

    def test_conditional_or_malformed_rows_are_rejected(self):
        row = recorded(['a'], ['a','b'])
        row['left'] = 'b'
        with self.assertRaises(ValueError):
            canonicalize_initial_features(row)
        row = recorded(['a'], ['a','b'])
        row['features'][CONNECTED_INDEX] = 0.
        with self.assertRaises(ValueError):
            canonicalize_initial_features(row)
        row = recorded(['a','b'], ['a','b','c'])
        with self.assertRaises(ValueError):
            canonicalize_initial_features(row)


if __name__=='__main__':
    unittest.main()
