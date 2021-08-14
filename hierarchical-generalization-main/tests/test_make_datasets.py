import logging

import pytest
import numpy as np

import hierarchical_generalization.default_configuration as config
import hierarchical_generalization.make_datasets as md

logger = logging.getLogger(__name__)


def test_generate_phase_train_test_data():
    data = md.generate_phase_train_test_data()
    for d in data:
        for key in d.keys():
            # Has all phases
            assert key in config.phase_names
            
            # Assert each phase is a tuple
            assert isinstance(d[key], tuple)

            # Split the tuple
            X, y =  d[key]
            
            # Assert the shape of the X dataset
            n_samples = getattr(
                config,
                "phase_{}_n_samples".format(key.split()[-1].lower()),
            )
            expected_x_shape = [
                n_samples,
                config.N_COLORS,
                config.N_SHAPES,
            ]
            assert np.array_equal(X.shape, expected_x_shape)

            # Assert the shape of the labels
            n_samples = getattr(
                config,
                "phase_{}_n_samples".format(key.split()[-1].lower()),
            )
            expected_y_shape = [
                n_samples,
                config.N_ACTIONS,
            ]
            assert np.array_equal(y.shape, expected_y_shape)

            # Assert X and y are not empty (all zeros)
            assert np.any(X) and np.any(y)

            # Assert not nans
            assert not np.isnan(np.sum(X))
            assert not np.isnan(np.sum(y))

def test_generate_taskset_test_data():
    data = md.generate_taskset_test_data()
    for value in data.values():         
        # Assert each ts is a tuple
        assert isinstance(value, tuple)

        # Split the tuple
        X, y =  value

        # Assert X and y are not empty (all zeros)
        assert np.any(X) and np.any(y)

        # Assert not nans
        assert not np.isnan(np.sum(X))
        assert not np.isnan(np.sum(y))

def test_generate_task_data():
    data = md.generate_taskset_test_data()
    def tests(val):
        # Assert X and y are not empty (all zeros)
        assert np.any(X) and np.any(y)

        # Assert not nans
        assert not np.isnan(np.sum(X))
        assert not np.isnan(np.sum(y))
    
    for value in data.values():
        if isinstance(value, dict):
            for inner in value.values():
                X, y = inner
                tests(X)
                tests(y)
        else:
            X, y = value
            tests(X)
            tests(y)
