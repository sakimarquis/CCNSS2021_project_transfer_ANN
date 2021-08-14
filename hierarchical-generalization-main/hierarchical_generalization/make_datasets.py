"""Script to hold the functions for making the datasets."""
import logging

import hierarchical_generalization.default_configuration as config
import hierarchical_generalization.taskset as ts

logger = logging.getLogger(__name__)


def generate_phase_train_test_data(
        phase_a_args={},
        phase_b_args={},
        phase_c_args={},
):
    """Generates two dictionaries corresponding to the training and testing sets
    for the three phases of the task.
    """
    phase_args = [phase_a_args, phase_b_args, phase_c_args]
    datasets = ({}, {})
    for key, args in zip(ts.phase_labels_funcs.keys(), phase_args):
        for dataset in datasets:
            choices = ts.phase_labels_funcs[key](**args)
            inputs = ts.input_array(*choices)
            labels = ts.actions(*choices)
            dataset[key] = (inputs, labels)
    return datasets

def generate_taskset_test_data(
        ts1_colors=None,
        ts2_colors=None,
        ts_old_colors=None,
        ts_new_colors=None,
        phase_a_shapes=None,
        phase_b_shapes=None,
        phase_c_shapes=None,
):
    """Generates the taskset testing sets to evaluate how the model is doing on
    a particular taskset as it trains.
    """
    ts1_colors = ts1_colors or config.ts1_colors
    ts2_colors = ts2_colors or config.ts2_colors
    ts_old_colors = ts_old_colors or config.ts_old_colors
    ts_new_colors = ts_new_colors or config.ts_new_colors
    phase_a_shapes = phase_a_shapes or config.phase_a_shapes
    phase_b_shapes = phase_b_shapes or config.phase_b_shapes
    phase_c_shapes = phase_c_shapes or config.phase_c_shapes
    
    ts_test_keys = [
        'TS 1 Phase A',
        'TS 2 Phase A',
        'TS 1 Phase B',
        'TS 2 Phase B',
        'TS Old Phase C',
        'TS New Phase C',
    ]
    ts_shape_color_pairs = {key : pairs for key, pairs in zip(ts_test_keys, [
        (ts1_colors, phase_a_shapes),
        (ts2_colors, phase_a_shapes),
        (ts1_colors, phase_b_shapes),
        (ts2_colors, phase_b_shapes),
        (ts_old_colors, phase_c_shapes),
        (ts_new_colors, phase_c_shapes),
    ])}

    ts_test_datasets = {}
    for key in ts_test_keys:
        choices = ts.explicit_phase_labels(*ts_shape_color_pairs[key])
        inputs = ts.input_array(*choices)
        labels = ts.actions(*choices)
        ts_test_datasets[key] = (inputs, labels)    

    return ts_test_datasets

def generate_task_data(
        phase_a_args={},
        phase_b_args={},
        phase_c_args={},
        ts1_colors=None,
        ts2_colors=None,
        ts_old_colors=None,
        ts_new_colors=None,
        phase_a_shapes=None,
        phase_b_shapes=None,
        phase_c_shapes=None,
):
    """Generate the phase training, testing, and taskset data"""
    phase_args = [phase_a_args, phase_b_args, phase_c_args]
    ts1_colors = ts1_colors or config.ts1_colors
    ts2_colors = ts2_colors or config.ts2_colors
    ts_old_colors = ts_old_colors or config.ts_old_colors
    ts_new_colors = ts_new_colors or config.ts_new_colors
    phase_a_shapes = phase_a_shapes or config.phase_a_shapes
    phase_b_shapes = phase_b_shapes or config.phase_b_shapes
    phase_c_shapes = phase_c_shapes or config.phase_c_shapes

    # Go through checking to make sure color arguments match between the phase
    # arguments and taskset arguments
    for phase, args in zip('abc', phase_args):
        colors = args.get('colors') or getattr(
            config, "phase_{}_colors".format(phase))
        if phase in 'ab':
            assert all([c in colors for c in ts1_colors])
            assert all([c in colors for c in ts2_colors])
        elif phase == 'c':
            assert all([c in colors for c in ts_old_colors])
            assert all([c in colors for c in ts_new_colors])
    
    train_data, test_phase_data = generate_phase_train_test_data(
        phase_a_args=phase_a_args,
        phase_b_args=phase_b_args,
        phase_c_args=phase_c_args,
    )
    test_ts_data = generate_taskset_test_data(
        ts1_colors=ts1_colors,
        ts2_colors=ts2_colors,
        ts_old_colors=ts_old_colors,
        ts_new_colors=ts_new_colors,
        phase_a_shapes=phase_a_shapes,
        phase_b_shapes=phase_b_shapes,
        phase_c_shapes=phase_c_shapes,
    )

    return train_data, test_phase_data, test_ts_data
        
        
