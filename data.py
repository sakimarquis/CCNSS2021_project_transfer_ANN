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