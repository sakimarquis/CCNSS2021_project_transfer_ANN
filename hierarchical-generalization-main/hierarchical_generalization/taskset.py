"""Low level code for the Hierarchical Generalization Task"""
import logging

import numpy as np

import hierarchical_generalization.default_configuration as config


logger = logging.getLogger(__file__)

# Phase generation functions

def phase_labels(n_samples,
                 phase_colors,
                 phase_shapes,
                 p_colors,
                 p_shapes,
                 n_colors=config.N_COLORS,
                 n_shapes=config.N_SHAPES,
                ):
    """Base label generator."""
    # N Color samples
    color_choices = np.eye(n_colors)[np.random.choice(
        phase_colors,
        size=n_samples,
        replace=True,
        p=p_colors,
    )].reshape((n_samples, n_colors, 1))
    
    # N Shape samples
    shape_choices = np.eye(n_shapes)[np.random.choice(
        [s-1 for s in phase_shapes],
        size=n_samples,
        replace=True,
        p=p_shapes,
    )].reshape((n_samples, n_shapes, 1))
    
    # Return the choices
    return color_choices, shape_choices    

def phase_a_labels(
        n_samples=config.phase_a_n_samples, 
        colors=config.phase_a_colors,
        shapes=config.phase_a_shapes,
        p_colors=None,
        p_shapes=None,
):
    """Label generator with defaults for phase a"""
    # Color and shape probabilities
    p_colors = p_colors or config.phase_a_color_statistics
    p_shapes = p_shapes or config.phase_a_shape_statistics
    return phase_labels(n_samples, colors, shapes, p_colors, p_shapes)

def phase_b_labels(
        n_samples=config.phase_b_n_samples, 
        colors=config.phase_b_colors,
        shapes=config.phase_b_shapes,
        p_colors=None,
        p_shapes=None,
):
    """Label generator with defaults for phase b"""
    # Color and shape probabilities
    p_colors = p_colors or config.phase_b_color_statistics
    p_shapes = p_shapes or config.phase_b_shape_statistics
    return phase_labels(n_samples, colors, shapes, p_colors, p_shapes)

def phase_c_labels(
        n_samples=config.phase_c_n_samples, 
        colors=config.phase_c_colors,
        shapes=config.phase_c_shapes,
        p_colors=None,
        p_shapes=None,
):
    """Label generator with defaults for phase c"""
    # Color and shape probabilities
    p_colors = p_colors or config.phase_c_color_statistics
    p_shapes = p_shapes or config.phase_c_shape_statistics
    return phase_labels(n_samples, colors, shapes, p_colors, p_shapes)

# Group the funcs above
phase_labels_funcs = {
    name : func for name, func in zip(
        config.phase_names, [phase_a_labels, phase_b_labels, phase_c_labels])
}

# Some helper functions

def integer_labels(labels):
    """Turns one-hot labels to integers"""
    return [np.where(r==1)[0][0] for r in labels]
    
def input_array(
        color_choices,
        shape_choices, 
        n_shapes=config.N_SHAPES,
        n_colors=config.N_COLORS,
):
    """Turns one-hot labels to 2D input arrays with lines"""
    # Full Color array
    color_array = np.tile(color_choices, n_shapes)
    # Full Shape Array
    shape_array = np.transpose(
        np.tile(shape_choices, n_colors),
        [0, 2, 1])
        
    # Full data with both
    x_data = np.maximum(color_array, shape_array)
    return x_data

def actions(color_choices, shape_choices, action_dict=None, n_actions=config.N_ACTIONS):
    """Implements the mapping from shapes and colors to actions"""
    # Create the action dictionary to compare to
    action_dict = action_dict or config.action_dictonary
    # Make binary from one hot and increment shape by 1
    color_binarized = integer_labels(color_choices)
    shape_binarized = [i+1 for i in integer_labels(shape_choices)]
    # Compile into a list of tuples
    label_tuples = [(color, shape) for color, shape in zip(
        color_binarized, shape_binarized)]
    # Substitute based on the action dictionary
    integer_actions = [action_dict[key] for key in label_tuples]
    return np.eye(n_actions)[np.array(integer_actions)-1]

# Task sets

def explicit_phase_labels(phase_colors,
                          phase_shapes,
                          n_colors=config.N_COLORS,
                          n_shapes=config.N_SHAPES,
                         ):
    # Meshgrid across the colors and shapes
    choices = np.array(np.meshgrid(phase_colors, phase_shapes)).T.reshape(-1,2)
    # Turn them into one hot vectors
    color_choices = np.eye(n_colors)[choices[:,0]]
    shape_choices = np.eye(n_shapes)[choices[:,1] - 1]
    # Reshape to the desired shape
    color_choices = color_choices.reshape((len(color_choices), n_colors, 1))
    shape_choices = shape_choices.reshape((len(shape_choices), n_shapes, 1))
    
    return color_choices, shape_choices
