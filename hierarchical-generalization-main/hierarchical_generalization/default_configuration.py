"""File to store the default experimental configuration."""
import logging

logger = logging.getLogger(__name__)

# High level constants
N_COLORS = 5
N_SHAPES = 4
N_ACTIONS = 4

phase_names = ("Phase A", "Phase B", "Phase C")


# All the colors and shapes
all_colors = [0, 1, 2, 3, 4]
all_shapes = [1 ,2, 3, 4]

# Phase A
phase_a_colors = [0, 1, 2]
phase_a_color_statistics = [.25, .25, .5]
phase_a_shapes = [1, 2]
phase_a_shape_statistics = [.5, .5]

# Phase B
phase_b_colors = [0, 1, 2]
phase_b_color_statistics = [.25, .25, .5]
phase_b_shapes = [3, 4]
phase_b_shape_statistics = [.5, .5]

# Phase C
phase_c_colors = [3, 4]
phase_c_color_statistics = [.5, .5]
phase_c_shapes = [3, 4]
phase_c_shape_statistics = [.5, .5]

# Color lines correspond to a particular horizontal line
# # colors are not uniformly selected for
# Shapes corespond to a particular vertical line
# Color, Shape combinations correspond to a particular action 1-4

# Recreates figure 1C
# (Color, Shape) : Action
action_dictonary = {
    #     Phase A     |       Phase B
    (0,1) : 1, (0,2) : 2,    (0,3) : 1, (0,4) : 3, # TS 1
    (1,1) : 1, (1,2) : 2,    (1,3) : 1, (1,4) : 3, # TS 1
    # -------------------------------------------
    (2,1) : 3, (2,2) : 4,    (2,3) : 4, (2,4) : 2, # TS 2
    # -------------------------------------------
    #                 |       Phase C
                        (3,3) : 1, (3,4) : 3, # TS Old
                        (4,3) : 1, (4,4) : 2 # TS New
}

# TS Colors
ts1_colors = [0, 1]
ts2_colors = [2]
ts_old_colors = [3]
ts_new_colors = [4]

# Dataset parameters
phase_a_n_samples = 120
phase_b_n_samples = 120
phase_c_n_samples = 120
