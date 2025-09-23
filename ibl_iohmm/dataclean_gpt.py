import numpy as np
import os

# Example loader functions for the IBL trial datasets (replace with your actual loader)
def load_trials(alf_path):
    choice = np.load(os.path.join(alf_path, '_ibl_trials.choice.npy'))         # shape (nTrials,), values -1, 0, +1
    contrast_left = np.load(os.path.join(alf_path, '_ibl_trials.contrastLeft.npy'))   # shape (nTrials,), values [0..1] or nan
    contrast_right = np.load(os.path.join(alf_path, '_ibl_trials.contrastRight.npy')) # shape (nTrials,), values [0..1] or nan

    choice = (choice == 1).astype(int) # -1,0 -> 0, 1 -> 1
    contrast_right = np.nan_to_num(contrast_right, nan=0.0)
    contrast_left = np.nan_to_num(contrast_left, nan=0.0)

    return choice, contrast_left, contrast_right


# Data cleaning and feature construction for 4D inputs
# 1. Signed stimulus contrast: contrastRight - contrastLeft (nan treated as 0)
# 2. Constant bias term: all ones
# 3. Previous choice: shifted choice array (with 0 for the first trial)
# 4. Win-stay/lose-switch regressor: previous choice * previous reward
#    (for now assume reward=1 if choice was non-zero and matched correct side)

def format_inputs(choice, contrast_left, contrast_right):
    n_trials = len(choice)

    # You’ll need to load or compute the correct_side per trial (array of -1 or +1)
    # Here is a placeholder:
    correct_side = np.sign(contrast_right - contrast_left)

    # Input 1: signed contrast
    signed_contrast = contrast_right - contrast_left

    # Input 2: bias (all ones)
    bias = np.ones(n_trials)

    # Input 3: previous choice (shifted)
    prev_choice = np.roll(choice, 1)
    prev_choice[0] = 0

    # Determine reward: 1 if choice matches correct side, else 0
    reward = (choice == correct_side).astype(int)

    # Input 4: win-stay/lose-switch regressor
    prev_reward = np.roll(reward, 1)
    prev_reward[0] = 0
    wsls = prev_choice * prev_reward

    return signed_contrast, bias, prev_choice, wsls, n_trials

if __name__ == "__main__":
    alf_path = "/Users/sheng/Documents/U-FreiDocuments/25SS/Project_/neurorobotics/cvx_DLFM/dlfm-examples/ibl-behavioral-data-Dec2019_test/angelakilab/Subjects/NYU-01/2019-03-26/001/alf"
    choice, contrast_left, contrast_right = load_trials(alf_path)

    X, y = format_inputs(choice, contrast_left, contrast_right)
    print("Inputs shape:", X.shape)
    print("Output shape:", y.shape)