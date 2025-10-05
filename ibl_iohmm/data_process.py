import os
import numpy as np
from glob import glob

def load_trials(alf_path):
    choice = np.load(os.path.join(alf_path, '_ibl_trials.choice.npy'))         # shape (nTrials,), values -1, 0, +1
    contrast_left = np.load(os.path.join(alf_path, '_ibl_trials.contrastLeft.npy'))   # shape (nTrials,), values [0..1] or nan
    contrast_right = np.load(os.path.join(alf_path, '_ibl_trials.contrastRight.npy')) # shape (nTrials,), values [0..1] or nan

    choice = (choice == 1).astype(int) # -1,0 -> 0, 1 -> 1
    contrast_right = np.nan_to_num(contrast_right, nan=0.0)
    contrast_left = np.nan_to_num(contrast_left, nan=0.0)

    return choice, contrast_left, contrast_right

def format_inputs(choice, contrast_left, contrast_right):
    # 1. Signed stimulus contrast: contrastRight - contrastLeft (nan treated as 0)
    # 2. Constant bias term: all ones
    # 3. Previous choice: shifted choice array (with 0 for the first trial)
    # 4. Win-stay/lose-switch regressor: previous choice * previous reward

    n_trials = len(choice)
    correct_side = np.sign(contrast_right - contrast_left)

    # Input 1: signed contrast
    signed_contrast = contrast_right - contrast_left
    # Input 2: bias (all ones)
    bias = np.ones(n_trials)
    # Input 3: previous choice (shifted)
    prev_choice = np.roll(choice, 1)
    prev_choice[0] = 0

    reward = (choice == correct_side).astype(int)
    # Input 4: win-stay/lose-switch regressor
    prev_reward = np.roll(reward, 1)
    prev_reward[0] = 0
    wsls = prev_choice * prev_reward

    return signed_contrast, bias, prev_choice, wsls, n_trials

def clean(choice, contrast_left, contrast_right): 
    n_trials = min(len(choice), len(contrast_right), len(contrast_left))
    if n_trials < 2: return                        
    choice = choice[:n_trials]
    contrast_right = contrast_right[:n_trials]
    contrast_left = contrast_left[:n_trials]
                    
    stimulus = contrast_right - contrast_left
    bias = np.ones(n_trials)
                    
    prev_choice = np.zeros(n_trials)
    prev_choice[1:] = choice[:-1]
                    
    prev_contrast_right = np.zeros(n_trials)
    prev_contrast_left = np.zeros(n_trials)
    prev_contrast_right[1:] = contrast_right[:-1]
    prev_contrast_left[1:] = contrast_left[:-1]
    prev_stimulus_side = np.zeros(n_trials)
    prev_stimulus_side = prev_contrast_right - prev_contrast_left

    return stimulus, bias, prev_choice, prev_stimulus_side, n_trials

def build_input_trials():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(root_dir), "ibl-behavioral-data-Dec2019_test")
    output_dir = os.path.join(root_dir, "output", "ibl_data")
    os.makedirs(output_dir, exist_ok=True)
    
    all_choices = []
    all_contrast_right = []
    all_contrast_left = []

    all_stimulus = []
    all_bias = []
    all_prev_choice = []
    all_prev_stimulus_side = []
    
    lab_dirs = [d for d in os.listdir(data_dir) 
                if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    
    session_count = 0
    total_trials = 0
    
    # Iterate through all labs
    for lab in lab_dirs:
        lab_path = os.path.join(data_dir, lab)
        subjects_path = os.path.join(lab_path, "Subjects")
        if not os.path.exists(subjects_path): continue
            
        subjects = [s for s in os.listdir(subjects_path) 
                   if os.path.isdir(os.path.join(subjects_path, s)) and not s.startswith('.')]
        for subject in subjects:
            subject_path = os.path.join(subjects_path, subject)
            dates = [d for d in os.listdir(subject_path) 
                    if os.path.isdir(os.path.join(subject_path, d)) and not d.startswith('.')]
            
            for date in dates[:1]:
                date_path = os.path.join(subject_path, date)
                sessions = [s for s in os.listdir(date_path) 
                           if os.path.isdir(os.path.join(date_path, s)) and not s.startswith('.')]
                for session in sessions:
                    alf_path = os.path.join(date_path, session, "alf")
                    if not os.path.exists(alf_path): continue
                    
                    choice, contrast_left, contrast_right = load_trials(alf_path)

                    stimulus, bias, prev_choice, prev_stimulus_side, n_trials = format_inputs(choice, contrast_left, contrast_right)
                    # stimulus, bias, prev_choice, prev_stimulus_side, n_trials = clean(choice, contrast_left, contrast_right)

                    # 3 raw
                    all_choices.append(choice)
                    all_contrast_right.append(contrast_right)
                    all_contrast_left.append(contrast_left)
                    # 4 inputs
                    all_stimulus.append(stimulus)
                    all_bias.append(bias)
                    all_prev_choice.append(prev_choice)
                    all_prev_stimulus_side.append(prev_stimulus_side)
                    
                    session_count += 1
                    total_trials += n_trials
    
    print(f"Finished processing. Total sessions: {session_count}, Total trials: {total_trials}")
    
    if not all_choices:
        print("No valid data found!")
        return None
    
    final_data = {
        'choice': np.concatenate(all_choices),
        'contrast_right': np.concatenate(all_contrast_right),
        'contrast_left': np.concatenate(all_contrast_left),
        'stimulus': np.concatenate(all_stimulus),
        'bias': np.concatenate(all_bias),
        'prev_choice': np.concatenate(all_prev_choice),
        'prev_stimulus_side': np.concatenate(all_prev_stimulus_side)
    }
    
    for key, value in final_data.items():
        output_file = os.path.join(output_dir, f"{key}.npy")
        np.save(output_file, value)
    
    input_matrix = np.column_stack([
        final_data['stimulus'],
        final_data['bias'],
        final_data['prev_choice'],
        final_data['prev_stimulus_side']
    ])
    input_file = os.path.join(output_dir, "input_features.npy")
    np.save(input_file, input_matrix)

    obs_file = os.path.join(output_dir, "observations.npy")
    np.save(obs_file, final_data['choice'])

    return final_data


if __name__ == "__main__":
    data = build_input_trials()
    if data is not None:
        print("\nData processing completed successfully!")
        for key, value in data.items():
            print(f"  {key}: {value.shape}")
    else:
        print("Data processing failed!")

