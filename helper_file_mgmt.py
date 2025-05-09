import os
import pickle

def load_mates_list(reverse=False, folder='cache'):
    if reverse:
        filename = f'{folder}/mates_list_reversed.pkl'
    else:
        filename = f'{folder}/mates_list_normal.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            print("Loaded mates_list from file.")
            return pickle.load(f)
    else:
        print("No cached mates_list found.")
        return None

def save_mates_list(mates_list, reverse=False, folder='cache'):
    if reverse:
        filename = f'{folder}/mates_list_reversed.pkl'
    else:
        filename = f'{folder}/mates_list_normal.pkl'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(mates_list, f)
        print(f"Saved mates_list to {filename}")

def save_simulation_state(step_id, pieces, active_pids, all_scored_moves, metadata=None, folder='results', suffix=''):
    os.makedirs(folder, exist_ok=True)
    data = {
        'pieces': pieces,
        'assembly': active_pids,
        'available_moves': all_scored_moves, # All scored moves going FROM the pieces here
        'metadata': metadata or {}
    }
    with open(f'{folder}/step_{step_id:03d}{suffix}.pkl', 'wb') as f:
        pickle.dump(data, f)

def load_simulation_state(step_id, folder='results'):
    with open(f'{folder}/step_{step_id:03d}.pkl', 'rb') as f:
        return pickle.load(f)