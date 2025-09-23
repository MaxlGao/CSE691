import numpy as np
import time
import contextvars
from pathlib import Path
import re
from helper_burr_piece_creator import create_floor, define_all_burr_pieces, FLOOR_INDEX_OFFSET
from helper_geometric_functions import get_valid_mates, move_piece, set_mates_list
from helper_display import compile_gif, show_and_save_frames
from helper_file_mgmt import load_mates_list, load_simulation_state, save_mates_list, save_simulation_state
from helper_dynamic_programming import cost_function, get_moves_scored_lookahead
from datetime import datetime

# Nomenclature
# Scored Move = {scores, mate, pids, translation}
#             = {[score 0, score 1, ..., Rollout score], ((pid1, cid1), (pid2, cid2)), pids, translation}
# Move Sequence = (Scores, Moves)
# Pieces = [Piece, ..., Piece] 
#        = [{Mesh, Corners, ID, Gripper}, ..., Piece]
#        = [{Mesh, (ID, World Position), ID, Gripper}, ..., Piece]

np.set_printoptions(formatter={'int': '{:2d}'.format})
NUM_PIECES = 6

# Top-Level Scripts
def run_assembler(n_stages=16,top_k=[float("inf")], rollout_depth=0, start_from=None, folder='results', reverse=False):
    # Automatically find all simulation step files
    sim_folder = Path(folder)
    step_files = sorted(sim_folder.glob('step_*.pkl'))
    step_indices = sorted([int(re.search(r'step_(\d+)\.pkl', f.name).group(1)) for f in step_files])
    latest_step = step_indices[-1] if step_indices else None
    prev_piece_and_vec = None

    def print_best_move_info(path_list, scored_moves=[]):
        if not scored_moves:
            first_scored_moves = [path[0] for path in path_list]
            first_moves = [scored_move[1] for scored_move in first_scored_moves]
            score_list_list = []
            for i, path in enumerate(path_list):
                score_list = [move[0] for move in path]
                # move_list = [move[1][:2] for move in path]
                # print(f"Route # {i}, {move_list}")
                score_list_list.append(score_list)
            scored_moves = [(score_list, first_move) for score_list, first_move in zip(score_list_list, first_moves)]

        best_costs, ((best_pid, _), (other_pid, _), vec) = scored_moves[0]
        first_cost = best_costs[0]
        sum_cost = sum(best_costs)
        best_costs += [sum_cost, sum_cost-first_cost]
        labels = [f"Lookahead {i+1}" for i in range(len(best_costs)-3)] + ["Rollout"] + ["Total"] + ["Tot. After 1st"]
        scores = ", ".join(f"{l}: {s:.2f}" for l, s in zip(labels, best_costs))
        x, y, z = vec
        print(f"| Score [{scores}]")
        print(f"| → Best move: Piece {best_pid} → {other_pid}, vec = <{x: .0f},{y: .0f},{z: .0f}>.")
        return best_pid, vec, scored_moves

    # If there are no files available, we must restart (otherwise go with start_from)
    if latest_step is None:
        # Create Floor and Real Pieces. The floor is a static base piece. 
        floor = create_floor(reverse=reverse)
        pieces = define_all_burr_pieces(reverse=reverse)
        if reverse:
            active_pids = [i for i in range(NUM_PIECES)] + [FLOOR_INDEX_OFFSET]
        else:
            active_pids = [FLOOR_INDEX_OFFSET]
        pieces_augmented = pieces + [floor]
        print(f"Initial Cost Measure: {cost_function(pieces):.2f}")
        start_from = 0
    else:
        if start_from is None or start_from > latest_step:
            start_from = latest_step
        print(f"Loading Data from Stage {start_from}...")
        state = load_simulation_state(start_from, folder=folder)
        pieces_augmented = state['pieces']
        active_pids = state['assembly']
        scored_moves = state['available_moves']
        pieces = [piece for piece in pieces_augmented if piece['id'] != FLOOR_INDEX_OFFSET]
        floor = next(piece for piece in pieces_augmented if piece['id'] == FLOOR_INDEX_OFFSET)
        print(f"| Loaded {len(scored_moves)} moves.")
        
        if not scored_moves:
            print(f"No valid moves found in loaded state {start_from}. Assembly might be complete or stuck.")
            return
        
        best_pid, best_vec, _ = print_best_move_info([], scored_moves=scored_moves)

        moved_piece = move_piece(pieces[best_pid], best_vec)
        moved_piece['gripper_config']['active'] = True # Activate gripper for moved piece
        active_pids = active_pids + [best_pid] if best_pid not in active_pids else active_pids
        if reverse:
            if np.linalg.norm(moved_piece['mesh'].bounding_box.centroid - moved_piece['target']) <= 0.01:
                active_pids.remove(best_pid)
                print(f"Retiring Piece {best_pid}; active Pieces now {active_pids}")
        print(f"→ Fast-Finished {start_from+1} / {n_stages}.")

        if cost_function(pieces) < 0.1:
            save_simulation_state(start_from+1, pieces_augmented, active_pids, [], folder=folder)
            return # If we're at zero cost, we're done.
        start_from += 1
    
    # Precompute Mates (p1,c1), (p2,c2)
    mates_list = load_mates_list(reverse=reverse)
    if mates_list is None:
        mates_list = get_valid_mates(pieces, floor)
        save_mates_list(mates_list, reverse=reverse)
    mates_contextvar = contextvars.ContextVar("mates_list")
    mates_contextvar.set(mates_list)
    set_mates_list(mates_contextvar)

    for stage in range(start_from, n_stages):
        start_time = time.time()
        # Defining the assembly object
        assembly = {
            "active_pids": active_pids,
            "prev_piece_and_vec": prev_piece_and_vec,
            "reverse": reverse
        }
        path_list, moves_compared = get_moves_scored_lookahead(pieces_augmented, assembly, top_k=top_k, rollout_depth=rollout_depth)
        # scored_moves
        print(f"| | Processed {moves_compared} moves.")
        
        # Add check for empty scored_moves list
        if not path_list:
            print(f"No valid moves found at stage {stage}. Assembly might be complete or stuck.")
            save_simulation_state(stage, pieces_augmented, active_pids, [], folder=folder)
            return
        
        best_pid, best_vec, scored_moves = print_best_move_info(path_list)
        save_simulation_state(stage, pieces_augmented, active_pids, scored_moves, folder=folder)
        
        # Now execute
        moved_piece = pieces[best_pid]
        moved_piece = move_piece(moved_piece, best_vec)
        prev_piece_and_vec = (best_pid, best_vec)
        moved_piece['gripper_config']['active'] = True # Activate gripper for moved piece
        active_pids = active_pids + [best_pid] if best_pid not in active_pids else active_pids
        if reverse:
            if np.linalg.norm(moved_piece['mesh'].bounding_box.centroid - moved_piece['target']) <= 0.01:
                active_pids.remove(best_pid)
                print(f"Retiring Piece {best_pid}; active Pieces now {active_pids}")
        print(f"→ Done with Stage {stage+1} / {n_stages}. This took {time.time() - start_time:.2f} seconds.")

        if cost_function(pieces) < 0.1:
            save_simulation_state(stage+1, pieces_augmented, active_pids, [], folder=folder)
            return # If we're at zero cost, we're done.
    return

if __name__=="__main__":
    reverse = True # Reverse lets you do Assembly by disassembly, which is much faster. 
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder = f"results/{timestamp}"
    folder = f"results/2025-08-06_15-05-47" # Existing folder
    folder_sim = f"{folder}/states"
    folder_img = f"{folder}/frames"

    start_time = time.time()
    run_assembler(n_stages=30, top_k=[100, 10], rollout_depth=1, folder=folder_sim, reverse=reverse, start_from=14)
    print(f"Planning took {time.time() - start_time:4f} seconds")

    # Visualize and save frames. Hold=True lets you drag around the scene
    # show_and_save_frames(folder_sim, folder_img, hold=False, reverse=reverse, steps=20, start_from=13)

    # compile_gif(folder=folder_img, reverse=False, fps=20)
    # compile_gif(folder=folder_img, reverse=True, fps=20)
    print(f"This script took {time.time() - start_time:4f} seconds")