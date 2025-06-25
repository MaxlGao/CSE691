import numpy as np
import time
import copy
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
import re
from helper_burr_piece_creator import create_floor, define_all_burr_pieces, FLOOR_INDEX_OFFSET
from helper_geometric_functions import check_path_clear, get_feasible_motions,\
get_valid_mates, move_piece, get_unsupported_pids
from helper_display import compile_gif, show_and_save_frames, display_pieces
from helper_file_mgmt import load_mates_list, load_simulation_state, save_mates_list, save_simulation_state
from datetime import datetime
import pyglet
from functools import wraps
import warnings
from helper_friction import find_contacting_pieces, check_path_clear_with_friction, get_cost_change_with_friction, move_piece_with_friction

# Nomenclature
# Scored Move = (Scores, Move) 
#             = (Scores, (Mate[0], Mate[1], Translation)) 
#             = ([Score 0, Score 1, ..., Rollout Score]], ((pid1, cid1), (pid2, cid2), Translation))
# Pieces = [Piece, ..., Piece] 
#        = [{Mesh, Corners, ID}, ..., Piece]
#        = [{Mesh, (ID, World Position), ID}, ..., Piece]

# Patch to prevent ZeroDivisionError in Trimesh viewer
def patch_pyglet_on_resize():
    original_dispatch_event = pyglet.window.Window.dispatch_event
    
    @wraps(original_dispatch_event)
    def safe_dispatch_event(self, event_type, *args):
        try:
            # For on_resize event, ensure height is never zero
            if event_type == 'on_resize' and len(args) >= 2:
                width, height = args[0], args[1]
                if height <= 0:
                    return False  # Skip the event if height is zero or negative
            return original_dispatch_event(self, event_type, *args)
        except ZeroDivisionError:
            warnings.warn("Prevented ZeroDivisionError during window resize")
            return False
        except Exception as e:
            warnings.warn(f"Error in window event handling: {str(e)}")
            return False
    
    pyglet.window.Window.dispatch_event = safe_dispatch_event

patch_pyglet_on_resize()

np.set_printoptions(formatter={'int': '{:2d}'.format})
NUM_PIECES = 6
TARGET_OFFSETS = np.array([[0,0,1],[-1,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,-1]])
START_OFFSETS = np.array([[4,-8,1],[-8,0,1],[8,0,1],[-4,8,3],[-4,-8,3],[4,8,1]])
use_friction = True

# Dynamic Programming Functions
def get_top_k_scored_moves(pieces, active_pids, target_offsets, k=float("inf"), mates_list=None, max_held=2):
    # First get a headcount of who's unsupported, except the floor
    pieces_not_floor = [piece for piece in pieces if piece['id'] != FLOOR_INDEX_OFFSET]
    unsupported_ids = get_unsupported_pids(pieces_not_floor)
    for piece in pieces_not_floor:
        piece['gripper_config'][1] = False
    for id in unsupported_ids:
        piece = pieces[id]
        piece['gripper_config'][1] = True # Activate gripper for unsupported pieces
    # If we're at the limit, only these unsupported pieces can get moved. Otherwise, carry on.
    movable_ids = [i for i in range(NUM_PIECES)]
    if len(unsupported_ids) >= max_held:
        movable_ids = unsupported_ids
    
    # Second, quickly make a broad list of semi-verified moves, not checking for intermediate collision
    unverified = []
    unverified_counts = []
    active_pieces = [piece for piece in pieces if piece['id'] in active_pids]
    for id in movable_ids:
        piece = [piece for piece in pieces if piece['id'] == id][0]
        new_unverified = get_feasible_motions(piece, active_pieces, mates_list, steps=1, check_path=False)
        unverified = unverified + new_unverified
        unverified_counts.append(len(new_unverified))

    # Then sort by score. 
    unverified_scored_moves = []
    has_null_move = False
    for (pid1, cid1), (pid2, cid2), vec in unverified:
        if np.linalg.norm(vec) <= 0.01:
            if has_null_move:
                continue # Don't collect more than one null move
            else:
                has_null_move = True
        
        this_piece = pieces[pid1]
        
        # Use friction-aware cost calculation if enabled
        if use_friction:
            # Find pieces in contact with the piece to be moved
            other_pieces = [piece for piece in active_pieces if piece['id'] != pid1]
            contacting_pieces = find_contacting_pieces(this_piece, other_pieces)
            cost_change = get_cost_change_with_friction(this_piece, vec, target_offsets[pid1], contacting_pieces)
        else:
            cost_change = get_cost_change(this_piece, vec, target_offsets[pid1])
        
        # cost_change = get_cost_change(pieces[pid1], vec, target_offsets[pid1])
        cost_change += 0.001 # Constant cost to encourage fewer moves (and break ties)
        unverified_scored_moves.append((cost_change, ((pid1, cid1), (pid2, cid2), vec)))
    unverified_scored_moves.sort(key=lambda x: x[0])

    # Then check feasibility until you get your quota. (or run out of feasible moves)
    feasible_scored_moves = []
    for scored_move in unverified_scored_moves:
        _, ((pid1, _), (_, _), vec) = scored_move
        this_piece = pieces[pid1]
        
        other_pieces = [piece for piece in active_pieces if piece['id'] != pid1]
        
        # Use friction-aware path checking if enabled
        if use_friction:
            path_clear = check_path_clear_with_friction(this_piece, other_pieces, vec, steps=20)
        else:
            test_mesh = this_piece['mesh'].copy()
            other_meshes = [piece['mesh'] for piece in other_pieces]
            path_clear = check_path_clear(test_mesh, other_meshes, vec, 20)
            
        if path_clear:
            # If removing THIS piece leads to 2+ unsupported pieces, it won't work. 
            if len(get_unsupported_pids(other_pieces)) >= max_held:
                continue
            feasible_scored_moves.append(scored_move)
            if len(feasible_scored_moves) == k:
                break
        
        # test_mesh = this_piece['mesh'].copy()
        # other_pieces = [piece for piece in active_pieces if piece['id'] != pid1]
        # other_meshes = [piece['mesh'] for piece in other_pieces]
        # if check_path_clear(test_mesh, other_meshes, vec, 20):
        #     # If removing THIS piece leads to 2+ unsupported pieces, it won't work. 
        #     if len(get_unsupported_pids(other_pieces)) >= max_held:
        #         continue
        #     feasible_scored_moves.append(scored_move)
        #     if len(feasible_scored_moves) == k:
        #         break
            
    k = min(len(feasible_scored_moves), k)
    return feasible_scored_moves

def get_moves_scored_lookahead(pieces, active_pids, target_offsets, mates_list=None, top_k=[float("inf")], rollout_depth=2):
    print("\n\nGetting primary moves...")
    top_scored_moves = get_top_k_scored_moves(pieces, active_pids, target_offsets, k=top_k[0], mates_list=mates_list)
    print(f"| Looking Ahead from top {len(top_scored_moves)} moves...")

    args_list = []
    for scored_move in top_scored_moves:
        # No need for serialization with ThreadPoolExecutor
        args_list.append((scored_move, pieces, active_pids, target_offsets, mates_list, rollout_depth, top_k[1:]))
    
    new_scored_moves = []
    total_moves_compared = 0
    
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(execute_lookahead_recursive, *args) for args in args_list]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), ncols=100, desc='| '):
            try:
                result = future.result()
                scored_move, num_moves = result
                new_scored_moves.append(scored_move)
                total_moves_compared += num_moves
            except Exception as e:
                print(f"Error in worker thread: {str(e)}")
                import traceback
                traceback.print_exc()  # This will give more detailed error information
                continue

    if new_scored_moves:
        new_scored_moves.sort(key=lambda x: x[0][0]) 
        new_scored_moves.sort(key=lambda x: np.trunc(sum(x[0])*1e7))
    else:
        print("Warning: No valid scored moves were produced by the lookahead")
    
    return new_scored_moves, total_moves_compared

# def get_moves_scored_lookahead(pieces, active_pids, target_offsets, mates_list=None, top_k=[float("inf")], rollout_depth=2):
#     print("\n\nGetting primary moves...")
#     top_scored_moves = get_top_k_scored_moves(pieces, active_pids, target_offsets, k=top_k[0], mates_list=mates_list)
#     print(f"| Looking Ahead from top {len(top_scored_moves)} moves...")

#     args_list = []
#     # args_list = [
#     #     (scored_move, pieces, active_pids, target_offsets, mates_list, rollout_depth, top_k[1:])
#     #     for scored_move in top_scored_moves
#     # ]
#     for scored_move in top_scored_moves:
#         # Ensure any numpy arrays in scored_move are converted to lists
#         cost, ((pid1, cid1), (pid2, cid2), vec) = scored_move
#         vec_list = vec.tolist() if hasattr(vec, 'tolist') else vec
#         safe_scored_move = (cost, ((pid1, cid1), (pid2, cid2), vec_list))
        
#         # Convert target_offsets to list for serialization
#         target_offsets_list = target_offsets.tolist() if hasattr(target_offsets, 'tolist') else target_offsets
        
#         args_list.append((safe_scored_move, pieces, active_pids, target_offsets_list, mates_list, rollout_depth, top_k[1:]))
    
#     new_scored_moves = []
#     total_moves_compared = 0
#     # for i, move in tqdm(enumerate(top_scored_moves), total=len(top_scored_moves), ncols=100, desc='| '):
#     #     result = execute_lookahead_recursive(*args_list[i])
#     #     scored_move, num_moves = result
#     #     new_scored_moves.append(scored_move)
#     #     total_moves_compared += num_moves
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         futures = [executor.submit(execute_lookahead_recursive, *args) for args in args_list]
#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), ncols=100, desc='| '):
#             try:
#                 result = future.result()
#                 scored_move, num_moves = result
#                 new_scored_moves.append(scored_move)
#                 total_moves_compared += num_moves
#             except Exception as e:
#                 print(f"Error in worker process: {str(e)}")
#                 # Continue with other calculations
#                 continue

#     # Break ties by picking the best immediate move
#     # new_scored_moves.sort(key=lambda x: x[0][0]) 
#     # new_scored_moves.sort(key=lambda x: np.trunc(sum(x[0])*1e7))
#     if new_scored_moves:
#         new_scored_moves.sort(key=lambda x: x[0][0]) 
#         new_scored_moves.sort(key=lambda x: np.trunc(sum(x[0])*1e7))
#     else:
#         print("Warning: No valid scored moves were produced by the lookahead")
#     return new_scored_moves, total_moves_compared

def execute_lookahead_recursive(scored_move, pieces, active_pids, target_offsets, mates_list, rollout_depth, top_k_remaining):
    primary_cost, ((pid1, cid1), (pid2, cid2), vec) = scored_move
    
    # Convert vec back to numpy array if it's a list
    vec = np.array(vec) if isinstance(vec, list) else vec
    target_offsets = np.array(target_offsets) if isinstance(target_offsets, list) else target_offsets

    # Build a virtual assembly according to scored_move
    # temp_piece = move_piece(copy.deepcopy(pieces[pid1]), vec)
    if use_friction:
        active_pieces = [piece for piece in pieces if piece['id'] in active_pids]
        other_pieces = [piece for piece in active_pieces if piece['id'] != pid1]
        contacting_pieces = find_contacting_pieces(pieces[pid1], other_pieces)
        temp_piece, can_move = move_piece_with_friction(copy.deepcopy(pieces[pid1]), vec, contacting_pieces)
        # If movement is prevented by friction, return high cost
        if not can_move:
            return ([float('inf')], ((pid1, cid1), (pid2, cid2), vec)), 1
    else:
        temp_piece = move_piece(copy.deepcopy(pieces[pid1]), vec)
    
    temp_pieces = copy.deepcopy(pieces)
    temp_pieces[pid1] = temp_piece
    temp_active_pids = active_pids + [temp_piece['id']] if temp_piece['id'] not in active_pids else active_pids
    
    temp_pieces = copy.deepcopy(pieces)
    temp_pieces[pid1] = temp_piece
    temp_active_pids = active_pids + [temp_piece['id']] if temp_piece['id'] not in active_pids else active_pids

    # Recurse or Rollout
    if not top_k_remaining:
        future_cost = greedy_rollout_score(temp_pieces, temp_active_pids, target_offsets, mates_list, depth=rollout_depth)
        future_cost_r = np.round(future_cost, 9)
        total_scores = [primary_cost] + [future_cost_r]
        return (total_scores, ((pid1, cid1), (pid2, cid2), vec)), 1

    next_k = top_k_remaining[0]
    next_moves = get_top_k_scored_moves(temp_pieces, temp_active_pids, target_offsets, k=next_k, mates_list=mates_list)
    
    child_scores = []
    num_children = 0
    for move in next_moves:
        child_result, num_child = execute_lookahead_recursive(move, temp_pieces, temp_active_pids, target_offsets, mates_list, rollout_depth, top_k_remaining[1:])
        num_children += num_child
        child_scores.append(child_result)

    # Choose best child
    # Break ties by picking the best immediate move
    presorted_child_scores = sorted(child_scores, key=lambda x: x[0][0]) 
    postsorted_child_scores = sorted(presorted_child_scores, key=lambda x: np.trunc(sum(x[0])*1e7))
    # sums = [np.trunc(sum(nsm[0])*1e7)/1e7 for nsm in postsorted_child_scores]
    best_child = postsorted_child_scores[0]
    total_scores = [primary_cost] + best_child[0]
    return (total_scores, ((pid1, cid1), (pid2, cid2), vec)), num_children

def greedy_rollout_score(pieces, active_pids, target_offsets, mates_list, depth=2):
    if depth == 0:
        return 0

    # start_time = time.time()
    best_cost = float("inf")

    best_move = get_top_k_scored_moves(pieces, active_pids, target_offsets, k=1, mates_list=mates_list)
    best_cost, ((best_pid, _), (_, _), best_vec) = best_move[0]
    # x, y, z = best_vec
    # print(f"| | | Greedy: Best move is p{best_pid} moving by <{x: .0f},{y: .0f},{z: .0f}>. Depth to go: {depth-1}. This took {time.time() - start_time:.2f}s.")
    if best_move is None or np.linalg.norm(best_vec) <= 0.1:
        return 0  # No valid move or best greedy move is zero.

    # Execute best move
    temp_piece = move_piece(copy.deepcopy(pieces[best_pid]), best_vec)
    temp_pieces = copy.deepcopy(pieces)
    temp_pieces[best_pid] = temp_piece
    new_active_pids = active_pids + [temp_piece['id']] if temp_piece['id'] not in active_pids else active_pids

    return best_cost + greedy_rollout_score(temp_pieces, new_active_pids, target_offsets, mates_list, depth=depth-1)

def cost_function(pieces, target_offsets):
    offsets = [piece['mesh'].bounding_box.centroid for piece in pieces]
    diff = offsets - target_offsets
    dist = [np.linalg.norm(dif) for dif in diff]
    return sum(dist)

# Update the get_cost_change function to maintain backwards compatibility
def get_cost_change(piece, translation, target_offset):
    current_location = piece['mesh'].bounding_box.centroid
    current_cost = np.linalg.norm(current_location - target_offset)
    new_location = current_location + translation
    new_cost = np.linalg.norm(new_location - target_offset)
    diff_cost = new_cost - current_cost
    return np.round(diff_cost, 9)


# Top-Level Scripts
def run_assembler(n_stages=16,top_k=[float("inf")], rollout_depth=0, start_from=None, folder='results', reverse=False, use_friction=True):
    # Automatically find all simulation step files
    sim_folder = Path(folder)
    step_files = sorted(sim_folder.glob('step_*.pkl'))
    step_indices = sorted([
        int(re.search(r'step_(\d+)\.pkl', f.name).group(1))
        for f in step_files
    ])
    latest_step = step_indices[-1] if step_indices else None

    if reverse:
        target_offsets = START_OFFSETS
    else:
        target_offsets = TARGET_OFFSETS + [0,0,3]

    def print_best_move_info(scored_moves):
        best_costs, ((best_pid, _), (other_pid, _), vec) = scored_moves[0]
        first_cost = best_costs[0]
        sum_cost = sum(best_costs)
        best_costs += [sum_cost, sum_cost-first_cost]
        labels = [f"Lookahead {i+1}" for i in range(len(best_costs)-3)] + ["Rollout"] + ["Total"] + ["Tot. After 1st"]
        scores = ", ".join(f"{l}: {s:.2f}" for l, s in zip(labels, best_costs))
        x, y, z = vec
        print(f"| Score [{scores}]")
        print(f"| → Best move: Piece {best_pid} → {other_pid}, vec = <{x: .0f},{y: .0f},{z: .0f}>.")
        return best_pid, vec

    # If there are no files available, we must restart (otherwise go with start_from)
    if latest_step is None:
        # Create Floor and Real Pieces. The floor is a static base piece. 
        floor = create_floor(reverse=reverse)
        if reverse:
            pieces = define_all_burr_pieces(TARGET_OFFSETS + [0,0,3])
            active_pids = [i for i in range(NUM_PIECES)] + [FLOOR_INDEX_OFFSET]
        else:
            pieces = define_all_burr_pieces(START_OFFSETS)
            active_pids = [FLOOR_INDEX_OFFSET]
        pieces_augmented = pieces + [floor]
        print(f"Initial Cost Measure: {cost_function(pieces, target_offsets):.2f}")
        start_from = 0
    else:
        if start_from is None or start_from > latest_step:
            start_from = latest_step
        print(f"Loading Data from Stage {start_from}...")
        state = load_simulation_state(start_from, folder=folder)
        pieces_augmented = state['pieces']
        active_pids = state['assembly']
        scored_moves = state['available_moves']
        pieces = [piece for piece in pieces_augmented if piece['id'] != FLOOR_INDEX_OFFSET ]
        floor = next(piece for piece in pieces_augmented if piece['id'] == FLOOR_INDEX_OFFSET)
        print(f"| Loaded {len(scored_moves)} moves.")
        
        if not scored_moves:
            print(f"No valid moves found in loaded state {start_from}. Assembly might be complete or stuck.")
            return
        
        best_pid, best_vec = print_best_move_info(scored_moves)

        moved_piece = move_piece(pieces[best_pid], best_vec)
        moved_piece['gripper_config'][1] = True # Activate gripper for moved piece
        active_pids = active_pids + [best_pid] if best_pid not in active_pids else active_pids
        print(f"→ Fast-Finished {start_from+1} / {n_stages}.")

        if cost_function(pieces, target_offsets) < 0.1:
            save_simulation_state(start_from+1, pieces_augmented, active_pids, [], folder=folder)
            return # If we're at zero cost, we're done.
        start_from += 1
    
    # Precompute Mates (p1,c1), (p2,c2)
    mates_list = load_mates_list(reverse=reverse)
    if mates_list is None:
        mates_list = get_valid_mates(pieces, floor)
        save_mates_list(mates_list, reverse=reverse)

    for stage in range(start_from, n_stages):
        start_time = time.time()
        scored_moves, moves_compared = get_moves_scored_lookahead(pieces_augmented, active_pids, target_offsets, mates_list, top_k=top_k, rollout_depth=rollout_depth)
        print(f"| | Processed {moves_compared} moves.")
        
        # Add check for empty scored_moves list
        if not scored_moves:
            print(f"No valid moves found at stage {stage}. Assembly might be complete or stuck.")
            save_simulation_state(stage, pieces_augmented, active_pids, [], folder=folder)
            return
        
        best_pid, best_vec = print_best_move_info(scored_moves)
        save_simulation_state(stage, pieces_augmented, active_pids, scored_moves, folder=folder)
        
        # Now execute
        moved_piece = pieces[best_pid]
        moved_piece = move_piece(moved_piece, best_vec)
        moved_piece['gripper_config'][1] = True # Activate gripper for moved piece
        active_pids = active_pids + [best_pid] if best_pid not in active_pids else active_pids
        print(f"→ Done with Stage {stage+1} / {n_stages}. This took {time.time() - start_time:.2f} seconds.")

        if cost_function(pieces, target_offsets) < 0.1:
            save_simulation_state(stage+1, pieces_augmented, active_pids, [], folder=folder)
            return # If we're at zero cost, we're done.
    return

if __name__=="__main__":
    reverse = True # Reverse lets you do Assembly by disassembly, which is much faster. 
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder = f"results/{timestamp}"
    # folder = f"results/2025-04-26_19-39-03" # Existing folder
    folder_sim = f"{folder}/states"
    folder_img = f"{folder}/frames"

    start_time = time.time()
    run_assembler(n_stages=30, top_k=[100, 10], rollout_depth=1, folder=folder_sim, reverse=reverse, use_friction=True)

    # Visualize and save frames. Hold=True lets you drag around the scene
    show_and_save_frames(folder_sim, folder_img, hold=False, reverse=reverse, steps=20)

    compile_gif(folder=folder_img, reverse=False)
    compile_gif(folder=folder_img, reverse=True)
    print(f"This script took {time.time() - start_time:4f} seconds")