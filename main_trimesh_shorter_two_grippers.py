import numpy as np
import time
import copy
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
import re
from helper_burr_piece_creator import create_floor, define_all_burr_pieces, FLOOR_INDEX_OFFSET
from helper_geometric_functions import check_path_clear, get_feasible_motions, get_valid_mates, move_piece, get_unsupported_pids
from helper_display import compile_gif, show_and_save_frames
from helper_file_mgmt import load_mates_list, load_simulation_state, save_mates_list, save_simulation_state
from datetime import datetime
import os

# Nomenclature
# Scored Move = (Scores, Move) 
#             = (Scores, (Mate[0], Mate[1], Translation)) 
#             = ([Score 0, Score 1, ..., Rollout Score]], ((pid1, cid1), (pid2, cid2), Translation))
# Pieces = [Piece, ..., Piece] 
#        = [{Mesh, Corners, ID}, ..., Piece]

np.set_printoptions(formatter={'int': '{:2d}'.format})
NULL_MOVE = ((0,0),(0,0),np.array([0,0,0])) # Definition of zero action
NUM_PIECES = 6
TARGET_OFFSETS = np.array([[0,0,1],[-1,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,-1]])

# Dynamic Programming Functions
def get_top_k_scored_moves(pieces, active_pids, target_offsets, k=float("inf"), mates_list=None, max_held=2):
    # First get a headcount of who's unsupported. 
    # Except the floor
    pieces_not_floor = [piece for piece in pieces if piece['id'] != FLOOR_INDEX_OFFSET]
    unsupported_ids = get_unsupported_pids(pieces_not_floor)
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
        new_unverified = get_feasible_motions(piece, active_pieces, mates_list, steps=1, check_collision=False)
        unverified = unverified + new_unverified
        unverified_counts.append(len(new_unverified))
    unverified.append(NULL_MOVE)

    # Then sort by score. 
    unverified_scored_moves = []
    for (pid1, cid1), (pid2, cid2), vec in unverified:
        cost_change = get_cost_change(pieces[pid1], vec, target_offsets[pid1])
        unverified_scored_moves.append((cost_change, ((pid1, cid1), (pid2, cid2), vec)))
    unverified_scored_moves.sort(key=lambda x: x[0])
    
    # Add adaptive path clearing for final pieces
    active_piece_count = len([p for p in pieces if p['id'] in active_pids and p['id'] < FLOOR_INDEX_OFFSET])
    is_final_pieces = (active_piece_count >= NUM_PIECES-2)
    path_steps = 10 if is_final_pieces else 20  # Fewer steps = less strict path validation

    # Then check feasibility until you get your quota. (or run out of feasible moves)
    feasible_scored_moves = []
    for scored_move in unverified_scored_moves:
        _, ((pid1, _), (_, _), vec) = scored_move
        this_piece = pieces[pid1]
        test_mesh = this_piece['mesh'].copy()
        active_meshes = [piece['mesh'] for piece in active_pieces if piece['id'] != pid1]
        if check_path_clear(test_mesh, active_meshes, vec, path_steps):
            feasible_scored_moves.append(scored_move)
            if len(feasible_scored_moves) == k:
                break
    k = min(len(feasible_scored_moves), k)
    return feasible_scored_moves

def get_moves_scored_lookahead(pieces, active_pids, target_offsets, mates_list=None, top_k=[float("inf")], rollout_depth=2):
    print("\n\nGetting primary moves...")
    # Limit the number of primary moves to consider (e.g., 20 instead of infinite)
    primary_limit = 20 if top_k[0] == float("inf") else top_k[0]
    top_scored_moves = get_top_k_scored_moves(pieces, active_pids, target_offsets, k=primary_limit, mates_list=mates_list)
    # top_scored_moves = get_top_k_scored_moves(pieces, active_pids, target_offsets, k=top_k[0], mates_list=mates_list)
    print(f"| Looking Ahead from top {len(top_scored_moves)} moves...")

    args_list = [
        (scored_move, pieces, active_pids, target_offsets, mates_list, rollout_depth, top_k[1:])
        for scored_move in top_scored_moves
    ]
    new_scored_moves = []
    total_moves_compared = 0
    
    # Use fewer processes for more efficient processing
    max_workers = min(os.cpu_count(), 8)  # Limit number of processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(execute_lookahead_recursive, *args) for args in args_list]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), ncols=100, desc='| '):
            result = future.result()
            scored_move, num_moves = result
            new_scored_moves.append(scored_move)
            total_moves_compared += num_moves

    new_scored_moves.sort(key=lambda x: sum(x[0]))
    return new_scored_moves, total_moves_compared

def execute_lookahead_recursive(scored_move, pieces, active_pids, target_offsets, mates_list, rollout_depth, top_k_remaining):
    primary_cost, ((pid1, cid1), (pid2, cid2), vec) = scored_move

    # Build a virtual assembly according to scored_move
    temp_piece = move_piece(copy.deepcopy(pieces[pid1]), vec)
    temp_pieces = copy.deepcopy(pieces)
    temp_pieces[pid1] = temp_piece
    temp_active_pids = active_pids + [temp_piece['id']] if temp_piece['id'] not in active_pids else active_pids

    # Piece-specific depths: same for first 4 pieces, deeper for last 2
    piece_specific_depths = {
        0: 1,  # Same depth for piece 0
        1: 1,  # Same depth for piece 1
        2: 1,  # Same depth for piece 2
        3: 1,  # Same depth for piece 3
        4: 3,  # Deeper for piece 4
        5: 3,  # Deeper for piece 5
    }
    
    # Recurse or Rollout
    if not top_k_remaining:
        future_cost = greedy_rollout_score(temp_pieces, temp_active_pids, target_offsets, mates_list, 
                                         depth=rollout_depth, piece_specific_depths=piece_specific_depths)
        total_scores = [primary_cost] + [future_cost]
        return (total_scores, ((pid1, cid1), (pid2, cid2), vec)), 1

    next_k = top_k_remaining[0]
    next_moves = get_top_k_scored_moves(temp_pieces, temp_active_pids, target_offsets, k=next_k, mates_list=mates_list)
    
    child_scores = []
    num_children = 0
    for move in next_moves:
        child_result, num_child = execute_lookahead_recursive(move, temp_pieces, temp_active_pids, target_offsets, mates_list, rollout_depth, top_k_remaining[1:])
        num_children += num_child
        child_scores.append(child_result)

    # Choose best child (FIXED - removed duplicate code)
    if child_scores:
        best_child = sorted(child_scores, key=lambda x: sum(x[0]))[0]
        total_scores = [primary_cost] + best_child[0]
    else:
        # Handle case with no children
        total_scores = [primary_cost, 0]
    
    return (total_scores, ((pid1, cid1), (pid2, cid2), vec)), num_children

def execute_greedy(args):
    move, pieces, active_pids, target_offsets, mates_list, depth = args
    primary_cost, ((pid1, cid1), (pid2, cid2), vec) = move

    # Piece-specific depths - same for first 4, deeper for last 2
    piece_specific_depths = {
        0: 1,  # Same depth for piece 0
        1: 1,  # Same depth for piece 1
        2: 1,  # Same depth for piece 2
        3: 1,  # Same depth for piece 3
        4: 3,  # Deeper for piece 4 
        5: 3,  # Deeper for piece 5
    }

    temp_piece = move_piece(copy.deepcopy(pieces[pid1]), vec)
    temp_pieces = copy.deepcopy(pieces)
    temp_pieces[pid1] = temp_piece
    temp_active_pids = active_pids + [temp_piece['id']] if temp_piece['id'] not in active_pids else active_pids

    future_cost = greedy_rollout_score(
        temp_pieces,
        temp_active_pids,
        target_offsets,
        mates_list,
        depth=depth,
        piece_specific_depths=piece_specific_depths
    )
    scores = [primary_cost] + [future_cost]
    return (scores, ((pid1, cid1), (pid2, cid2), vec))

def greedy_rollout_score(pieces, active_pids, target_offsets, mates_list, depth=2, piece_specific_depths=None):
    """
    Compute the greedy rollout score with different depths based on piece IDs.
    piece_specific_depths: Dictionary mapping piece IDs to rollout depths
    """
    if depth == 0:
        return 0

    # Get best move
    best_move = get_top_k_scored_moves(pieces, active_pids, target_offsets, k=1, mates_list=mates_list)
    
    if not best_move:
        return 0
        
    best_cost, ((best_pid, _), (_, _), best_vec) = best_move[0]
    
    if best_cost == 0:
        return 0  # No valid move or best greedy move is zero.

    # Execute best move
    temp_piece = move_piece(copy.deepcopy(pieces[best_pid]), best_vec)
    temp_pieces = copy.deepcopy(pieces)
    temp_pieces[best_pid] = temp_piece
    new_active_pids = active_pids + [temp_piece['id']] if temp_piece['id'] not in active_pids else active_pids
    
    # Determine next depth based on piece ID if piece_specific_depths is provided
    next_depth = depth - 1
    if piece_specific_depths and best_pid in piece_specific_depths:
        # Use the remaining depth for this piece ID or default to depth-1
        next_depth = piece_specific_depths[best_pid] - 1
        # Don't go below 0
        next_depth = max(0, next_depth)

    return best_cost + greedy_rollout_score(temp_pieces, new_active_pids, target_offsets, mates_list, 
                                          depth=next_depth, piece_specific_depths=piece_specific_depths)

def cost_function(pieces, target_offsets):
    offsets = [piece['mesh'].bounding_box.centroid for piece in pieces]
    diff = offsets - target_offsets
    dist = [np.linalg.norm(dif) for dif in diff]
    return sum(dist)

def get_cost_change(piece, translation, target_offset):
    current_location = piece['mesh'].bounding_box.centroid
    current_cost = np.linalg.norm(current_location - target_offset)
    new_location = current_location + translation
    new_cost = np.linalg.norm(new_location - target_offset)
    return new_cost - current_cost


# Top-Level Scripts
def run_assembler(n_stages=16,top_k=[float("inf")], rollout_depth=0, start_from=None, folder='results'):
    # Automatically find all simulation step files
    sim_folder = Path(folder_sim)
    step_files = sorted(sim_folder.glob('step_*.pkl'))

    # Extract step indices from filenames
    step_indices = sorted([
        int(re.search(r'step_(\d+)\.pkl', f.name).group(1))
        for f in step_files
    ])

    latest_step = None
    if len(step_indices) > 0:
        latest_step = step_indices[-1]
        if start_from is None:
            start_from = latest_step # Pick latest step
        else:
            start_from = min(latest_step, start_from) # Try to pick start_from, else latest step

    target_offsets = TARGET_OFFSETS.copy()
    target_offsets = target_offsets + np.array([0, 0, 3]) 

    # If there are no files available, we must restart (otherwise go with start_from)
    if latest_step is None:
        # Create Floor and Real Pieces. The floor is a static base piece. 
        floor = create_floor()
        start_offsets = np.array([[4,-8,1],[-8,0,1],[8,0,1],[-4,8,3],[-4,-8,3],[4,8,1]])
        pieces = define_all_burr_pieces(start_offsets)
        
        cost = cost_function(pieces, target_offsets)
        print(f"Initial Cost Measure: {cost:.2f}")

        pieces_augmented = pieces + [floor]
        active_pids = [FLOOR_INDEX_OFFSET]
        start_from = 0
    else:
        # An awkward setup where a given start_from can be 0, which means to load all data about stage 0, which can contain a lot of scored move data.
        print(f"Loading Data from Stage {start_from}...")
        state = load_simulation_state(start_from, folder=folder)
        pieces_augmented = state['pieces']
        pieces = [piece for piece in pieces_augmented if piece['id'] != FLOOR_INDEX_OFFSET ]
        active_pids = state['assembly']
        scored_moves = state['available_moves']
        print(f"| Loaded {len(scored_moves)} moves.")
        best_move = scored_moves[0]
        best_costs, ((best_pid, _), (other_pid, _), best_vec) = best_move
        cost_labels = [f"Lookahead {i+1}" for i in range(len(best_costs)-1)] + ["Rollout"]
        score_str = ", ".join(f"{label}: {score:.2f}" for label, score in zip(cost_labels, best_costs))
        print(f"| Score [{score_str}]")
        x, y, z = best_vec
        print(f"| → Best move: Piece {best_pid} → {other_pid}, vec = <{x: .0f},{y: .0f},{z: .0f}>.")
        # Here, it is easiest to just execute the move and start from start_from+1. Most of the heavy lifting comes from calculating moves.
        moved_piece = pieces[best_pid]
        moved_piece = move_piece(moved_piece, best_vec)
        pieces[best_pid] = move_piece(pieces[best_pid], best_vec)
        active_pids = active_pids + [best_pid] if best_pid not in active_pids else active_pids
        print(f"→ Fast-Finished {start_from+1} / {n_stages}.")

        if cost_function(pieces, target_offsets) < 0.1:
            return # If we're at zero cost, we're done.
        
        start_from += 1
    
    # Precompute Mates (p1,c1), (p2,c2)
    mates_list = load_mates_list()
    if mates_list is None:
        mates_list = get_valid_mates(pieces, floor)
        save_mates_list(mates_list)

    for stage in range(start_from, n_stages):
        start_time = time.time()
        scored_moves, moves_compared = get_moves_scored_lookahead(pieces_augmented, active_pids, target_offsets, mates_list, top_k=top_k, rollout_depth=rollout_depth)
        print(f"| | Processed {moves_compared} moves.")
        best_move = scored_moves[0]
        best_costs, ((best_pid, _), (other_pid, _), best_vec) = best_move
        cost_labels = [f"Lookahead {i+1}" for i in range(len(best_costs)-1)] + ["Rollout"]
        score_str = ", ".join(f"{label}: {score:.2f}" for label, score in zip(cost_labels, best_costs))
        print(f"| Score [{score_str}]")
        x, y, z = best_vec
        print(f"| → Best move: Piece {best_pid} → {other_pid}, vec = <{x: .0f},{y: .0f},{z: .0f}>.")
        save_simulation_state(stage, pieces_augmented, active_pids, scored_moves, folder=folder)
        
        # Now execute
        moved_piece = pieces[best_pid]
        moved_piece = move_piece(moved_piece, best_vec)
        active_pids = active_pids + [best_pid] if best_pid not in active_pids else active_pids
        print(f"→ Done with Stage {stage+1} / {n_stages}. This took {time.time() - start_time:.2f} seconds.")

        if cost_function(pieces, target_offsets) < 0.1:
            return # If we're at zero cost, we're done.
    return

if __name__=="__main__":
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder = f"results/{timestamp}"
    folder_sim = f"{folder}/states"
    folder_img = f"{folder}/frames"

    start_time = time.time()
    # Optimize parameters for hard-to-solve assemblies
    run_assembler(n_stages=30, top_k=[20, 10, 5], rollout_depth=2, folder=folder_sim)
    
    try:
        # Now show with robot arms
        show_and_save_frames(folder_sim, folder_img, TARGET_OFFSETS, hold=False, start_from=0)
        compile_gif(folder=folder_img, fps=2)  # Slower fps to see arm movements
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Animation generation skipped, but simulation results are saved")
    
    print(f"This script took {time.time() - start_time:4f} seconds")