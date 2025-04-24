import numpy as np
import time
import copy
import concurrent.futures
from helper_burr_piece_creator import create_floor, define_all_burr_pieces, REFERENCE_INDEX_OFFSET, FLOOR_INDEX_OFFSET
from helper_geometric_functions import check_path_clear, get_feasible_motions, get_valid_mates, is_supported, move_piece, get_unsupported_pids
from helper_display import render_scene, show_moves_scored, save_animation_frame, compile_gif
from helper_file_mgmt import load_mates_list, load_simulation_state, save_mates_list, save_simulation_state
from datetime import datetime

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

    # Then check feasibility until you get your quota. (or run out of feasible moves)
    feasible_scored_moves = []
    for scored_move in unverified_scored_moves:
        _, ((pid1, _), (_, _), vec) = scored_move
        this_piece = pieces[pid1]
        test_mesh = this_piece['mesh'].copy()
        active_meshes = [piece['mesh'] for piece in active_pieces if piece['id'] != pid1]
        if check_path_clear(test_mesh, active_meshes, vec, 20):
            feasible_scored_moves.append(scored_move)
            if len(feasible_scored_moves) == k:
                break
    k = min(len(feasible_scored_moves), k)
    return feasible_scored_moves

def get_moves_scored_lookahead(pieces, active_pids, target_offsets, mates_list=None, top_k=[float("inf"), 10], rollout_depth=2, show_all_scores=False):
    print("Getting primary moves...")
    top_scored_moves = get_top_k_scored_moves(pieces, active_pids, target_offsets, k=top_k[0], mates_list=mates_list)
    print(f"| Looking Ahead from top {len(top_scored_moves)} moves...")

    args_list = [
        (scored_move, pieces, active_pids, target_offsets, mates_list, rollout_depth, top_k[1], show_all_scores)
        for scored_move in top_scored_moves
    ]
    new_scored_moves = []
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, scored_move in enumerate(executor.map(execute_2nd_lookahead, args_list)):
            new_scored_moves = new_scored_moves + scored_move
            if np.mod(i, 50) == 49:
                print(f"| | Evaluated {i+1:3d} moves of {len(top_scored_moves)}. Running Time: {time.time() - start_time:.2f}")

    new_scored_moves.sort(key=lambda x: sum(x[0]))
    return new_scored_moves

def execute_2nd_lookahead(args):
    scored_move, pieces, active_pids, target_offsets, mates_list, depth, top_k, show_all_scores = args
    primary_cost, ((pid1, cid1), (pid2, cid2), vec) = scored_move

    # Build a virtual assembly according to scored_move
    temp_piece = move_piece(copy.deepcopy(pieces[pid1]), vec)
    temp_pieces = copy.deepcopy(pieces)
    temp_pieces[pid1] = temp_piece
    temp_active_pids = active_pids + [temp_piece['id']] if temp_piece['id'] not in active_pids else active_pids

    top_scored_moves = get_top_k_scored_moves(temp_pieces, temp_active_pids, target_offsets, k=top_k, mates_list=mates_list)
    top_k = len(top_scored_moves)
    
    args_list = [
        (scored_move, temp_pieces, temp_active_pids, target_offsets, mates_list, depth)
        for scored_move in top_scored_moves
    ]
    new_scored_moves = []
    for i, _ in enumerate(top_scored_moves):
        best_scored_move = execute_greedy(args_list[i])
        new_scored_moves.append(best_scored_move)
        # print(f"| | | Evaluated move {i+1:3d}/{top_k}. Score: {best_move[0]:.2f}.")


    new_scored_moves.sort(key=lambda x: sum(x[0]))
    if show_all_scores:
        all_scored_moves = [([primary_cost] + scored_move[0], ((pid1, cid1), (pid2, cid2), vec)) for scored_move in new_scored_moves]
        return all_scored_moves
    else:
        total_scores = [primary_cost] + new_scored_moves[0][0]
        return total_scores, ((pid1, cid1), (pid2, cid2), vec)

def execute_greedy(args):
    move, pieces, active_pids, target_offsets, mates_list, depth = args
    primary_cost, ((pid1, cid1), (pid2, cid2), vec) = move

    temp_piece = move_piece(copy.deepcopy(pieces[pid1]), vec)
    temp_pieces = copy.deepcopy(pieces)
    temp_pieces[pid1] = temp_piece
    temp_active_pids = active_pids + [temp_piece['id']] if temp_piece['id'] not in active_pids else active_pids

    future_cost = greedy_rollout_score(
        temp_pieces,
        temp_active_pids,
        target_offsets,
        mates_list,
        depth=depth
    )
    scores = [primary_cost] + [future_cost]
    return (scores, ((pid1, cid1), (pid2, cid2), vec))

def greedy_rollout_score(pieces, active_pids, target_offsets, mates_list, depth=2):
    if depth == 0:
        return 0

    # start_time = time.time()
    best_cost = float("inf")

    best_move = get_top_k_scored_moves(pieces, active_pids, target_offsets, k=1, mates_list=mates_list)
    best_cost, ((best_pid, _), (_, _), best_vec) = best_move[0]
    # x, y, z = best_vec
    # print(f"| | | Greedy: Best move is p{best_pid} moving by <{x: .0f},{y: .0f},{z: .0f}>. Depth to go: {depth-1}. This took {time.time() - start_time:.2f}s.")
    if best_move is None or best_cost == 0:
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

def get_cost_change(piece, translation, target_offset):
    current_location = piece['mesh'].bounding_box.centroid
    current_cost = np.linalg.norm(current_location - target_offset)
    new_location = current_location + translation
    new_cost = np.linalg.norm(new_location - target_offset)
    return new_cost - current_cost


# Top-Level Scripts
def run_assembler(n_stages=16,top_k=[float("inf"), float("inf")], rollout_depth=3, show_all_scores=True, start_from=None, folder='results'):
    target_offsets = TARGET_OFFSETS
    target_offsets += [0,0,3]

    # Begin assembly set (active pids) with the floor. The floor is a static base piece. 
    if start_from is None:
        # Create Floor and Real Pieces
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
        state = load_simulation_state(start_from)
        pieces_augmented, active_pids, scored_moves = state
        print(f"| Loaded {len(scored_moves)} moves.")
        best_move = scored_moves[0]
        best_costs, ((best_pid, _), (other_pid, _), best_vec) = best_move
        lookahead_1, lookahead_2, rollout = best_costs
        x, y, z = best_vec
        print(f"| → Best move: Piece {best_pid} → {other_pid}, vec = <{x: .0f},{y: .0f},{z: .0f}>.")
        # Here, it is easiest to just execute the move and start from start_from+1. Most of the heavy lifting comes from calculating moves.
        moved_piece = pieces[best_pid]
        moved_piece = move_piece(moved_piece, best_vec)
        active_pids = active_pids + [best_pid] if best_pid not in active_pids else active_pids
        print(f"→ Fast-Finished {stage+1} / {n_stages}.")
        print(f"Score [Lookahead 1, Lookahead 2, Rollout]: [{lookahead_1:.2f}, {lookahead_2:.2f}, {rollout:.2f}] long-term.")

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
        scored_moves = get_moves_scored_lookahead(pieces_augmented, active_pids, target_offsets, mates_list, top_k=top_k, rollout_depth=rollout_depth, show_all_scores=show_all_scores)
        print(f"| | Processed {len(scored_moves)} moves.")
        best_move = scored_moves[0]
        best_costs, ((best_pid, _), (other_pid, _), best_vec) = best_move
        lookahead_1, lookahead_2, rollout = best_costs
        x, y, z = best_vec
        print(f"| → Best move: Piece {best_pid} → {other_pid}, vec = <{x: .0f},{y: .0f},{z: .0f}>.")
        save_simulation_state(stage, pieces_augmented, active_pids, scored_moves, folder=folder)
        
        # Now execute
        moved_piece = pieces[best_pid]
        moved_piece = move_piece(moved_piece, best_vec)
        active_pids = active_pids + [best_pid] if best_pid not in active_pids else active_pids
        print(f"→ Done with Stage {stage+1} / {n_stages}. This took {time.time() - start_time:.2f} seconds.")
        print(f"Score [Lookahead 1, Lookahead 2, Rollout]: [{lookahead_1:.2f}, {lookahead_2:.2f}, {rollout:.2f}]")

        if cost_function(pieces, target_offsets) < 0.1:
            return # If we're at zero cost, we're done.
    return

if __name__=="__main__":
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # folder = f"results/{timestamp}"
    folder = f"results/2025-04-23_22-54-56" # Sample folder
    folder_sim = f"{folder}/states"
    folder_img = f"{folder}/frames"

    start_time = time.time()
    # run_assembler(n_stages=16, top_k=[float("inf"), 1], rollout_depth=0, folder=folder_sim)

    # Visualize and save frames
    # for sc in range(16):
    #     state = load_simulation_state(sc, folder=folder_sim)
    #     pieces = state['pieces']
    #     if sc == 0: # Create Reference Pieces for frame 0
    #         reference_pieces = define_all_burr_pieces(reference=True)
    #         target_offsets = TARGET_OFFSETS
    #         target_offsets += [0,0,3]
    #         for piece in reference_pieces:
    #             piece = move_piece(piece, target_offsets[piece['id']-REFERENCE_INDEX_OFFSET])
    #             pieces.append(piece)
    #     scene = render_scene(pieces)
    #     save_animation_frame(scene, sc, folder=folder_img)
    #     # scene.show() # Uncomment to drag around

    #     floor = create_floor()
    #     arrows = show_moves_scored(state['available_moves'], state['pieces'], floor)
    #     scene = render_scene(state['pieces'], arrows=arrows)
    #     save_animation_frame(scene, sc, folder=folder_img, suffix='a')
    #     # scene.show() # Uncomment to drag around

    compile_gif(folder=folder_img)
    print(f"This script took {time.time() - start_time:4f} seconds")