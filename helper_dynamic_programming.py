import numpy as np
import copy
import concurrent.futures
from tqdm import tqdm
from helper_burr_piece_creator import FLOOR_INDEX_OFFSET
from helper_geometric_functions import check_path_clear, get_feasible_motions,\
move_piece, get_unsupported_pids

NUM_PIECES = 6

# Dynamic Programming Functions
def get_top_k_scored_moves(pieces, assembly, k=float("inf"), max_held=2):
    if assembly["prev_piece_and_vec"] is not None:
        prev_pid, prev_vec = assembly["prev_piece_and_vec"]
    else:
        prev_pid, prev_vec = -1, 0

    # First get a headcount of who's unsupported, except the floor
    pieces_not_floor = [piece for piece in pieces if piece['id'] != FLOOR_INDEX_OFFSET]
    unsupported_ids = get_unsupported_pids(pieces_not_floor)
    for piece in pieces_not_floor:
        piece['gripper_config']['active'] = False
    for id in unsupported_ids:
        piece = pieces[id]
        piece['gripper_config']['active'] = True # Activate gripper for unsupported pieces
    # If we're at the limit, only these unsupported pieces can get moved. Otherwise, carry on.
    movable_ids = [i for i in range(NUM_PIECES)]
    if len(unsupported_ids) >= max_held:
        movable_ids = unsupported_ids
    # If reversed, filter by not active PIDs
    if assembly["reverse"]:
        movable_ids = list(set(movable_ids) & set(assembly["active_pids"]))
    
    # Second, quickly make a broad list of semi-verified moves, not checking for intermediate collision
    unverified = []
    unverified_counts = []
    active_pieces = [piece for piece in pieces if piece['id'] in assembly["active_pids"]]
    for id in movable_ids:
        piece = [piece for piece in pieces if piece['id'] == id][0]
        new_unverified = get_feasible_motions(piece, active_pieces, steps=1, check_path=False)
        unverified = unverified + new_unverified
        unverified_counts.append(len(new_unverified))

    # Then sort by score. 
    unverified_scored_moves = []
    for (pid1, cid1), (pid2, cid2), vec in unverified:
        if np.linalg.norm(vec) <= 0.01:
            continue # Don't collect null moves
        if pid1 == prev_pid and np.linalg.norm(vec + prev_vec) <= 0.01:
            continue # Don't reverse the previous move
        cost_change = get_cost_change(pieces[pid1], vec)
        unverified_scored_moves.append((cost_change, ((pid1, cid1), (pid2, cid2), vec)))
    unverified_scored_moves.sort(key=lambda x: x[0])

    # Then check feasibility until you get your quota. (or run out of feasible moves)
    feasible_scored_moves = []
    for scored_move in unverified_scored_moves:
        _, ((pid1, _), (_, _), vec) = scored_move
        this_piece = pieces[pid1]
        test_mesh = this_piece['mesh'].copy()
        other_pieces = [piece for piece in active_pieces if piece['id'] != pid1]
        other_meshes = [piece['mesh'] for piece in other_pieces]
        if check_path_clear(test_mesh, other_meshes, vec, 20):
            # If removing THIS piece leads to 2+ unsupported pieces, it won't work. 
            if len(get_unsupported_pids(other_pieces)) >= max_held:
                continue
            feasible_scored_moves.append(scored_move)
            if len(feasible_scored_moves) == k:
                break
    k = min(len(feasible_scored_moves), k)
    return feasible_scored_moves

def get_moves_scored_lookahead(pieces, assembly, top_k=[float("inf")], rollout_depth=2):
    print("\n\nGetting primary moves...")
    # Unrolling branches is good for parallel performance
    branches, remaining_top_k = generate_unrolled_branches(pieces, assembly, top_k)
    print(f"| Unrolled to get {len(branches)} moves...")

    args_list = [
        (path, temp_pieces, temp_assembly, rollout_depth, remaining_top_k)
        for path, temp_pieces, temp_assembly in branches
    ]
    path_list = []
    total_moves_compared = 0
    # two methods: serial or parallel
    # for i, move in tqdm(enumerate(branches), total=len(branches), ncols=100, desc='| '):
    #     result = execute_lookahead_recursive(*args_list[i])
    #     path, num_moves = result
    #     path_list.append(path)
    #     total_moves_compared += num_moves
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(execute_lookahead_recursive, *args) for args in args_list]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), ncols=100, desc='| '):
            result = future.result()
            path, num_moves = result
            path_list.append(path)
            total_moves_compared += num_moves

    # Break ties by picking the best immediate move
    presorted_paths = sorted(path_list, key=lambda x: x[0][0]) 
    postsorted_paths = sorted(presorted_paths, key=lambda x: np.trunc(sum([move[0] for move in x])*1e7))
    return postsorted_paths, total_moves_compared

def generate_unrolled_branches(pieces, assembly, top_k, min_branches=50):
    queue = [([], pieces, assembly)]
    level = 0
    while len(queue) < min_branches and level < len(top_k):
        new_queue = []
        for path, temp_pieces, temp_assembly in queue:
            moves = get_top_k_scored_moves(temp_pieces, temp_assembly, k=top_k[level])
            for move in moves:
                primary_cost, ((pid1, cid1), (pid2, cid2), vec) = move
                new_piece = move_piece(copy.deepcopy(temp_pieces[pid1]), vec)
                new_pieces = copy.deepcopy(temp_pieces)
                new_pieces[pid1] = new_piece
                new_assembly = copy.deepcopy(temp_assembly)
                if temp_assembly["reverse"]:
                    if np.linalg.norm(new_piece['mesh'].bounding_box.centroid - new_piece['target']) <= 0.01:
                        new_assembly["active_pids"].remove(pid1)
                else:
                    if new_piece['id'] not in new_assembly["active_pids"]:
                        new_assembly["active_pids"] = new_assembly["active_pids"] + [new_piece['id']]
                new_queue.append((path + [move], new_pieces, new_assembly))
        queue = new_queue
        level += 1
        print(f"| Unrolling... Got {len(queue)} paths")
    return queue, top_k[level:]

def execute_lookahead_recursive(path, pieces, assembly, rollout_depth, top_k_remaining):
    """
    Recursive function that searches for the highest scoring future path, given a present one.
    Returns the augmented path and a count of all the children searched. 
    """

    # Build a virtual assembly according to scored_move
    scores = []
    for move in path:
        score, _ = move[0], move[1]
        scores.append(score)

    if not top_k_remaining:
        rollout_score = greedy_rollout_score(pieces, assembly, depth=rollout_depth)
        path += [(np.round(rollout_score, 9), ['rollout'])]
        return path, 1

    next_k = top_k_remaining[0]
    next_moves = get_top_k_scored_moves(pieces, assembly, k=next_k)
    
    path_list = []
    num_children = 0
    for move in next_moves:
        new_path = path + [move]
        new_path, num_child = execute_lookahead_recursive(new_path, pieces, assembly, rollout_depth, top_k_remaining[1:])
        num_children += num_child
        path_list.append(new_path)

    # Choose best path
    # Break ties by picking the best immediate move
    presorted_paths = sorted(path_list, key=lambda x: x[0][0]) 
    postsorted_paths = sorted(presorted_paths, key=lambda x: np.trunc(sum([move[0] for move in x])*1e7))
    best_path = postsorted_paths[0]
    return best_path, num_children

def greedy_rollout_score(pieces, assembly, depth=2):
    if depth == 0:
        return 0

    best_cost = float("inf")

    best_move = get_top_k_scored_moves(pieces, assembly, k=1)
    if best_move == []:
        return 0 # No valid move
    best_cost, ((best_pid, _), (_, _), best_vec) = best_move[0]
    if best_cost >= 0:
        return 0  # Best greedy move is no move at all.

    # Execute best move
    temp_piece = move_piece(copy.deepcopy(pieces[best_pid]), best_vec)
    temp_pieces = copy.deepcopy(pieces)
    temp_pieces[best_pid] = temp_piece
    temp_assembly = copy.deepcopy(assembly)
    if temp_assembly["reverse"]:
        if np.linalg.norm(temp_piece['mesh'].bounding_box.centroid - temp_piece['target']) <= 0.01:
            temp_assembly["active_pids"].remove(best_pid)
    else:
        if temp_piece['id'] not in temp_assembly["active_pids"]:
            temp_assembly["active_pids"] = temp_assembly["active_pids"] + [temp_piece['id']]
    
    return best_cost + greedy_rollout_score(temp_pieces, temp_assembly, depth=depth-1)

def cost_function(pieces):
    diff = [piece['mesh'].bounding_box.centroid - piece['target'] for piece in pieces]
    dist = [np.linalg.norm(dif) for dif in diff]
    return sum(dist)

def get_cost_change(piece, translation):
    target_offset = piece['target']
    current_location = piece['mesh'].bounding_box.centroid
    current_cost = np.linalg.norm(current_location - target_offset)
    new_location = current_location + translation
    new_cost = np.linalg.norm(new_location - target_offset)
    diff_cost = new_cost - current_cost
    diff_cost += 0.001 # Constant cost to encourage fewer moves (and break ties)
    return np.round(diff_cost, 9)