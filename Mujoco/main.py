import mujoco
import mujoco.viewer
import numpy as np
import time
import copy
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
import re
from helper_burr_piece_creator import create_floor, define_all_burr_pieces, FLOOR_INDEX_OFFSET
from helper_geometric_functions import check_path_clear, get_feasible_motions, get_valid_mates, move_piece, get_unsupported_pids
from helper_display import compile_gif  # Keep only compile_gif, remove show_and_save_frames
from helper_file_mgmt import load_mates_list, load_simulation_state, save_mates_list, save_simulation_state
from datetime import datetime
import pyglet
from functools import wraps
import warnings
from PIL import Image

# Nomenclature and constants from main_trimesh.py
np.set_printoptions(formatter={'int': '{:2d}'.format})
NUM_PIECES = 6
SCALE = 0.02  # meters per grid unit
TARGET_OFFSETS = np.array([[0,0,1],[-1,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,-1]])
# array([[ 0.  ,  0.  ,  0.02],
#        [-0.02,  0.  ,  0.  ],
#        [ 0.02,  0.  ,  0.  ],
#        [ 0.  ,  0.02,  0.  ],
#        [ 0.  , -0.02,  0.  ],
#        [ 0.  ,  0.  , -0.02]])
START_OFFSETS = np.array([[4,-8,1],[-8,0,1],[8,0,1],[-4,8,3],[-4,-8,3],[4,8,1]])
# array([[ 0.08, -0.16,  0.02],
#        [-0.16,  0.  ,  0.02],
#        [ 0.16,  0.  ,  0.02],
#        [-0.08,  0.16,  0.06],
#        [-0.08, -0.16,  0.06],
#        [ 0.08,  0.16,  0.02]])
## Friction helpers removed; all planning uses geometry-only checks

# Load MuJoCo model from scene.xml
model = mujoco.MjModel.from_xml_path('aloha/scene.xml')
data = mujoco.MjData(model)

def joint_index_for_piece(model, piece_id):
    """Resolve the MuJoCo joint index for a given burr piece by name, with fallbacks.
    This avoids assuming joint indices == piece IDs, which is brittle.
    """
    # Try common naming patterns; adjust if your aloha/scene.xml uses different names
    candidates = [f"piece{piece_id}_free", f"piece{piece_id}", f"p{piece_id}_free"]
    for name in candidates:
        try:
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        except Exception:
            continue
    warnings.warn(f"[WARN] Falling back to joint index == piece_id for piece {piece_id}. Verify XML joint names.")
    return piece_id

def freejoint_addresses(model, piece_id):
    """Return (joint_id, qpos_adr, qvel_adr) for the free joint on body 'piece{piece_id}'.
    This walks the body->joint mapping so it works regardless of other robots/joints in the model.
    """
    body_name = f"piece{piece_id}"
    try:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    except Exception as e:
        raise ValueError(f"Body '{body_name}' not found in model") from e
    jadr = model.body_jntadr[bid]
    jnum = model.body_jntnum[bid]
    for j in range(jnum):
        jid = jadr + j
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            qpos_adr = model.jnt_qposadr[jid]
            qvel_adr = model.jnt_dofadr[jid]
            return jid, qpos_adr, qvel_adr
    raise ValueError(f"No free joint found for body '{body_name}'")

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
        
        # Geometry-only cost calculation (no friction)
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
        
        # Geometry-only path checking (no friction)
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

def execute_lookahead_recursive(scored_move, pieces, active_pids, target_offsets, mates_list, rollout_depth, top_k_remaining):
    primary_cost, ((pid1, cid1), (pid2, cid2), vec) = scored_move
    
    # Convert vec back to numpy array if it's a list
    vec = np.array(vec) if isinstance(vec, list) else vec
    target_offsets = np.array(target_offsets) if isinstance(target_offsets, list) else target_offsets

    # Build a virtual assembly according to scored_move (geometry-only)
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
    if not best_move:
        return 0
    best_cost, ((best_pid, _), ((_, _)), best_vec) = best_move[0]
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
    offsets = np.array([piece['mesh'].bounding_box.centroid for piece in pieces])
    diff = offsets - target_offsets
    dist = np.linalg.norm(diff, axis=1)
    return float(np.sum(dist))

# Update the get_cost_change function to maintain backwards compatibility
def get_cost_change(piece, translation, target_offset):
    current_location = piece['mesh'].bounding_box.centroid
    current_cost = np.linalg.norm(current_location - target_offset)
    new_location = current_location + translation
    new_cost = np.linalg.norm(new_location - target_offset)
    diff_cost = new_cost - current_cost
    return np.round(diff_cost, 9)

# Function to update piece positions in MuJoCo data based on moves
def update_mujoco_positions(pieces, data):
    for piece in pieces:
        if piece['id'] < NUM_PIECES:  # Only update burr pieces, not floor
            _, qpos_adr, _ = freejoint_addresses(model, piece['id'])
            centroid = piece['mesh'].bounding_box.centroid * SCALE
            # Set position (first 3 elements of the 7-DOF free joint)
            data.qpos[qpos_adr : qpos_adr + 3] = centroid
            # Set orientation to identity quaternion (next 4 elements: w, x, y, z)
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

def show_and_save_frames_mujoco(folder_sim, folder_img, hold=False, reverse=False, steps=20):
    """
    Load simulation states, update MuJoCo positions, render scenes, and save frames as images.
    """
    sim_folder = Path(folder_sim)
    img_folder = Path(folder_img)
    img_folder.mkdir(parents=True, exist_ok=True)
    
    step_files = sorted(sim_folder.glob('step_*.pkl'))
    if not step_files:
        print("No simulation states found to render.")
        return
    
    # Initialize MuJoCo renderer (using offscreen rendering for saving images)
    renderer = mujoco.Renderer(model, width=640, height=480)  # Adjust resolution as needed
    
    for step_file in tqdm(step_files, desc="Rendering frames"):
        step_num = int(re.search(r'step_(\d+)\.pkl', step_file.name).group(1))
        state = load_simulation_state(step_num, folder=folder_sim)
        pieces_augmented = state['pieces']
        pieces = [piece for piece in pieces_augmented if piece['id'] != FLOOR_INDEX_OFFSET]
        
        # Update MuJoCo positions from the state
        update_mujoco_positions(pieces, data)
        mujoco.mj_forward(model, data)  # Ensure kinematics are updated
        
        # Render the scene
        renderer.update_scene(data)
        img = renderer.render()
        
        # Save the image (convert RGB to PIL Image and save as PNG)
        pil_img = Image.fromarray(img)
        img_path = img_folder / f"frame_{step_num:04d}.png"
        pil_img.save(img_path)
    
    renderer.close()
    print(f"Frames saved to {img_folder}")

def simulate_move_in_mujoco(piece_id, target_position, data, model, steps=50, dt=0.01):
    """
    Simulate movement in MuJoCo by setting velocities and stepping the simulation.
    piece_id: ID of the piece (0-5)
    target_position: Target [x, y, z] position
    steps: Number of simulation steps
    dt: Time step (seconds)
    """
    jid, qpos_adr, qvel_adr = freejoint_addresses(model, piece_id)
    current_pos = data.qpos[qpos_adr : qpos_adr + 3]
    velocity = (target_position - current_pos) / (steps * dt)  # Velocity to reach target
    
    # Set velocity for position (first 3 DOFs)
    data.qvel[qvel_adr : qvel_adr + 3] = velocity
    # Set angular velocity to zero (next 3 DOFs, assuming no rotation)
    data.qvel[qvel_adr + 3 : qvel_adr + 6] = [0.0, 0.0, 0.0]
    
    # Step the simulation
    for _ in range(steps):
        mujoco.mj_step(model, data)
        time.sleep(dt)  # Optional: Slow down for visualization
    
    # Clear velocities after move
    data.qvel[qvel_adr : qvel_adr + 6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Return the final position
    return data.qpos[qpos_adr : qpos_adr + 3]

# Modified run_assembler to integrate with MuJoCo
def run_assembler_mujoco(n_stages=16, top_k=[float("inf")], rollout_depth=0, start_from=None, folder='results', reverse=False):
    # Removed unused use_friction parameter
    # Similar setup as in main_trimesh.py
    sim_folder = Path(folder)
    step_files = sorted(sim_folder.glob('step_*.pkl'))
    step_indices = sorted([int(re.search(r'step_(\d+)\.pkl', f.name).group(1)) for f in step_files])
    latest_step = step_indices[-1] if step_indices else None

    if reverse:
        target_offsets = START_OFFSETS
    else:
        target_offsets = TARGET_OFFSETS + np.array([0,0,3])

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

    # Launch viewer at the start for live visualization
    viewer = mujoco.viewer.launch_passive(model, data)

    if latest_step is None:
        floor = create_floor(reverse=reverse)
        if reverse:
            # Reverse mode: assembled target lifted by +3 grid units
            pieces = define_all_burr_pieces(TARGET_OFFSETS + np.array([0,0,3]))
            active_pids = [i for i in range(NUM_PIECES)] + [FLOOR_INDEX_OFFSET]
        else:
            pieces = define_all_burr_pieces(START_OFFSETS)
            active_pids = [FLOOR_INDEX_OFFSET]
        pieces_augmented = pieces + [floor]
        print(f"Initial Cost Measure: {cost_function(pieces, target_offsets):.2f}")
        start_from = 0
        # Sync MuJoCo positions with current Trimesh piece centroids
        update_mujoco_positions(pieces, data)
        mujoco.mj_forward(model, data)
        viewer.sync()
        # Debug: print initial positions
        try:
            positions = []
            for i in range(NUM_PIECES):
                _, qpos_adr, _ = freejoint_addresses(model, i)
                pos = np.array(data.qpos[qpos_adr : qpos_adr + 3])
                cen = np.array(pieces[i]['mesh'].bounding_box.centroid)
                positions.append(pos)
                print(f"[DEBUG] piece {i} centroid={cen.round(3)} qpos={pos.round(3)}")
            # Try to center viewer on mean position of pieces
            mean_pos = np.mean(positions, axis=0)
            try:
                viewer.cam.lookat[:] = mean_pos
                viewer.cam.distance = 0.6
            except Exception:
                pass
        except Exception as e:
            print(f"[DEBUG] startup pose debug failed: {e}")
        # Optional: seed one burr piece active if only floor is active to bootstrap planning
        if len([p for p in active_pids if p != FLOOR_INDEX_OFFSET]) == 0:
            active_pids.append(0)
            print("[INFO] Seeding active set with piece 0 to bootstrap motion search.")
        if cost_function(pieces, target_offsets) < 0.1:
            save_simulation_state(start_from+1, pieces_augmented, active_pids, [], folder=folder)
            return
        # Do not increment start_from; we want to run stage 0 when starting fresh

    def validate_mates_list(mates_list):
        if mates_list is None:
            return False, "mates_list is None"
        counts = {}
        for a, b in mates_list:
            counts[a] = counts.get(a, 0) + 1
            counts[b] = counts.get(b, 0) + 1
        missing = [i for i in range(NUM_PIECES) if counts.get(i, 0) == 0]
        if missing:
            return False, f"pieces with no mates: {missing}"
        return True, "ok"

    mates_list = load_mates_list(reverse=reverse)
    if mates_list is None:
        mates_list = get_valid_mates(pieces, floor)
        save_mates_list(mates_list, reverse=reverse)
    ok, msg = validate_mates_list(mates_list)
    if not ok:
        print(f"[WARN] mates_list invalid: {msg}. Regenerating.")
        mates_list = get_valid_mates(pieces, floor)
        save_mates_list(mates_list, reverse=reverse)

    for stage in range(start_from, n_stages):
        start_time = time.time()
        scored_moves, moves_compared = get_moves_scored_lookahead(pieces_augmented, active_pids, target_offsets, mates_list, top_k=top_k, rollout_depth=rollout_depth)
        print(f"| | Processed {moves_compared} moves.")
        if not scored_moves:
            print("[WARN] No valid moves found. Regenerating mates_list and retrying once...")
            mates_list = get_valid_mates(pieces, floor)
            save_mates_list(mates_list, reverse=reverse)
            scored_moves, moves_compared = get_moves_scored_lookahead(pieces_augmented, active_pids, target_offsets, mates_list, top_k=top_k, rollout_depth=rollout_depth)
            print(f"| | Re-Processed {moves_compared} moves after mates_list regeneration.")
        if not scored_moves:
            print(f"No valid moves found at stage {stage}. Assembly might be complete or stuck.")
            save_simulation_state(stage, pieces_augmented, active_pids, [], folder=folder)
            return
        best_pid, best_vec = print_best_move_info(scored_moves)
        save_simulation_state(stage, pieces_augmented, active_pids, scored_moves, folder=folder)
        
        # Simulate the move in MuJoCo environment
        current_pos = pieces[best_pid]['mesh'].bounding_box.centroid
        target_pos_m = (current_pos + best_vec) * SCALE
        final_pos_m = simulate_move_in_mujoco(best_pid, target_pos_m, data, model)
        final_pos = final_pos_m / SCALE
        
        # Update Trimesh piece with MuJoCo's final position (apply_translation is in-place)
        pieces[best_pid]['mesh'].apply_translation(final_pos - current_pos)
        pieces[best_pid]['gripper_config'][1] = True
        active_pids = active_pids + [best_pid] if best_pid not in active_pids else active_pids
        
        # Sync viewer to show the move
        viewer.sync()
        
        print(f"→ Done with Stage {stage+1} / {n_stages}. This took {time.time() - start_time:.2f} seconds.")
        if cost_function(pieces, target_offsets) < 0.1:
            save_simulation_state(stage+1, pieces_augmented, active_pids, [], folder=folder)
            return

# Include other functions from main_trimesh.py as needed (get_top_k_scored_moves, etc.)
# ... (copy the relevant functions here)

def show_final_assembly_mujoco(reverse=True):
    """
    Place pieces directly into their final assembled configuration in MuJoCo and display.
    - reverse=True: shows target assembly lifted by +3 grid units (to float above floor)
    - reverse=False: shows base target assembly position (with +3 applied to target_offsets in run)
    """
    # Build pieces in their target configuration (grid units)
    if reverse:
        target_offsets = TARGET_OFFSETS + np.array([0, 0, 3])
    else:
        target_offsets = TARGET_OFFSETS + np.array([0, 0, 3])

    pieces = define_all_burr_pieces(target_offsets)
    floor = create_floor(reverse=reverse)
    pieces_augmented = pieces + [floor]

    # Sync MuJoCo with these positions (convert to meters inside updater)
    update_mujoco_positions(pieces, data)
    mujoco.mj_forward(model, data)

    # Launch passive viewer to display final assembly
    viewer = mujoco.viewer.launch_passive(model, data)
    try:
        # Keep the viewer open until the window is closed
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)
    except Exception:
        pass

if __name__ == "__main__":
    reverse = True
    # Run planner and move pieces step-by-step (friction removed)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder = f"results/{timestamp}"
    folder_sim = f"{folder}/states"
    start_time = time.time()
    run_assembler_mujoco(n_stages=1, top_k=[100, 10], rollout_depth=1, folder=folder_sim, reverse=reverse)
    print(f"This script took {time.time() - start_time:.4f} seconds to run.")