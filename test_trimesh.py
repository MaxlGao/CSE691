import trimesh
import numpy as np
import time
import copy
import imageio
import os

np.set_printoptions(formatter={'int': '{:2d}'.format})
REFERENCE_INDEX_OFFSET = 50 # Bump up reference piece indices
FLOOR_INDEX_OFFSET = 100 # Bump up floor piece index

def find_cube_corners(mesh, tol=1e-6):
    """
    Returns a list of tuples (index, TrackedArray([x, y, z])). xyz is position in body frame. 
    """
    corners = []
    vertices = mesh.vertices
    edges = mesh.edges_unique
    edge_vertices = vertices[edges]
    edge_dirs = edge_vertices[:, 1] - edge_vertices[:, 0]

    # Normalize directions
    edge_dirs = np.array([v / np.linalg.norm(v) for v in edge_dirs])

    # Build map from vertex index to connected edge directions
    vertex_to_dirs = {}
    for i, edge in enumerate(edges):
        for vid in edge:
            vertex_to_dirs.setdefault(int(vid), []).append(edge_dirs[i])
            a = 0

    for vidx, dirs in sorted(vertex_to_dirs.items()):
        if len(dirs) < 3:
            continue

        found = False
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                for k in range(j + 1, len(dirs)):
                    d1, d2, d3 = dirs[i], dirs[j], dirs[k]
                    if (
                        abs(np.dot(d1, d2)) < tol and
                        abs(np.dot(d1, d3)) < tol and
                        abs(np.dot(d2, d3)) < tol
                    ):
                        corners.append((vidx, vertices[vidx]))
                        found = True
                        break
                if found:
                    break
            if found:
                break
    
    return corners

def create_floor():
    """
    Create a Floor. Returns a dict like create_burr_piece
    """
    floor = trimesh.creation.box(extents=[20,20,1])
    floor.apply_translation([0, 0, -0.5])
    floor.visual.face_colors = [200, 200, 200, 255]

    # Only points on the floor where piece 3 and 4's corners touch
    corners = [(0,[-1,0,0]), (1,[1,0,0]),
               (2,[-1,2,0]), (3,[1,2,0]),
               (4,[-1,-2,0]), (5,[1,-2,0])] 
    return {'mesh': floor, 'corners': corners, 'id': FLOOR_INDEX_OFFSET}

def create_burr_piece(blocks, color, idx, reference=False):
    """
    Create a Burr piece from a list of block specs.
    Each block is a dict: {'size': [x, y, z], 'position': [x, y, z]}
    Returns a dict: {'mesh': trimesh.Trimesh, 
                     'corners': list of (index, [x, y, z]),
                     'id': id of piece}
    """
    meshes = []
    for block in blocks:
        size = block['size']
        position = block['position']
        box = trimesh.creation.box(extents=size)
        box.apply_translation(position)
        # box.visual.face_colors = color
        meshes.append(box)

    composite = trimesh.boolean.union(meshes)
    composite.process(validate=True)
    composite.visual.face_colors = color
    corners = find_cube_corners(composite)
    if reference:
        idx += REFERENCE_INDEX_OFFSET
    return {'mesh': composite, 'corners': corners, 'id': idx}

def define_all_burr_pieces(reference=False):
    """
    Returns a list of dicts: {'mesh': trimesh.Trimesh, 'corners': list of (index, position)}
    """
    if reference:
        opacity = 50
    else:
        opacity = 255
    colors = [
        [255, 0, 0, opacity],    # Red
        [255, 127, 0, opacity],  # Orange
        [255, 255, 0, opacity],  # Yellow
        [0, 255, 0, opacity],    # Green
        [0, 0, 255, opacity],    # Blue
        [255, 0, 255, opacity]   # Magenta
    ]

    pieces = []

    pieces.append(create_burr_piece([
        {'size': [6, 2, 2], 'position': [0, 0, 0]},
    ], colors[0], 0, reference=reference))
    pieces.append(create_burr_piece([
        {'size': [1, 2, 1], 'position': [-0.5, 0, -0.5]},
        {'size': [1, 2, 2], 'position': [-0.5, -2, 0]},
        {'size': [1, 2, 2], 'position': [-0.5, 2, 0]},
        {'size': [1, 1, 2], 'position': [0.5, -2.5, 0]},
        {'size': [1, 1, 2], 'position': [0.5, 2.5, 0]},
    ], colors[1], 1, reference=reference))
    pieces.append(create_burr_piece([
        {'size': [2, 2, 2], 'position': [0, -2, 0]},
        {'size': [1, 2, 1], 'position': [0.5, 0, -0.5]},
        {'size': [1, 2, 2], 'position': [0.5, 2, 0]},
        {'size': [1, 1, 2], 'position': [-0.5, 2.5, 0]},
        {'size': [1, 1, 1], 'position': [-0.5, -0.5, -0.5]},
    ], colors[2], 2, reference=reference))
    pieces.append(create_burr_piece([
        {'size': [2, 1, 6], 'position': [0, 0.5, 0]},
        {'size': [2, 1, 1], 'position': [0, -0.5, 2.5]},
        {'size': [2, 1, 1], 'position': [0, -0.5, -0.5]},
        {'size': [2, 1, 1], 'position': [0, -0.5, -2.5]},
    ], colors[3], 3, reference=reference))
    pieces.append(create_burr_piece([
        {'size': [2, 2, 1], 'position': [0, 0, -2.5]},
        {'size': [2, 2, 1], 'position': [0, 0, 2.5]},
        {'size': [2, 1, 1], 'position': [0, -0.5, -1.5]},
        {'size': [2, 1, 1], 'position': [0, -0.5, 1.5]},
        {'size': [1, 1, 2], 'position': [-0.5, -0.5, 0]},
        {'size': [1, 1, 1], 'position': [-0.5, 0.5, -0.5]},
    ], colors[4], 4, reference=reference))
    pieces.append(create_burr_piece([
        {'size': [6, 2, 1], 'position': [0, 0, -0.5]},
        {'size': [1, 2, 1], 'position': [2.5, 0, 0.5]},
        {'size': [1, 2, 1], 'position': [-2.5, 0, 0.5]},
    ], colors[5], 5, reference=reference))
    return pieces



def check_path_clear(this_piece, other_pieces, translation, steps, tol=0.01):
    """
    Check whether this_piece can be translated without colliding with other piece(s)
    at any step along the way. Handles both single-piece and multi-piece cases.

    Args:
        this_piece: trimesh.Trimesh - The moving piece
        other_pieces: Union[trimesh.Trimesh, List[trimesh.Trimesh]] - Either a single piece or list of pieces
        translation: np.ndarray, shape (3,) - Translation vector
        steps: int - Number of interpolation steps to test
    Returns:
        bool - True if path is clear
    """
    # Convert single piece to list for uniform handling
    if not isinstance(other_pieces, (list, tuple)):
        other_pieces = [other_pieces]
    
    # Create vertex sample points for fast collision detection
    # sample_points = this_piece.vertices - 0.05 * this_piece.vertex_normals

    this_bbox = this_piece.bounds
    for step in reversed(range(1, steps + 1)):
        frac_translation = (translation * step) / (steps)
        test_bbox = this_bbox + frac_translation
        # test_points = sample_points + frac_translation
        test_piece = this_piece.copy()
        test_piece.apply_translation(frac_translation)

        for other_piece in other_pieces:
            other_bbox = other_piece.bounds
            if not (all(test_bbox[0] < other_bbox[1]) and # Check for lower test < upper other
                    all(test_bbox[1] > other_bbox[0])):   # Check for upper test > lower other
                continue # If the boxes don't touch, don't bother

            # Check sample points and exit early if there's a violation
            # if any(other_piece.nearest.signed_distance(test_points) > tol):
                # return False
            # Otherwise, do full volumetric collision detection.
            if trimesh.boolean.intersection([test_piece, other_piece], check_volume=False).volume > tol:
                return False

    return True  # No collision detected along the entire path

def get_feasible_motions(this_piece, pieces, valid_mates=None, steps=20, check_collision=True):
    """
    Compute feasible, unique linear motions from `this_piece` to any other in `pieces`.

    Parameters:
        - this_piece: dict with 'mesh', 'corners', 'id'
        - pieces: list of all pieces
        - valid_mates: optional dict of {pid1: ((pid1, cid1), (pid2, cid2))} to restrict testing
        - steps: number of interpolation steps for collision checks (1 means skip motion path check)
        - tolerance: rounding tolerance for deduplication
    Returns:
        List of ((this_pid, this_cid), (other_pid, other_cid), [dx, dy, dz])
    """
    this_pid = this_piece['id']
    this_corners = this_piece['corners']

    # Convert single piece to list for uniform handling
    if not isinstance(pieces, (list, tuple)):
        pieces = [pieces]
    other_meshes = [piece["mesh"] for piece in pieces if piece["id"] != this_pid]
    # Convert single piece to list for uniform handling
    if not isinstance(other_meshes, (list, tuple)):
        other_meshes = [other_meshes]
    other_corners = []
    for piece in pieces:
        if piece["id"] == this_piece["id"]:
            continue
        for cid, pos in piece["corners"]:
            other_corners.append((piece["id"], cid, pos))
    # Convert single piece to list for uniform handling
    if not isinstance(other_corners, (list, tuple)):
        other_corners = [other_corners]

    # Restricted motions in (+x, -x, +y, -y, +z, -z)
    restricted = [False, False, False, False, False, False]
        
    if valid_mates is not None:
        mates_list = valid_mates[this_pid] # Mates pertaining to this piece
        # Match corners to actual positions
        this_corner_dict = {cid: pos for cid, pos in this_corners}
        other_corner_dict = {(pid, cid): pos for (pid,cid,pos) in other_corners}
        available_pieces = {p['id'] for p in pieces}
        combinations = [
            ((this_pid, cid1), (pid2, cid2), other_corner_dict[(pid2, cid2)] - this_corner_dict[cid1])
            for ((pid1, cid1), (pid2, cid2)) in mates_list
            if pid2 in available_pieces
        ]
        # Rule: You cannot move in (.., .., +x) if you cannot move in (0, 0, +x), etc
        dd = 0.25
        test_vecs = np.array([[dd, 0, 0], [-dd, 0, 0],
                              [0, dd, 0], [0, -dd, 0],
                              [0, 0, dd], [0, 0, -dd]])
        for i, test_vec in enumerate(test_vecs):
            test_piece = this_piece['mesh'].copy()
            if not check_path_clear(test_piece, other_meshes, test_vec, 1):
                restricted[i] = True
    else:
        combinations = []
        for cid1, pos1 in this_corners:
            for pid2, cid2, pos2 in other_corners:
                vec = pos2 - pos1
                combinations.append(
                    ((this_pid, cid1), (pid2, cid2), vec)
                )


    unique_motions = {}
    feasible_motions = []

    for (p1, c1), (p2, c2), vec in combinations:
        tol = 0.001 # Ignore small movements
        if any([vec[0] > tol and restricted[0],
                vec[0] < -tol and restricted[1],
                vec[1] > tol and restricted[2],
                vec[1] < -tol and restricted[3],
                vec[2] > tol and restricted[4],
                vec[2] < -tol and restricted[5]]):
            continue # Vec uses a restricted path; count as colliding
        key = tuple(np.round(vec, 4))  # avoid float fuzz
        if key in unique_motions:
            continue # Already seen this transformation
        unique_motions[key] = True # Save to seen motions if not

        # Check Path of Motion
        test_piece = this_piece['mesh'].copy()
        if not check_collision or check_path_clear(test_piece, other_meshes, vec, steps):
            feasible_motions.append(((p1, c1), (p2, c2), vec))

    return feasible_motions

def get_valid_mates(pieces, floor):
    """
    Precompute and store valid mating corner pairs for each piece-to-piece combo.
    Returns a dict of lists of mates (p1,c1),(p2,c2)
    """
    print("Precomputing mates...")
    # Augment piece list with floor
    all_pieces = copy.deepcopy(pieces)
    all_pieces.append(copy.deepcopy(floor))
    cache = {}
    cache_length = 0
    for p1 in pieces:
        start_time = time.time()
        pid = p1['id']
        valid_motions = []
        for p2 in all_pieces:
            that_pid = p2['id']
            if pid == that_pid:
                continue
            valid_motions_p2 = get_feasible_motions(p1, p2, steps=1)
            valid_motions = valid_motions + valid_motions_p2
        valid_mates = [motion[:2] for motion in valid_motions]
        cache[pid] = valid_mates
        cache_length += len(valid_mates)
        print(f"| Got {len(valid_mates):4d} mates for piece {pid}. This took {(time.time()-start_time):.2f} seconds.")
    print(f"→ For this assembly, there are {cache_length} valid piece-to-piece mates")
    return cache


def create_arrow(move, pieces, floor, color=[0, 0, 0, 255]):
    """
    Create a line-shaped arrow from start to end using a path.
    
    Returns a trimesh.path.Path3D object.
    """
    (pid1, cid1), (pid2, cid2), vec = move
    start = pieces[pid1]['corners'][cid1][1]
    if pid2 == FLOOR_INDEX_OFFSET:
        end = floor['corners'][cid2][1]
    else:
        end = pieces[pid2]['corners'][cid2][1]
    path = trimesh.load_path(np.array([
        [start, end]
    ]))
    path.colors = np.array([color])
    return path

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


def move_piece(piece, translation):
    piece['mesh'].apply_translation(translation)
    piece['corners'] = [(cid, pos+translation) for cid,pos in piece['corners']]
    return piece

def get_top_k_scored_moves(pieces, assembly, target_offsets, k=float("inf"), mates_list=None, talk=True):
    start_time = time.time()

    # First quickly make a broad list of semi-verified moves, not checking for intermediate collision
    unverified = []
    unverified_counts = []
    for i in range(6):
        new_unverified = get_feasible_motions(pieces[i], assembly, mates_list, steps=1, check_collision=False)
        unverified = unverified + new_unverified
        unverified_counts.append(len(new_unverified))
    unverified.append(((0,0),(0,0),np.array([0,0,0]))) # zero-action move

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
        test_piece = this_piece['mesh'].copy()
        assembly_meshes = [piece["mesh"] for piece in assembly if piece["id"] != pid1]
        if check_path_clear(test_piece, assembly_meshes, vec, 20):
            feasible_scored_moves.append(scored_move)
            if len(feasible_scored_moves) == k:
                break
    k = min(len(feasible_scored_moves), k)
    if talk:
        print(f"| Acquired Best {k} moves. This took {time.time() - start_time:.2f} seconds.")

    return feasible_scored_moves

def get_moves_scored_lookahead(pieces, assembly, target_offsets, mates_list=None, top_k=2, lookahead = 2, rollout_depth=2):
    print("Getting primary moves...")
    top_moves = get_top_k_scored_moves(pieces, assembly, target_offsets, k=top_k, mates_list=mates_list)
    print(f"| Looking Ahead from top {top_k} moves...")
    
    assembly_list = [piece['id'] for piece in assembly]
    scored_moves = []
    for movei, move in enumerate(top_moves):
        primary_cost, ((pid1,cid1),(pid2,cid2), vec) = move
        # Test move with virtual objects
        temp_piece = move_piece(copy.deepcopy(pieces[pid1]), vec)
        temp_pieces = copy.deepcopy(pieces)
        temp_pieces[pid1] = temp_piece
        if not temp_piece['id'] in assembly_list:
            temp_assembly = assembly + [temp_piece]
        else:
            temp_assembly = assembly

        # Find best *secondary* move from new state
        print(f"| | Rolling out from primary move {movei + 1}/{top_k}...")
        start_time = time.time()

        future_cost = greedy_rollout_score(temp_pieces, temp_assembly, target_offsets, mates_list, depth=rollout_depth)
        total_score = primary_cost + future_cost
        print(f"| | → Rollout total score: {total_score:.4f}. This all took {time.time() - start_time:.2f} seconds.")
        scored_moves.append((total_score, ((pid1,cid1),(pid2,cid2), vec)))

    scored_moves.sort(key=lambda x: x[0])
    return scored_moves

def greedy_rollout_score(pieces, assembly, target_offsets, mates_list, depth=2):
    if depth == 0:
        return 0

    start_time = time.time()
    best_cost = float("inf")
    best_move = None

    best_move = get_top_k_scored_moves(pieces, assembly, target_offsets, k=1, mates_list=mates_list, talk=False)
    best_cost, ((best_pid, _), (_, _), best_vec) = best_move[0]
    x, y, z = best_vec
    print(f"| | | Greedy: Best move is p{best_pid} moving by <{x: .0f},{y: .0f},{z: .0f}>. Depth to go: {depth-1}. This took {time.time() - start_time:.2f}s.")
    if best_move is None or best_cost == 0:
        return 0  # No valid move or best greedy move is zero.

    # Execute best move
    temp_piece = move_piece(copy.deepcopy(pieces[best_pid]), best_vec)
    temp_pieces = copy.deepcopy(pieces)
    temp_pieces[best_pid] = temp_piece
    new_assembly = assembly + [temp_piece]

    return best_cost + greedy_rollout_score(temp_pieces, new_assembly, target_offsets, mates_list, depth=depth-1)



def render_scene(all_pieces, arrows=None, remake_pieces=True, camera = [14.0, -16.0, 20.0, 0.0]):
    scene = trimesh.Scene()
    if remake_pieces:
        for piece in all_pieces:
            pid = piece['id']
            scene.add_geometry(piece['mesh'], node_name=f"piece_{pid}")
            if pid == FLOOR_INDEX_OFFSET or pid < REFERENCE_INDEX_OFFSET:
                show_corners(scene, piece)
    if arrows:
        # Remove old arrows
        arrow_nodes = [name for name in scene.graph.nodes_geometry if name.startswith('arrow_')]
        for name in arrow_nodes:
            scene.delete_geometry(name)

        for i, arrow in enumerate(arrows):
            scene.add_geometry(arrow, node_name=f"arrow_{i}")
    scene.camera_transform = get_transform_matrix(camera)
    return scene

def show_moves_scored(scored_moves, pieces, floor):
    # Show moves, make the first one green, and make the next 19 black
    arrows = []
    for i, (score, move) in enumerate(scored_moves):
        if i == 0:
            color = [0, 255, 0, 255]
        elif i <= 19:
            color = [0, 0, 0, 255]
        else:
            color = [0, 0, 0, 50]
        arrows.append(create_arrow(move, pieces, floor, color=color))
    return arrows

def show_corners(scene, piece):
    for cid, pos in piece['corners']:
        pid = piece['id']
        sphere = trimesh.creation.uv_sphere(radius=0.1)
        sphere.apply_translation(pos)
        sphere.visual.face_colors = [255, 255, 255, 255]
        scene.add_geometry(sphere, node_name=f'corner_{pid}_{cid}')

def get_transform_matrix(position):
    x, y, z, roll = position
    # Set Camera as x, y, z, roll staring at origin
    forward = np.array([x,y,z])
    forward /= np.linalg.norm(forward)
    right = np.cross(np.array([0, 0, 1]), forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    roll_right = right * np.cos(roll) + up * np.sin(roll)
    roll_up = -right * np.sin(roll) + up * np.cos(roll)
    transform = np.eye(4)
    transform[:3, 0] = roll_right    # Right vector
    transform[:3, 1] = roll_up       # Up vector
    transform[:3, 2] = forward       # Forward vector
    transform[:3, 3] = [x, y, z]     # Position
    return transform

def save_animation_frame(scene, index, out_dir="frames"):
    os.makedirs(out_dir, exist_ok=True)

    # Save as image
    image_path = os.path.join(out_dir, f"frame_{index:03d}.png")
    png = scene.save_image(resolution=(800, 600), visible=True)
    with open(image_path, 'wb') as f:
        f.write(png)

    print(f"Saved frame {index} to {image_path}")

def test_script(n_stages=12,top_k=8,lookahead=2,rollout_depth=8):
    # Create Reference Pieces
    reference_pieces = define_all_burr_pieces(reference=True)
    target_offsets = np.array([[0,0,1],[-1,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,-1]])
    target_offsets += [0,0,3]
    for piece in reference_pieces:
        piece = move_piece(piece, target_offsets[piece['id']-REFERENCE_INDEX_OFFSET])
    
    # Create Floor
    floor = create_floor()
    
    # Create Real Pieces
    pieces = define_all_burr_pieces()
    start_offsets = np.array([[4,-8,1],[-8,0,1],[8,0,1],[-4,8,3],[-4,-8,3],[4,8,1]])
    for piece in pieces:
        piece = move_piece(piece, start_offsets[piece['id']])
    
    cost = cost_function(pieces, target_offsets)
    print(f"Initial Cost Measure: {cost:.4f}")

    # Create initial scene
    all_pieces = reference_pieces + pieces + [floor]
    scene = render_scene(all_pieces)
    save_animation_frame(scene, 0)

    # Precompute Mates (p1,c1), (p2,c2)
    mates_list = get_valid_mates(pieces, floor)
    # mates_list = None

    # Begin assembly set with the floor. The floor is a static base piece. 
    assembly = [floor]
    assembly_list = [FLOOR_INDEX_OFFSET]
    for stage in range(n_stages):
        scored_moves = get_moves_scored_lookahead(pieces, assembly, target_offsets, mates_list, top_k=top_k, lookahead=lookahead, rollout_depth=rollout_depth)
        best_move = scored_moves[0]
        _, ((best_pid, _), (other_pid, _), best_vec) = best_move
        x, y, z = best_vec
        print(f"| → Best move: Piece {best_pid} → {other_pid}, vec = <{x: .0f},{y: .0f},{z: .0f}>.")
        # Now execute, and add the moved piece to the assembly
        moved_piece = pieces[best_pid]
        moved_piece = move_piece(moved_piece, best_vec)
        # Add part to assembly, if not there already
        if not best_pid in assembly_list:
            assembly.append(moved_piece)
            assembly_list.append(best_pid)
        print(f"→ Done with Stage {stage+1} / {n_stages}. New Cost: {cost_function(pieces, target_offsets):.4f}")

        arrows = show_moves_scored(scored_moves, pieces, floor)
        all_pieces = reference_pieces + pieces + [floor]
        scene = render_scene(all_pieces, arrows)
        save_animation_frame(scene, stage+1)
    scene.show()
start_time = time.time()
test_script()
print(f"This script took {time.time() - start_time:4.0f} seconds")

