import trimesh
import numpy as np
import time
import copy
import imageio
import pickle
import os
import colorsys
import concurrent.futures
from shapely.geometry import Point, Polygon

# Nomenclature
# Scored Move = (Scores, Move) 
#             = (Scores, (Mate[0], Mate[1], Translation)) 
#             = ([Score 0, Score 1, Rollout Score]], ((pid1, cid1), (pid2, cid2), Translation))
# Pieces = [Piece, ..., Piece] 
#        = [{Mesh, Corners, ID}, ..., Piece]

np.set_printoptions(formatter={'int': '{:2d}'.format})
REFERENCE_INDEX_OFFSET = 50 # Bump up reference piece indices
FLOOR_INDEX_OFFSET = 100 # Bump up floor piece index
NULL_MOVE = ((0,0),(0,0),np.array([0,0,0])) # Definition of zero action
NUM_PIECES = 6
UNINTERESTING_CORNERS = { # Uses Corner IDs according to find_cube_corners
    1: {1, 5, 10, 12, 15, 21, 23, 29},
    2: {1, 13, 14, 19, 21, 22, 28},
    3: {6, 12, 20, 26},
    4: {1, 2, 3, 4, 12, 19, 23},
    5: {1, 6, 12, 17},
    # piece 0: none
}
DOWN = np.array([0,0,-1])

# Creation Functions
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
    bad_cids = UNINTERESTING_CORNERS.get(idx, set())
    filtered_corners = [(cid, pos) for cid, pos in corners if cid not in bad_cids]
    if reference:
        idx += REFERENCE_INDEX_OFFSET
    return {'mesh': composite, 'corners': filtered_corners, 'id': idx}

def define_all_burr_pieces(start_offsets=None, reference=False):
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

    if start_offsets is not None:
        for piece in pieces:
            piece = move_piece(piece, start_offsets[piece['id']])
    return pieces


# Geometry Functions
def check_path_clear(this_mesh, other_meshes, translation, steps=20, tol=0.01):
    """
    Check whether this_mesh can be translated without colliding with other meshes(s)
    at any step along the way. Handles both single-piece and multi-piece cases.

    Args:
        this_mesh: trimesh.Trimesh - The moving piece
        other_meshes: EITHER trimesh.Trimesh OR List[trimesh.Trimesh] - the other pieces
        translation: np.ndarray, shape (3,) - Translation vector
        steps: int - Number of interpolation steps to test
    Returns:
        bool - True if path is clear
    """
    # Convert single piece to list for uniform handling
    if not isinstance(other_meshes, (list, tuple)):
        other_meshes = [other_meshes]
    
    this_bbox = this_mesh.bounds
    for step in reversed(range(1, steps + 1)): # Start from end of trajectory, where collisions are most likely
        frac_translation = (translation * step) / (steps)
        test_bbox = this_bbox + frac_translation
        test_mesh = this_mesh.copy()
        test_mesh.apply_translation(frac_translation)

        for other_mesh in other_meshes:
            other_bbox = other_mesh.bounds
            if not (all(test_bbox[0] < other_bbox[1]) and # Check for lower test < upper other
                    all(test_bbox[1] > other_bbox[0])):   # Check for upper test > lower other
                continue # If the boxes don't touch, don't bother

            # Otherwise, do full volumetric collision detection.
            if trimesh.boolean.intersection([test_mesh, other_mesh], check_volume=False).volume > tol:
                return False

    return True  # No collision detected along the entire path

def get_feasible_motions(this_piece, pieces, valid_mates=None, steps=20, check_collision=True, is_being_held=False):
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
    other_meshes = [piece['mesh'] for piece in pieces if piece['id'] != this_pid]
    other_corners = []
    for piece in pieces:
        if piece['id'] == this_piece['id']:
            continue
        for cid, pos in piece['corners']:
            other_corners.append((piece['id'], cid, pos))

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
            test_mesh = this_piece['mesh'].copy()
            if not check_path_clear(test_mesh, other_meshes, test_vec, 1):
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
        test_mesh = this_piece['mesh'].copy()
        if not check_collision or check_path_clear(test_mesh, other_meshes, vec, steps):
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

def is_supported(this_mesh, other_meshes):
    offset = 0.01 * DOWN
    translated = this_mesh.copy()
    translated.apply_translation(offset)

    intersects = []
    for other_mesh in other_meshes:
        intersect = trimesh.boolean.intersection([translated, other_mesh], check_volume=False)
        if not intersect.is_empty and intersect.volume > 0.001:
            intersect = intersect.convex_hull
            intersects.append(intersect)
    if len(intersects) > 0:
        try:
            intersection = trimesh.boolean.union(intersects)
        except:
            scene = trimesh.Scene()
            scene.camera_transform = get_transform_matrix([14.0, -16.0, 20.0, 0.0])
            scene.add_geometry(intersects[0], node_name=f"intersect")
            scene.show()
        intersection_cvhull = intersection.convex_hull
    else:
        return False # No intersects means no support

    CoM_xy = this_mesh.center_mass[0:2] # only interested in x/y coordinate, if Z is down
    hull_points_xy = intersection_cvhull.vertices[:, :2]
    polygon = Polygon(hull_points_xy).convex_hull
    # if polygon.contains(Point(CoM_xy)):
    #     scene = trimesh.Scene()
    #     scene.camera_transform = get_transform_matrix([14.0, -16.0, 20.0, 0.0])
    #     scene.add_geometry(intersection_cvhull, node_name=f"intersect")
    #     scene.show()
    return polygon.contains(Point(CoM_xy))


# Dynamic Programming Functions
def move_piece(piece, translation):
    piece['mesh'].apply_translation(translation)
    piece['corners'] = [(cid, pos+translation) for cid,pos in piece['corners']]
    return piece

def get_top_k_scored_moves(pieces, active_pids, target_offsets, k=float("inf"), mates_list=None):
    # First quickly make a broad list of semi-verified moves, not checking for intermediate collision
    unverified = []
    unverified_counts = []
    active_pieces = [piece for piece in pieces if piece['id'] in active_pids]
    for i in range(NUM_PIECES):
        new_unverified = get_feasible_motions(pieces[i], active_pieces, mates_list, steps=1, check_collision=False)
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


# Rendering Scripts
def render_scene(all_pieces, arrows=None, remake_pieces=True, camera = [14.0, -16.0, 20.0, 0.0]):
    scene = trimesh.Scene()
    scene.camera_transform = get_transform_matrix(camera)
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
    return scene

def show_moves_scored(scored_moves, pieces, floor, opacity=0.6):
    num_moves = len(scored_moves)
    # hue_range = 0.333 * np.flip(np.arange(num_moves)) / num_moves
    arrows = []
    for i, (score, move) in enumerate(scored_moves):
        # color = colorsys.hsv_to_rgb(hue_range[i], 1, 1)
        color = [0, 0, 0, 1-float(i)/num_moves]
        # color =  + [opacity]
        # color = np.floor(256*color) + [opacity]
        arrows.append(create_arrow(move, pieces, floor, color=color))
    return arrows

def create_arrow(move, pieces, floor, color=[0, 0, 0, 255]):
    """
    Create a line-shaped arrow from start to end using a path.
    
    Returns a trimesh.path.Path3D object.
    """
    (pid1, cid1), (pid2, cid2), vec = move
    start_corners_dict = dict(pieces[pid1]['corners'])
    start = start_corners_dict[cid1]
    if pid2 == FLOOR_INDEX_OFFSET:
        end = floor['corners'][cid2][1]
    else:
        end_corners_dict = dict(pieces[pid2]['corners'])
        end = end_corners_dict[cid2]
    path = trimesh.load_path(np.array([[start, end]]))
    path.colors = np.array([color])
    return path

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

def save_animation_frame(scene, index, out_dir="frames", suffix=''):
    os.makedirs(out_dir, exist_ok=True)

    # Save as image
    image_path = os.path.join(out_dir, f"frame_{index:03d}{suffix}.png")
    png = scene.save_image(resolution=(800, 600), visible=True)
    with open(image_path, 'wb') as f:
        f.write(png)

    print(f"Saved frame {index} to {image_path}")

def load_mates_list(filename='cache/mates_list.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            print("Loaded mates_list from file.")
            return pickle.load(f)
    else:
        print("No cached mates_list found.")
        return None

def save_mates_list(mates_list, filename='cache/mates_list.pkl'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(mates_list, f)
        print(f"Saved mates_list to {filename}")

def save_simulation_state(step_id, pieces, active_pids, all_scored_moves, metadata=None, folder='logs', suffix=''):
    os.makedirs(folder, exist_ok=True)
    data = {
        'pieces': pieces,
        'assembly': active_pids,
        'available_moves': all_scored_moves,
        'metadata': metadata or {}
    }
    with open(f'{folder}/step_{step_id:03d}{suffix}.pkl', 'wb') as f:
        pickle.dump(data, f)

def load_simulation_state(step_id, folder='logs'):
    with open(f'{folder}/step_{step_id:03d}.pkl', 'rb') as f:
        return pickle.load(f)


# Top-Level Scripts
def run_assembler(n_stages=16,top_k=[float("inf"), float("inf")], rollout_depth=3, render=False, show_all_scores=True, start_from=None):
    target_offsets = np.array([[0,0,1],[-1,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,-1]])
    target_offsets += [0,0,3]

    # Begin assembly set (active pids) with the floor. The floor is a static base piece. 
    if start_from is None:
        # Create Reference Pieces
        reference_pieces = define_all_burr_pieces(reference=True)
        for piece in reference_pieces:
            piece = move_piece(piece, target_offsets[piece['id']-REFERENCE_INDEX_OFFSET])
        
        # Create Floor and Real Pieces
        floor = create_floor()
        start_offsets = np.array([[4,-8,1],[-8,0,1],[8,0,1],[-4,8,3],[-4,-8,3],[4,8,1]])
        pieces = define_all_burr_pieces(start_offsets)
        
        cost = cost_function(pieces, target_offsets)
        print(f"Initial Cost Measure: {cost:.2f}")

        # Create initial scene
        pieces_augmented = pieces + [floor]
        all_pieces = reference_pieces + pieces_augmented
        if render:
            scene = render_scene(all_pieces)
            save_animation_frame(scene, 0)
        else:
            scene = None

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
        if render:
            scene = render_scene(pieces_augmented)
            save_animation_frame(scene, start_from+1)
        active_pids = active_pids + [best_pid] if best_pid not in active_pids else active_pids
        print(f"→ Fast-Finished {stage+1} / {n_stages}.")
        print(f"Score [Lookahead 1, Lookahead 2, Rollout]: [{lookahead_1:.2f}, {lookahead_2:.2f}, {rollout:.2f}] long-term.")

        if cost_function(pieces, target_offsets) < 0.1:
            return scene # If we're at zero cost, we're done.
        
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
        if render:
            arrows = show_moves_scored(scored_moves, pieces, floor)
            scene = render_scene(pieces_augmented, arrows=arrows)
            save_animation_frame(scene, stage, suffix='a')
        save_simulation_state(stage, pieces_augmented, active_pids, scored_moves)
        
        # Now execute
        moved_piece = pieces[best_pid]
        moved_piece = move_piece(moved_piece, best_vec)
        if render:
            scene = render_scene(pieces_augmented)
            save_animation_frame(scene, stage+1)
        active_pids = active_pids + [best_pid] if best_pid not in active_pids else active_pids
        print(f"→ Done with Stage {stage+1} / {n_stages}. This took {time.time() - start_time:.2f} seconds.")
        print(f"Score [Lookahead 1, Lookahead 2, Rollout]: [{lookahead_1:.2f}, {lookahead_2:.2f}, {rollout:.2f}] long-term.")

        if cost_function(pieces, target_offsets) < 0.1:
            return scene # If we're at zero cost, we're done.
    return scene

# Example script running through mates and checking which ones need support. For verifying support finder only.
def test_support(): 
    # Create Floor and Real Pieces
    floor = create_floor()
    start_offsets = np.array([[4,-8,1],[-8,0,1],[8,0,1],[-4,8,3],[-4,-8,3],[4,8,1]])
    pieces = define_all_burr_pieces(start_offsets)
    
    # Create initial scene
    all_pieces = pieces + [floor]

    # Precompute Mates (p1,c1), (p2,c2)
    mates_list = load_mates_list()
    if mates_list is None:
        mates_list = get_valid_mates(pieces, floor)
        save_mates_list(mates_list)

    supports = []
    for this_piece in pieces:
        this_pid = this_piece['id']
        this_corners = this_piece['corners']

        other_meshes = [piece['mesh'] for piece in all_pieces if piece['id'] != this_pid]
        other_meshes_not_floor = [piece['mesh'] for piece in pieces if piece['id'] != this_pid]
        other_corners = []
        for piece in all_pieces:
            if piece['id'] == this_piece['id']:
                continue
            for cid, pos in piece['corners']:
                other_corners.append((piece['id'], cid, pos))

        this_mates_list = mates_list[this_pid] # Mates pertaining to this piece
        # Match corners to actual positions
        this_corner_dict = {cid: pos for cid, pos in this_corners}
        other_corner_dict = {(pid, cid): pos for (pid,cid,pos) in other_corners}
        available_pieces = {p['id'] for p in pieces}
        combinations = [
            ((this_pid, cid1), (pid2, cid2), other_corner_dict[(pid2, cid2)] - this_corner_dict[cid1])
            for ((pid1, cid1), (pid2, cid2)) in this_mates_list
            if pid2 in available_pieces
        ]

        for (p1, c1), (p2, c2), vec in combinations:
            # Check Support
            test_piece = copy.deepcopy(this_piece)
            test_piece = move_piece(test_piece, vec)
            test_mesh = test_piece['mesh']
            test_bbox = test_mesh.bounds
            collide = False
            for other_mesh in other_meshes:
                other_bbox = other_mesh.bounds
                if (all(test_bbox[0] < other_bbox[1]) and # Check for lower test < upper other
                    all(test_bbox[1] > other_bbox[0])):   # Check for upper test > lower other
                    collide = True
            if collide:
                supports.append(False)
                continue
            if test_bbox[0][2] == 0:
                supports.append(True) # Equivalent to the block being on the floor. Thus, we don't include the floor in support checks.
                continue
            support = is_supported(test_mesh, other_meshes_not_floor)
            supports.append(support)
            # if support and vec[2] > 0: # Show cases where a piece lays on top of another.
            #     scene = render_scene(all_pieces + [test_piece])
            #     scene.show()
    print(f"There are {supports.count(True)} configurations with support and {supports.count(False)} without.")
    return

# Place two pieces and draw arrows between all feasible mates
def display_connections():
    target_offsets = np.array([[0,0,1],[-1,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,-1]])
    target_offsets += [0,0,3]

    # Create Reference Pieces
    reference_pieces = define_all_burr_pieces(reference=True)
    for piece in reference_pieces:
        piece = move_piece(piece, target_offsets[piece['id']-REFERENCE_INDEX_OFFSET])
    
    # Create Floor and Real Pieces
    floor = create_floor()
    start_offsets = np.array([[4,-8,1],[-8,0,1],[8,0,1],[-4,8,3],[-4,-8,3],[4,8,1]])
    pieces = define_all_burr_pieces(start_offsets)
    
    # Move Piece 3 to its correct location
    # pieces[3] = move_piece(pieces[3], target_offsets[3] - start_offsets[3])
    # pieces[5] = move_piece(pieces[5], [0, -16, 0])

    # Create initial scene
    pieces_augmented = pieces + [floor]
    all_pieces = reference_pieces + pieces_augmented
    scene = render_scene(all_pieces)

    # Precompute Mates (p1,c1), (p2,c2)
    mates_list = load_mates_list()
    if mates_list is None:
        mates_list = get_valid_mates(pieces, floor)
        save_mates_list(mates_list)

    # Do a mock assembly looking through all immediately available moves 
    active_pids = [FLOOR_INDEX_OFFSET]
    scored_moves = get_top_k_scored_moves(pieces_augmented, active_pids, target_offsets, mates_list=mates_list)
    # print(f"| Processed {len(scored_moves)} moves.")

    # Filter to only look at piece 5
    # scored_moves = [scored_move for scored_move in scored_moves if scored_move[1][0][0] == 5]
    # print(f"| Filtered to {len(scored_moves)} moves.")

    arrows = show_moves_scored(scored_moves, pieces, floor)
    # Filter arrows list to be every 4th arrow
    # arrows = [arrow for i, arrow in enumerate(arrows) if np.mod(i, 4) == 0]
    # print(f"| Filtered to {len(arrows)} moves.")

    # scene_pieces = [reference_pieces[3], reference_pieces[5], pieces[3], pieces[5], floor]
    scene = render_scene(pieces_augmented, arrows=arrows)
    return scene

if __name__=="__main__":
    start_time = time.time()
    # scene = display_connections()
    scene = run_assembler(n_stages=30, top_k=[float("inf"), 1], rollout_depth=100, render=True)
    # for sc in range(17):
    #     state = load_simulation_state(sc)
    #     scene = render_scene(state['pieces'])
    #     save_animation_frame(scene, sc+1)
    print(f"This script took {time.time() - start_time:4f} seconds")
    # scene.show()
