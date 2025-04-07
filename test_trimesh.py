import trimesh
import numpy as np
import time
import copy

REFERENCE_INDEX_OFFSET = 50 # Bump up reference piece indices

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
    return {'mesh': floor, 'corners': corners, 'id': 100}

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
        box.visual.face_colors = color
        meshes.append(box)

    composite = trimesh.util.concatenate(meshes)
    composite.process(validate=True)
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

    
    sample_points = this_piece.vertices - 0.05 * this_piece.vertex_normals
    center_points = []
    for center, normal in zip(this_piece.triangles_center, this_piece.face_normals):
        point = center - normal * 0.05
        center_points.append(point)
    sample_points = np.vstack((sample_points, np.array(center_points)))



    this_bbox = this_piece.bounds
    for step in reversed(range(1, steps + 1)):
        frac_translation = (translation * step) / (steps)
        test_bbox = this_bbox + frac_translation
        test_points = sample_points + frac_translation

        for other_piece in other_pieces:
            other_bbox = other_piece.bounds
            if not (all(test_bbox[0] < other_bbox[1]) and # Check for lower test < upper other
                    all(test_bbox[1] > other_bbox[0])):   # Check for upper test > lower other
                continue # If the boxes don't touch, don't bother

            if any(other_piece.nearest.signed_distance(test_points) > tol):
                return False

    return True  # No collision detected along the entire path

def get_feasible_motions(this_piece, pieces, valid_mates=None, steps=20):
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
        key = tuple(np.round(vec, 4))  # avoid float fuzz
        if key in unique_motions:
            continue # Already seen this transformation
        unique_motions[key] = True # Save to seen motions if not

        # Check Path of Motion
        test_piece = this_piece['mesh'].copy()
        if check_path_clear(test_piece, other_meshes, vec, steps):
            feasible_motions.append(((p1, c1), (p2, c2), vec))

    return feasible_motions

def get_valid_mates(pieces, floor):
    """
    Precompute and store valid mating corner pairs for each piece-to-piece combo.
    Returns a dict of lists of mates (p1,c1),(p2,c2)
    """
    print("Precomputing mates...")
    start_time = time.time()
    # Augment piece list with floor
    all_pieces = copy.deepcopy(pieces)
    all_pieces.append(copy.deepcopy(floor))
    cache = {}
    cache_length = 0
    for p1 in pieces:
        pid = p1['id']
        valid_motions = get_feasible_motions(p1, all_pieces, steps=1)
        # cache = cache + valid_motions
        valid_mates = [motion[:2] for motion in valid_motions]
        cache[pid] = valid_mates
        end_time = time.time()
        cache_length += len(valid_mates)
        print(f"Got {len(valid_mates)} mates for piece {pid}. t = {(end_time-start_time):4f}s")
    print(f"For this assembly, there are {cache_length} valid piece-to-piece mates")
    return cache


def create_arrow(move, pieces, floor, color=[0, 0, 0, 255]):
    """
    Create a line-shaped arrow from start to end using a path.
    
    Returns a trimesh.path.Path3D object.
    """
    (pid1, cid1), (pid2, cid2), vec = move
    start = pieces[pid1]['corners'][cid1][1]
    if pid2 == 100:
        end = floor['corners'][cid2][1]
    else:
        end = pieces[pid2]['corners'][cid2][1]
    path = trimesh.load_path(np.array([
        [start, end]
    ]))
    path.colors = np.array([color])
    return path

def show_corners(scene, piece):
    for cid, pos in piece['corners']:
        pid = piece['id']
        sphere = trimesh.creation.uv_sphere(radius=0.1)
        sphere.apply_translation(pos)
        sphere.visual.face_colors = [255, 255, 255, 255]
        scene.add_geometry(sphere, node_name=f'corner_{pid}_{cid}')


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

def get_moves_scored(pieces, assembly, mates_list, target_offsets):
    start_time = time.time()
    feasible = []
    feasible_counts = []
    for i in range(6):
        new_feasible = get_feasible_motions(pieces[i], assembly, mates_list)
        feasible = feasible + new_feasible
        feasible_counts.append(len(new_feasible))
    print(f"| Pieces 0-5 have {feasible_counts} available moves")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"| Time taken to get feasible moves: {elapsed_time:.4f} seconds")

    feasible.append(((0,0),(0,0),([0,0,0]))) # zero-action move

    scored_moves = []
    for (pid1, cid1), (pid2, cid2), vec in feasible:
        cost_change = get_cost_change(pieces[pid1], vec, target_offsets[pid1])
        scored_moves.append((cost_change, ((pid1, cid1), (pid2, cid2), vec)))
    scored_moves.sort(key=lambda x: x[0])
    return scored_moves

def get_moves_scored_lookahead(pieces, assembly, mates_list, target_offsets, top_k = 2):
    start_time = time.time()
    print("| Getting primary moves...")
    all_scored_moves = get_moves_scored(pieces, assembly, mates_list, target_offsets)
    top_moves = all_scored_moves[:top_k]
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
            temp_assembly_list = [piece['id'] for piece in temp_assembly]
        else:
            temp_assembly = assembly
            temp_assembly_list = assembly_list
        # print(f"| | Virtual Assembly consists of {temp_assembly_list}")

        # Find best *secondary* move from new state
        secondary_best = float("inf")
        secondary_counts = []
        for j in range(6):
            secondary_moves = get_feasible_motions(temp_pieces[j], temp_assembly, mates_list)
            secondary_counts.append(len(secondary_moves))
            for smove in secondary_moves:
                (_, _), (_, _), svec = smove
                cost = get_cost_change(temp_pieces[j], svec, target_offsets[j])
                if cost < secondary_best:
                    secondary_best = cost
        print(f"| | Pieces 0-5 have {secondary_counts} available moves.")

        total_score = primary_cost + secondary_best
        scored_moves.append((total_score, ((pid1,cid1),(pid2,cid2), vec)))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"| | Done with Move {movei+1} / {top_k}. Time: {elapsed_time:.4f}")

    scored_moves.sort(key=lambda x: x[0])
    return scored_moves

def show_moves_scored(scene, scored_moves, pieces, floor):
    # Remove old arrows
    arrow_nodes = [name for name in scene.graph.nodes_geometry if name.startswith('arrow_')]
    for name in arrow_nodes:
        scene.delete_geometry(name)

    # Show moves, make the first one green, and make the next 19 black
    for i, (score, move) in enumerate(scored_moves):
        if i == 0:
            color = [0, 255, 0, 255]
        elif i <= 19:
            color = [0, 0, 0, 255]
        else:
            color = [0, 0, 0, 50]
        arrow = create_arrow(move, pieces, floor, color=color)
        scene.add_geometry(arrow, node_name=f'arrow_{i}')

def test_script():
    scene = trimesh.Scene()
    # Create Reference Pieces
    reference_pieces = define_all_burr_pieces(reference=True)
    target_offsets = np.array([[0,0,1],[-1,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,-1]])
    target_offsets += [0,0,3]
    
    for piece in reference_pieces:
        _ = piece['mesh'].nearest
        pid = piece['id']
        piece = move_piece(piece, target_offsets[pid-REFERENCE_INDEX_OFFSET])
        scene.add_geometry(piece['mesh'], node_name=f'piece_reference_{pid}')
    
    # Create Floor
    floor = create_floor()
    scene.add_geometry(floor['mesh'], node_name='floor')
    show_corners(scene, floor)
    
    
    # Create Real Pieces
    pieces = define_all_burr_pieces()
    start_offsets = np.array([[4,-8,1],[-8,0,1],[8,0,1],[-4,8,3],[-4,-8,3],[4,8,1]])

    for piece in pieces:
        pid = piece['id']
        piece = move_piece(piece, start_offsets[pid])
        scene.add_geometry(piece['mesh'], node_name=f'piece_real_{pid}')
        show_corners(scene, piece)

    cost = cost_function(pieces, target_offsets)
    print(f"Initial Cost Measure: {cost:.4f}")

    # Precompute Mates (p1,c1), (p2,c2)
    mates_list = get_valid_mates(pieces, floor)

    # Begin assembly set with the floor. The floor is a static base piece. 
    assembly = [floor]
    assembly_list = [100]
    n_stages = 8
    for stage in range(n_stages):
        scored_moves = get_moves_scored_lookahead(pieces, assembly, mates_list, target_offsets)
        print("Best score change:", scored_moves[0][0])
        show_moves_scored(scene, scored_moves, pieces, floor)
        # Now execute, and add the moved piece to the assembly
        moved_piece = pieces[scored_moves[0][1][0][0]]
        translation = scored_moves[0][1][2]
        moved_piece = move_piece(moved_piece, translation)
        # Add part to assembly, if not there already
        if not moved_piece['id'] in assembly_list:
            assembly.append(moved_piece)
            assembly_list.append(moved_piece['id'])
        show_corners(scene, moved_piece)
        print(f"Done with Stage {stage+1} / {n_stages}")
        # print(f"Assembly consists of {assembly_list}")
    scene.show()

test_script()


