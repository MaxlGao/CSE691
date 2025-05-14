import trimesh
import numpy as np
import time
import copy
from shapely.geometry import Point, Polygon
from helper_burr_piece_creator import create_gripper, FLOOR_INDEX_OFFSET, GRIPPER_CONFIGS

DOWN = np.array([0,0,-1])

# Geometry Functions
def check_path_clear(this_mesh, other_meshes, translation, steps=20, tol=0.01, floor_active=True):
    """
    Check whether this_mesh can be translated without colliding with other meshes(s)
    at any step along the way. Handles both single-piece and multi-piece cases.

    Args:
        this_mesh: trimesh.Trimesh - The moving piece
        other_meshes: EITHER trimesh.Trimesh OR List[trimesh.Trimesh] - the other pieces
        translation: np.ndarray, shape (3,) - Translation vector
        steps: int - Number of interpolation steps to test
        floor_active: bool - whether or not the floor has collision
    Returns:
        bool - True if path is clear
    """
    # Convert single piece to list for uniform handling
    if not isinstance(other_meshes, (list, tuple)):
        other_meshes = [other_meshes]
    if not floor_active:
        other_meshes = other_meshes[:-1] # exclude last piece, which is floor
        
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

def get_feasible_motions(this_piece, pieces, mates_list=None, steps=20, check_path=True):
    """
    Compute feasible, unique linear motions from `this_piece` to any other in `pieces`.

    Parameters:
        - this_piece: dict with 'mesh', 'corners', 'id'
        - pieces: list of all pieces
        - mates_list: optional list of ((pid1, cid1), (pid2, cid2)) to restrict testing
        - steps: number of interpolation steps for collision checks (1 means skip motion path check)
        - tolerance: rounding tolerance for deduplication
    Returns:
        List of ((this_pid, this_cid), (other_pid, other_cid), [dx, dy, dz])
    """
    this_pid = this_piece['id']
    this_mesh = this_piece['mesh']
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
        
    if mates_list is not None:
        mates_list = [mate for mate in mates_list if mate[0][0] == this_pid] # Mates pertaining to this piece
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
            test_mesh = this_mesh.copy()
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
        test_mesh = this_mesh.copy()
        if not check_path or check_path_clear(test_mesh, other_meshes, vec, steps):
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
    mates_list = []
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
        mates_list += valid_mates
        print(f"| Got {len(valid_mates):4d} mates for piece {pid}. This took {(time.time()-start_time):.2f} seconds.")
    print(f"→ For this assembly, there are {len(mates_list)} valid piece-to-piece mates")
    return mates_list

def get_unsupported_pids(assembly_pieces, basic_method=True):
    """
    Recursively determine which pieces are unsupported based on actual supported ancestry.
    A piece is only supported if it's on the floor or supported by another supported piece.
    """
    supported_ids = set()
    remaining_ids = {p['id'] for p in assembly_pieces}
    id_to_piece = {p['id']: p for p in assembly_pieces}

    # Step 1: Add pieces directly supported by the floor (and the floor is supported)
    for piece in assembly_pieces:
        if piece['mesh'].bounds[0][2] == 0 or piece['id'] == FLOOR_INDEX_OFFSET:
            supported_ids.add(piece['id'])

    unconfirmed_ids = remaining_ids - supported_ids
    if basic_method:
        for pid in unconfirmed_ids:
            piece = id_to_piece[pid]
            other_meshes = [p['mesh'] for p in assembly_pieces if p['id'] != pid]
            if is_supported(piece['mesh'], other_meshes):
                piece['gripper_config']['active'] = False # Deactivate gripper for now
                supported_ids.add(pid)
    else:
        # Step 2: Propagate support
        changed = True
        while changed:
            changed = False
            for pid in remaining_ids - supported_ids:
                piece = id_to_piece[pid]
                potential_supporters = [id_to_piece[sid] for sid in supported_ids if sid != pid]
                support_meshes = [p['mesh'] for p in potential_supporters]
                if is_supported(piece['mesh'], support_meshes):
                    piece['gripper_config']['active'] = False # Deactivate gripper for now
                    supported_ids.add(pid)
                    changed = True

    # Step 3: Unsupported pieces are those not marked as supported
    return list(remaining_ids - supported_ids)

def is_supported(this_mesh, other_meshes):
    # Quick check to see if this mesh is on the floor
    this_bbox = this_mesh.bounds
    if this_bbox[0][2] == 0: # Lower bound Z is zero
        # Equivalent to the block being on the floor. Thus, we don't include the floor in support checks.
        return True

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
        except: # debug
            scene = trimesh.Scene()
            scene.add_geometry(intersects[0], node_name=f"intersect")
            scene.show()
        intersection_cvhull = intersection.convex_hull
    else:
        return False # No intersects means no support

    CoM_xy = this_mesh.center_mass[0:2] # only interested in x/y coordinate, if Z is down
    hull_points_xy = intersection_cvhull.vertices[:, :2]
    polygon = Polygon(hull_points_xy).convex_hull
    # if polygon.contains(Point(CoM_xy)): # Debug verification
    #     scene = trimesh.Scene()
    #     scene.add_geometry(intersection_cvhull, node_name=f"intersect")
    #     scene.show()
    return polygon.contains(Point(CoM_xy)) # If contains, return True

def move_piece(piece, translation):
    piece['mesh'].apply_translation(translation)
    piece['corners'] = [(cid, pos+translation) for cid,pos in piece['corners']]
    if piece['gripper_config'] is not None:
        piece['gripper_config']['position'] = piece['gripper_config']['position'] + translation
    return piece

# Alt form of move_piece
def apply_mate(pieces, mate):
    (this_pid, this_cid), (other_pid, other_cid) = mate
    this_piece = next(p for p in pieces if p['id'] == this_pid)
    other_piece = next(p for p in pieces if p['id'] == other_pid)
    this_corner = next(c for c in this_piece['corners'] if c[0] == this_cid)
    other_corner = next(c for c in other_piece['corners'] if c[0] == other_cid)
    vec = other_corner[1] - this_corner[1]
    this_piece = move_piece(this_piece, vec)
    # This should just update the piece info but I'll return the set
    return pieces