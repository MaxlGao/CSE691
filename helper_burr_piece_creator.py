import trimesh
import numpy as np
from helper_geometric_functions import move_piece

REFERENCE_INDEX_OFFSET = 50 # Bump up reference piece indices
FLOOR_INDEX_OFFSET = 100 # Bump up floor piece index
UNINTERESTING_CORNERS = { # Uses Corner IDs according to find_cube_corners
    1: {1, 5, 10, 12, 15, 21, 23, 29},
    2: {1, 13, 14, 19, 21, 22, 28},
    3: {6, 12, 20, 26},
    4: {1, 2, 3, 4, 12, 19, 23},
    5: {1, 6, 12, 17},
    # piece 0: none
}
START_OFFSETS_CORNERS = np.array([[1,-9,0],[-9,-3,0],[7,-3,0],[-5,7,0],[-5,-9,0],[1,7,0]])
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

def create_floor(reverse=False):
    """
    Create a Floor. Returns a dict like create_burr_piece
    """
    floor = trimesh.creation.box(extents=[20,20,1])
    floor.apply_translation([0, 0, -0.5])
    floor.visual.face_colors = [200, 200, 200, 255]

    if reverse:
        # One point for each starting position
        corners = [(i, pos) for i, pos in enumerate(START_OFFSETS_CORNERS)] 
    else:
        # A selection of six points in the center of the build space.
        corners = [(0,[-1,0,0]), (1,[1,0,0]),
                   (2,[-1,2,0]), (3,[1,2,0]),
                   (4,[-1,-2,0]), (5,[1,-2,0])] 
    return {'mesh': floor, 'corners': corners, 'gripper_config': None, 'id': FLOOR_INDEX_OFFSET}

def create_burr_piece(blocks, color, idx, reference=False, gripper=None):
    """
    Create a Burr piece from a list of block specs.
    Each block is a dict: {'size': [x, y, z], 'position': [x, y, z]}
    Returns a dict: {'mesh': trimesh.Trimesh, 
                     'corners': list of (index, [x, y, z]),
                     'gripper_config': (width, [x, y, z], rotation matrix)
                     'id': id of piece}
    """
    meshes = []
    for block in blocks:
        size = block['size']
        position = block['position']
        box = trimesh.creation.box(extents=size)
        box.apply_translation(position)
        meshes.append(box)

    composite = trimesh.boolean.union(meshes)
    composite.process(validate=True)
    composite.visual.face_colors = color
    corners = find_cube_corners(composite)
    bad_cids = UNINTERESTING_CORNERS.get(idx, set())
    filtered_corners = [(cid, pos) for cid, pos in corners if cid not in bad_cids]
    if reference:
        gripper = None
        idx += REFERENCE_INDEX_OFFSET
    return {'mesh': composite, 'corners': filtered_corners, 'gripper_config': gripper, 'id': idx}

def create_gripper(config=None, color=[127, 127, 127, 255]):
    if config is None:
        return None
    else:
        width = config[0]
    # Create parts
    thickness_tip = 0.25
    thickness_finger = 0.5
    fingertip = trimesh.creation.cylinder(radius=1, height=thickness_tip, sections=8)
    finger = trimesh.creation.box(extents=[4, 2, thickness_finger])
    palm = trimesh.creation.box(extents=[1.5, 2, 8])

    # Move parts
    left_fingertip = fingertip.copy()
    left_fingertip.apply_translation([0, 0, width/2+thickness_tip/2])
    left_finger = finger.copy()
    left_finger.apply_translation([-2, 0, width/2+thickness_tip+thickness_finger/2])

    right_fingertip = fingertip.copy()
    right_fingertip.apply_translation([0, 0, -width/2-thickness_tip/2])
    right_finger = finger.copy()
    right_finger.apply_translation([-2, 0, -width/2-thickness_tip-thickness_finger/2])

    palm.apply_translation([-4-1.5/2, 0, 0])

    gripper = trimesh.util.concatenate([left_finger, right_finger, left_fingertip, right_fingertip, palm])
    gripper.visual.face_colors = color
    if config is not None:
        gripper.apply_transform(config[2])
        gripper.apply_translation(config[1])
    return gripper

def trirotmat(angledeg, direction):
    return trimesh.transformations.rotation_matrix(angle=np.radians(angledeg), direction=direction, point=[0, 0, 0])

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

    gripper_configs = [
        (2, [ 3,  0, 1], trirotmat(90, [1, 0, 0]) @ trirotmat(-135, [0, 0, 1])),
        (6, [-1, 0, 1], trirotmat(90, [1, 0, 0]) @ trirotmat(-45, [0, 0, 1])),
        (6, [ 1, 0, 1], trirotmat(90, [1, 0, 0]) @ trirotmat(-135, [0, 0, 1])),
        (2, [ 0,  1, 3], trirotmat(90, [0, 1, 0]) @ trirotmat(-45, [0, 0, 1])),
        (2, [ 0, -1, 3], trirotmat(90, [0, 1, 0]) @ trirotmat(45, [0, 0, 1])),
        (2, [ 3,  0, 0], trirotmat(90, [1, 0, 0]) @ trirotmat(180, [0, 0, 1]))
    ]

    pieces = []

    pieces.append(create_burr_piece([
        {'size': [6, 2, 2], 'position': [0, 0, 0]},
    ], colors[0], 0, reference=reference, gripper=gripper_configs[0]))
    pieces.append(create_burr_piece([
        {'size': [1, 2, 1], 'position': [-0.5, 0, -0.5]},
        {'size': [1, 2, 2], 'position': [-0.5, -2, 0]},
        {'size': [1, 2, 2], 'position': [-0.5, 2, 0]},
        {'size': [1, 1, 2], 'position': [0.5, -2.5, 0]},
        {'size': [1, 1, 2], 'position': [0.5, 2.5, 0]},
    ], colors[1], 1, reference=reference, gripper=gripper_configs[1]))
    pieces.append(create_burr_piece([
        {'size': [2, 2, 2], 'position': [0, -2, 0]},
        {'size': [1, 2, 1], 'position': [0.5, 0, -0.5]},
        {'size': [1, 2, 2], 'position': [0.5, 2, 0]},
        {'size': [1, 1, 2], 'position': [-0.5, 2.5, 0]},
        {'size': [1, 1, 1], 'position': [-0.5, -0.5, -0.5]},
    ], colors[2], 2, reference=reference, gripper=gripper_configs[2]))
    pieces.append(create_burr_piece([
        {'size': [2, 1, 6], 'position': [0, 0.5, 0]},
        {'size': [2, 1, 1], 'position': [0, -0.5, 2.5]},
        {'size': [2, 1, 1], 'position': [0, -0.5, -0.5]},
        {'size': [2, 1, 1], 'position': [0, -0.5, -2.5]},
    ], colors[3], 3, reference=reference, gripper=gripper_configs[3]))
    pieces.append(create_burr_piece([
        {'size': [2, 2, 1], 'position': [0, 0, -2.5]},
        {'size': [2, 2, 1], 'position': [0, 0, 2.5]},
        {'size': [2, 1, 1], 'position': [0, -0.5, -1.5]},
        {'size': [2, 1, 1], 'position': [0, -0.5, 1.5]},
        {'size': [1, 1, 2], 'position': [-0.5, -0.5, 0]},
        {'size': [1, 1, 1], 'position': [-0.5, 0.5, -0.5]},
    ], colors[4], 4, reference=reference, gripper=gripper_configs[4]))
    pieces.append(create_burr_piece([
        {'size': [6, 2, 1], 'position': [0, 0, -0.5]},
        {'size': [1, 2, 1], 'position': [2.5, 0, 0.5]},
        {'size': [1, 2, 1], 'position': [-2.5, 0, 0.5]},
    ], colors[5], 5, reference=reference, gripper=gripper_configs[5]))

    if start_offsets is not None:
        for piece in pieces:
            piece = move_piece(piece, start_offsets[piece['id']])
    return pieces