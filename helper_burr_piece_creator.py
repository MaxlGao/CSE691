import trimesh
import numpy as np
from helper_burr_reference import BURR_DICT_EASY, BURR_DICT_HARD

BURR_DICT = BURR_DICT_EASY
boxes = BURR_DICT["boxes"]
bad_corners = BURR_DICT["bad_corners"]
target_offsets = BURR_DICT["target_offsets"]
initial_offsets = BURR_DICT["initial_offsets"]
floor_offsets_corners = BURR_DICT["floor_offsets_corners"]
gripper_configs = BURR_DICT["gripper_configs"]

REFERENCE_INDEX_OFFSET = 50 # Bump up reference piece indices
FLOOR_INDEX_OFFSET = 100 # Bump up floor piece index

def trirotmat(angledeg, direction):
    return trimesh.transformations.rotation_matrix(angle=np.radians(angledeg), direction=direction, point=[0, 0, 0])


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
        corners = [(i, pos) for i, pos in enumerate(floor_offsets_corners)] 
    else:
        # A selection of six points in the center of the build space.
        corners = [(0,[-1,0,0]), (1,[1,0,0]),
                   (2,[-1,2,0]), (3,[1,2,0]),
                   (4,[-1,-2,0]), (5,[1,-2,0])] 
    return {'mesh': floor, 'corners': corners, 'gripper_config': None, 'id': FLOOR_INDEX_OFFSET}

def create_burr_piece(blocks, color, idx, reverse, reference=False, gripper=None):
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
    bad_cids = bad_corners.get(idx, set())
    filtered_corners = [(cid, pos) for cid, pos in corners if cid not in bad_cids]
    target = initial_offsets[idx] if reverse else target_offsets[idx]
    if reference:
        gripper = None
        idx += REFERENCE_INDEX_OFFSET
    return {'mesh': composite, 'corners': filtered_corners, 'gripper_config': gripper, 'target': target, 'id': idx}

def create_gripper(config=None, color=[127, 127, 127, 255]):
    if config is None:
        return None
    else:
        width = config['width']
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
        gripper.apply_transform(config['rotation'])
        gripper.apply_translation(config['position'])
    return gripper

def define_all_burr_pieces(boxes=boxes, offsets=None, reference=False, reverse=False):
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

    for i in range(6):
        pieces.append(create_burr_piece(boxes[i], colors[i], i, reverse, reference=reference, gripper=gripper_configs[i]))

    if offsets is None: # If not hard-defined
        offsets = BURR_DICT['target_offsets'] if reverse else BURR_DICT['initial_offsets']
    for piece in pieces:
        piece = move_piece(piece, offsets[piece['id']])
    return pieces

def move_piece(piece, translation):
    piece['mesh'].apply_translation(translation)
    piece['corners'] = [(cid, pos+translation) for cid,pos in piece['corners']]
    if piece['gripper_config'] is not None:
        piece['gripper_config']['position'] = piece['gripper_config']['position'] + translation
    return piece