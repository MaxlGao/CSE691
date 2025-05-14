import trimesh
import numpy as np
import copy
from helper_burr_reference import ORIENTATIONS_DICTS

# Not correctly oriented
burr_shape_hard = [    
    [
        {'size': [2, 1, 2], 'position': [0, -2.5, 0]},
        {'size': [2, 1, 2], 'position': [0,  2.5, 0]},
        {'size': [1, 2, 2], 'position': [0.5,  1, 0]},
        {'size': [1, 1, 1], 'position': [0.5, -0.5, -0.5]},
        {'size': [1, 1, 2], 'position': [0.5, -1.5, 0]},
    ],
    [
        {'size': [2, 1, 2], 'position': [0, -2.5, 0]},
        {'size': [2, 1, 2], 'position': [0,  2.5, 0]},
        {'size': [1, 6, 2], 'position': [0.5,  0, 0]},
        {'size': [1, 1, 2], 'position': [-0.5, -1.5, 0]},
    ],
    [
        {'size': [2, 1, 2], 'position': [0, -2.5, 0]},
        {'size': [2, 1, 2], 'position': [0,  2.5, 0]},
        {'size': [1, 6, 2], 'position': [0.5,  0, 0]},
    ],
    [
        {'size': [2, 1, 2], 'position': [0, -2.5, 0]},
        {'size': [2, 1, 2], 'position': [0,  2.5, 0]},
        {'size': [1, 1, 2], 'position': [0.5,  1.5, 0]},
        {'size': [1, 1, 2], 'position': [0.5, -1.5, 0]},
        {'size': [1, 1, 1], 'position': [-0.5,  1.5, -0.5]},
        {'size': [1, 1, 1], 'position': [-0.5, -1.5, -0.5]},
        {'size': [1, 2, 1], 'position': [0.5, 0, -0.5]},
    ],
    [
        {'size': [2, 1, 2], 'position': [0, -2.5, 0]},
        {'size': [2, 1, 2], 'position': [0,  2.5, 0]},
        {'size': [1, 6, 2], 'position': [0.5,  0, 0]},
        {'size': [1, 2, 1], 'position': [-0.5,  0, -0.5]},
    ],
    [
        {'size': [2, 1, 2], 'position': [0, -2.5, 0]},
        {'size': [2, 1, 2], 'position': [0,  2.5, 0]},
        {'size': [1, 1, 2], 'position': [0.5,  1.5, 0]},
        {'size': [1, 1, 2], 'position': [0.5, -1.5, 0]},
        {'size': [1, 1, 1], 'position': [-0.5, 0.5, -0.5]},
        {'size': [1, 2, 1], 'position': [0.5, 0, -0.5]},
    ]
]

def trirotmat(angledeg, direction):
    return trimesh.transformations.rotation_matrix(angle=np.radians(angledeg), direction=direction, point=[0, 0, 0])
def rotate_boxes(boxes, matrix):
    transformed = []
    for box in boxes:
        size = np.array(box['size'] + [1])  # Convert to homogeneous coords
        pos = np.array(box['position'] + [1])  # Convert to homogeneous coords
        new_size = np.round(np.abs(matrix @ size), 6) # Absolute value and trim to 6th decimal point
        new_pos = np.round(matrix @ pos, 6)
        transformed.append({
            'size': new_size[:3].tolist(),
            'position': new_pos[:3].tolist()
        })
    return transformed

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
    return {'mesh': composite, 'id': idx}

# Opposing orientations are 0/6 and 1/7, 2/8 and 3/9, 4/10 and 5/11
# pcs (position) 1 and 2 must oppose
possible_orientations = [
    [0, 8, 3, 11, 10, 7],
    [0, 8, 4, 11, 3, 7]]
# Orientations for all pieces to correct position, picking from the following list
for orientation in possible_orientations:
    # ORI_IDS_HARD = [ 4, 7, 0, 8, 9, 11]
    print(orientation)

    ROTATIONS_HARD = [ORIENTATIONS_DICTS[i]['rotation'] for i in orientation]
    TRANSLATIONS_HARD = np.array([ORIENTATIONS_DICTS[i]['position'] for i in orientation])

    TF_BURR_HARD_SHAPE = [
        rotate_boxes(shape, rot)
        for shape, rot in zip(burr_shape_hard, ROTATIONS_HARD)
    ]

    opacity = 200
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
        pieces.append(create_burr_piece(TF_BURR_HARD_SHAPE[i], colors[i], i))

    def move_piece(piece, translation):
        piece['mesh'].apply_translation(translation)
        return piece


    scene = trimesh.Scene()
    for i, piece in enumerate(pieces):
        # piece = copy.deepcopy(pieces[0])
        # piece['mesh'].apply_transform(ROTATIONS_HARD[i])
        # piece = move_piece(piece, [5*i, 0, 0])
        piece = move_piece(piece, 4*TRANSLATIONS_HARD[i])
        scene.add_geometry(piece['mesh'], node_name=f"mesh_{i}")
    scene.show()



