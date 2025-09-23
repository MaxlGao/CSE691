import numpy as np
import trimesh

def trirotmat(angledeg, direction):
    return trimesh.transformations.rotation_matrix(angle=np.radians(angledeg), direction=direction, point=[0, 0, 0])

def transform_boxes(boxes, matrix):
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

GRIPPER_CONFIGS_EASY = [ # Width, Active, Position, Rotation
    {'width': 2, 'active': False, 'position': [ 3,  0, 1], 'rotation': trirotmat(90, [1, 0, 0]) @ trirotmat(-135, [0, 0, 1])},
    {'width': 6, 'active': False, 'position': [-1,  0, 1], 'rotation': trirotmat(90, [1, 0, 0]) @ trirotmat( -45, [0, 0, 1])},
    {'width': 6, 'active': False, 'position': [ 1,  0, 1], 'rotation': trirotmat(90, [1, 0, 0]) @ trirotmat(-135, [0, 0, 1])},
    {'width': 2, 'active': False, 'position': [ 0,  1, 3], 'rotation': trirotmat(90, [0, 1, 0]) @ trirotmat( -45, [0, 0, 1])},
    {'width': 2, 'active': False, 'position': [ 0, -1, 3], 'rotation': trirotmat(90, [0, 1, 0]) @ trirotmat(  45, [0, 0, 1])},
    {'width': 2, 'active': False, 'position': [ 3,  0, 0], 'rotation': trirotmat(90, [1, 0, 0]) @ trirotmat( 180, [0, 0, 1])}
]

GRIPPER_CONFIGS_HARD = [ # Width, Active, Position, Rotation
    {'width': 6, 'active': False, 'position': [ 1,  0, 1], 'rotation': trirotmat(90, [1, 0, 0]) @ trirotmat(-135, [0, 0, 1])},
    {'width': 2, 'active': False, 'position': [ 0,  1, 3], 'rotation': trirotmat(90, [0, 1, 0]) @ trirotmat( -45, [0, 0, 1])},
    {'width': 2, 'active': False, 'position': [ 3,  0, 0], 'rotation': trirotmat(90, [1, 0, 0]) @ trirotmat( 180, [0, 0, 1])},
    {'width': 2, 'active': False, 'position': [ 3,  0, 1], 'rotation': trirotmat(90, [1, 0, 0]) @ trirotmat(-135, [0, 0, 1])},
    {'width': 2, 'active': False, 'position': [ 0, -1, 3], 'rotation': trirotmat(90, [0, 1, 0]) @ trirotmat(  45, [0, 0, 1])},
    {'width': 6, 'active': False, 'position': [-1,  0, 1], 'rotation': trirotmat(90, [1, 0, 0]) @ trirotmat( -45, [0, 0, 1])}
]

BURR_DICT_EASY = {
    "boxes": [ # Easy boxes are already oriented correctly
        [
            {'size': [6, 2, 2], 'position': [0, 0, 0]},
        ],
        [
            {'size': [1, 2, 1], 'position': [-0.5, 0, -0.5]},
            {'size': [1, 2, 2], 'position': [-0.5, -2, 0]},
            {'size': [1, 2, 2], 'position': [-0.5, 2, 0]},
            {'size': [1, 1, 2], 'position': [0.5, -2.5, 0]},
            {'size': [1, 1, 2], 'position': [0.5, 2.5, 0]},
        ],
        [
            {'size': [2, 2, 2], 'position': [0, -2, 0]},
            {'size': [1, 2, 1], 'position': [0.5, 0, -0.5]},
            {'size': [1, 2, 2], 'position': [0.5, 2, 0]},
            {'size': [1, 1, 2], 'position': [-0.5, 2.5, 0]},
            {'size': [1, 1, 1], 'position': [-0.5, -0.5, -0.5]},
        ],
        [
            {'size': [2, 1, 6], 'position': [0, 0.5, 0]},
            {'size': [2, 1, 1], 'position': [0, -0.5, 2.5]},
            {'size': [2, 1, 1], 'position': [0, -0.5, -0.5]},
            {'size': [2, 1, 1], 'position': [0, -0.5, -2.5]},
        ],
        [
            {'size': [2, 2, 1], 'position': [0, 0, -2.5]},
            {'size': [2, 2, 1], 'position': [0, 0, 2.5]},
            {'size': [2, 1, 1], 'position': [0, -0.5, -1.5]},
            {'size': [2, 1, 1], 'position': [0, -0.5, 1.5]},
            {'size': [1, 1, 2], 'position': [-0.5, -0.5, 0]},
            {'size': [1, 1, 1], 'position': [-0.5, 0.5, -0.5]},
        ],
        [
            {'size': [6, 2, 1], 'position': [0, 0, -0.5]},
            {'size': [1, 2, 1], 'position': [2.5, 0, 0.5]},
            {'size': [1, 2, 1], 'position': [-2.5, 0, 0.5]},
        ]
    ],
    "bad_corners":
    { # Uses Corner IDs according to find_cube_corners
        1: {1, 5, 10, 12, 15, 21, 23, 29},
        2: {1, 13, 14, 19, 21, 22, 28},
        3: {6, 12, 20, 26},
        4: {1, 2, 3, 4, 12, 19, 23},
        5: {1, 6, 12, 17},
        # piece 0: none
    },
    "target_offsets": np.array([[0,0,4],[-1,0,3],[1,0,3],[0,1,3],[0,-1,3],[0,0,2]]),
    "initial_offsets": np.array([[4,-8,1],[-8,0,1],[8,0,1],[-4,8,3],[-4,-8,3],[4,8,1]]),
    "floor_offsets_corners": np.array([[1,-9,0],[-9,-3,0],[7,-3,0],[-5,7,0],[-5,-9,0],[1,7,0]]),
    "gripper_configs": GRIPPER_CONFIGS_EASY
}

# From here, assume base piece orientation is like this:
#    ^y     z up
#   OOOO
#     OO
#     OO  -> x
#     OO
#     OO
#   OOOO

ORIENTATIONS_DICTS = [ # Positions defined in world frame
    # Base six orientations
    {'position': [ 1, 0,3], 'rotation': trirotmat(   0, [0, 1, 0])},
    {'position': [-1, 0,3], 'rotation': trirotmat( 180, [0, 1, 0])},
    {'position': [ 0, 1,3], 'rotation': trirotmat(  90, [1, 0, 0]) @ trirotmat(  90, [0, 1, 0])},
    {'position': [ 0,-1,3], 'rotation': trirotmat(  90, [1, 0, 0]) @ trirotmat( -90, [0, 1, 0])},
    {'position': [ 0, 0,2], 'rotation': trirotmat(  90, [0, 0, 1]) @ trirotmat(  90, [0, 1, 0])},
    {'position': [ 0, 0,4], 'rotation': trirotmat(  90, [0, 0, 1]) @ trirotmat( -90, [0, 1, 0])},
    # Flipped of previous six
    {'position': [ 1, 0,3], 'rotation': trirotmat( 180, [1, 0, 0])},
    {'position': [-1, 0,3], 'rotation': trirotmat( 180, [1, 0, 0]) @ trirotmat( 180, [0, 1, 0])},
    {'position': [ 0, 1,3], 'rotation': trirotmat( 180, [0, 1, 0]) @ trirotmat(  90, [1, 0, 0]) @ trirotmat(  90, [0, 1, 0])},
    {'position': [ 0,-1,3], 'rotation': trirotmat( 180, [0, 1, 0]) @ trirotmat(  90, [1, 0, 0]) @ trirotmat( -90, [0, 1, 0])},
    {'position': [ 0, 0,2], 'rotation': trirotmat( 180, [0, 0, 1]) @ trirotmat(  90, [0, 0, 1]) @ trirotmat(  90, [0, 1, 0])},
    {'position': [ 0, 0,4], 'rotation': trirotmat( 180, [0, 0, 1]) @ trirotmat(  90, [0, 0, 1]) @ trirotmat( -90, [0, 1, 0])},
]


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

ori_ids_hard = [0, 8, 4, 11, 3, 7]
rotations_hard = [ORIENTATIONS_DICTS[i]['rotation'] for i in ori_ids_hard]
translations_hard = [ORIENTATIONS_DICTS[i]['position'] for i in ori_ids_hard]
burr_shape_hard = [
    transform_boxes(shape, rot)
    for shape, rot in zip(burr_shape_hard, rotations_hard)
]

BURR_DICT_HARD = {
    "boxes": burr_shape_hard,
    "bad_corners": {},
    "target_offsets": np.array(translations_hard),
    "initial_offsets": np.array([[4,-8,1],[-8,0,1],[8,0,1],[-4,8,3],[-4,-8,3],[4,8,1]]),
    "floor_offsets_corners": np.array([[1,-9,0],[-9,-3,0],[7,-3,0],[-5,7,0],[-5,-9,0],[1,7,0]]),
    "gripper_configs": GRIPPER_CONFIGS_HARD
}