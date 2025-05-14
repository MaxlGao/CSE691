import itertools
from helper_burr_reference import ORIENTATIONS_DICTS, transform_boxes
from helper_burr_piece_creator import define_all_burr_pieces
import concurrent.futures
from tqdm import tqdm
import numpy as np
import trimesh

# Pre-Oriented
BURR_SHAPE = [    
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

def generate_permutations():
    all_perms = []
    base_indices = [1, 2, 3, 4, 5]
    
    # Generate all permutations of base slots
    for perm in itertools.permutations(base_indices):
        # For each permutation, generate all 2^5 form combinations
        for form_bits in range(32):
            alt_perm = [0] # Fix first position
            for i in range(5):
                # Check if the i-th bit is 1 (use alternate form)
                if (form_bits >> i) & 1:
                    alt_perm.append(perm[i] + 6)  # Use alternate form
                else:
                    alt_perm.append(perm[i])      # Use original form
            all_perms.append(alt_perm)
    
    return all_perms

def verify_arrangement(permutation):
    rotations = [ORIENTATIONS_DICTS[i]['rotation'] for i in permutation]
    translations = [ORIENTATIONS_DICTS[i]['position'] for i in permutation]
    burr_shape_tf = [
        transform_boxes(shape, rot)
        for shape, rot in zip(BURR_SHAPE, rotations)
    ]
    offsets = np.array(translations)
    pieces = define_all_burr_pieces(boxes=burr_shape_tf, offsets=offsets)
    for piece in pieces:
        this_pid = piece["id"]
        this_mesh = piece["mesh"]
        other_meshes = [p["mesh"] for p in pieces if p['id'] != this_pid]
        for other_mesh in other_meshes:
            if trimesh.boolean.intersection([this_mesh, other_mesh], check_volume=False).volume > 0.001:
                return False, permutation
    # If you made it this far, return true
    return True, permutation

if __name__=="__main__":
    permutations = generate_permutations()

    feasible = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(verify_arrangement, permutation) for permutation in permutations]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), ncols=100):
            result = future.result()
            if result[0]:
                feasible.append(result)
                print("Hit!")
    print(f"There are {len(feasible)} solutions.")
    print(f"Listed: {feasible}")