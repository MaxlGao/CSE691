import numpy as np
import time
from helper_burr_piece_creator import create_floor, define_all_burr_pieces, REFERENCE_INDEX_OFFSET, FLOOR_INDEX_OFFSET
from helper_geometric_functions import check_path_clear, get_feasible_motions, get_valid_mates, is_supported, move_piece
from helper_display import render_scene, show_moves_scored, save_animation_frame, display_pieces
from helper_file_mgmt import load_mates_list, load_simulation_state, save_mates_list, save_simulation_state
from main_trimesh import get_top_k_scored_moves
import copy


# Scripts for making figures
# Place two pieces and draw arrows between all feasible mates
def display_connections(camera = [-15.0, -24.0, 15.0, 0.0]):
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
    pieces[3] = move_piece(pieces[3], target_offsets[3] - start_offsets[3])
    pieces[5] = move_piece(pieces[5], [0, -16, 0])

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
    active_pids = [FLOOR_INDEX_OFFSET, 3]
    scored_moves = get_top_k_scored_moves(pieces_augmented, active_pids, target_offsets, mates_list=mates_list)
    # print(f"| Processed {len(scored_moves)} moves.")

    # Filter to only look at piece 5
    scored_moves = [scored_move for scored_move in scored_moves if scored_move[1][0][0] == 5]
    print(f"| Filtered to {len(scored_moves)} moves.")

    arrows = show_moves_scored(scored_moves, pieces, floor)
    # Filter arrows list to be every 4th arrow
    # arrows = [arrow for i, arrow in enumerate(arrows) if np.mod(i, 4) == 0]
    print(f"| Filtered to {len(arrows)} moves.")

    scene_pieces = [reference_pieces[3], reference_pieces[5], pieces[3], pieces[5], floor]
    scene = render_scene(scene_pieces, arrows=arrows, camera=camera)
    save_animation_frame(scene, 100, suffix='b')

    # Now display the motion of the top move
    best_move = scored_moves[0]
    _, ((_,_),(_,_),vec) = best_move
    steps=20
    this_mesh = pieces[5]['mesh']
    scene = render_scene(scene_pieces, arrows=[arrows[0]], camera=camera)
    for step in range(1, steps + 1):
        frac_translation = (vec * step) / (steps)
        test_mesh = this_mesh.copy()
        test_mesh.apply_translation(frac_translation)
        test_mesh.visual.face_colors = [255, 0, 255, 20]
        scene.add_geometry(test_mesh, f"interstitial_{step}")
    save_animation_frame(scene, 101, suffix='b')

    # Lastly, display the final state
    pieces[5] = move_piece(pieces[5], vec)
    scene = render_scene(scene_pieces, camera=camera)
    save_animation_frame(scene, 102, suffix='b')

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
            # Quick and dirty check for collision
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
            # Check supports
            support = is_supported(test_mesh, other_meshes_not_floor) # Including the floor is inefficient
            supports.append(support)
            # if support and vec[2] > 0: # Show cases where a piece lays on top of another.
            #     scene = render_scene(all_pieces + [test_piece])
            #     scene.show()
    print(f"There are {supports.count(True)} configurations with support and {supports.count(False)} without.")
    return

if __name__=="__main__":
    start_time = time.time()
    # scene = display_pieces()
    scene = display_connections()
    scene.show()
