import trimesh
import numpy as np
import imageio
import os
from pathlib import Path
import re
from helper_burr_piece_creator import create_floor, define_all_burr_pieces
from helper_burr_piece_creator import REFERENCE_INDEX_OFFSET, FLOOR_INDEX_OFFSET
from helper_file_mgmt import load_simulation_state
from helper_geometric_functions import move_piece

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

def show_moves_scored(scored_moves, pieces, floor, opacity=0.6, limit=1000):
    num_moves = min(len(scored_moves), limit)
    # hue_range = 0.333 * np.flip(np.arange(num_moves)) / num_moves
    arrows = []
    for i, (score, move) in enumerate(scored_moves[:num_moves]):
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

def save_animation_frame(scene, index, folder="results", suffix=''):
    os.makedirs(folder, exist_ok=True)

    # Save as image
    image_path = os.path.join(folder, f"frame_{index:03d}{suffix}.png")
    png = scene.save_image(resolution=(800, 600), visible=True)
    with open(image_path, 'wb') as f:
        f.write(png)

    print(f"Saved frame {index} to {image_path}")

def show_and_save_frames(folder_sim, folder_img, target_offsets, hold=False, start_from=0):
    # Automatically find all simulation step files
    sim_folder = Path(folder_sim)
    step_files = sorted(sim_folder.glob('step_*.pkl'))

    # Extract step indices from filenames
    step_indices = sorted([
        int(re.search(r'step_(\d+)\.pkl', f.name).group(1))
        for f in step_files
    ])

    reference_pieces = None

    for sc in step_indices:
        if sc < start_from:
            continue
        state = load_simulation_state(sc, folder=folder_sim)
        pieces = state['pieces']

        if sc == 0:  # On first step only
            reference_pieces = define_all_burr_pieces(reference=True)
            target_offsets = target_offsets + [0, 0, 3]
            for piece in reference_pieces:
                piece = move_piece(piece, target_offsets[piece['id'] - REFERENCE_INDEX_OFFSET])
                pieces.append(piece)

        scene = render_scene(pieces)
        save_animation_frame(scene, sc, folder=folder_img)
        if hold:
            scene.show()

        floor = create_floor()
        arrows = show_moves_scored(state['available_moves'], state['pieces'], floor)
        scene = render_scene(state['pieces'], arrows=arrows)
        save_animation_frame(scene, sc, folder=folder_img, suffix='a')
        if hold:
            scene.show()

def compile_gif(folder="results", suffix='', gif_name='animation.gif', fps=4):
    path = Path(folder)
    frame_files = sorted(path.glob(f"frame_*{suffix}.png"))
    images = [imageio.imread(str(frame)) for frame in frame_files]
    # Hold last image for a bit
    for i in range(fps):
        images.append(images[-1])

    gif_path = path / gif_name
    imageio.mimsave(gif_path, images, fps=fps, loop=0)
    print(f"GIF saved to {gif_path}")

# Show all pieces in a line, then all of them put together.
def display_pieces():
    target_offsets = np.array([[0,0,1],[-1,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,-1]])
    target_offsets += [0,0,3]

    # Image 1: Pieces put together
    pieces = define_all_burr_pieces(target_offsets)
    floor = create_floor()
    scene = render_scene(pieces + [floor])
    save_animation_frame(scene, 200, suffix='b')

    # # Image 2: Pieces in a line
    # start_offsets = np.array([[-10,-6,1],[-5,-6,1],[-2,-6,1],[2,6,3],[5,6,3],[10,6,1]])
    # # target_offsets += [0,-4,0]
    # pieces = define_all_burr_pieces(start_offsets)
    # scene = render_scene(pieces + [floor], camera=[15.0, -24.0, 15.0, 0.0])
    # save_animation_frame(scene, 201, suffix='b')