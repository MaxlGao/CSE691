import numpy as np
import trimesh

# Friction constants
STATIC_FRICTION = 0.3      # Coefficient of static friction
KINETIC_FRICTION = 0.2     # Coefficient of kinetic friction
MIN_MOVEMENT_THRESHOLD = 0.05  # Minimum force needed to overcome static friction

def detect_contact_points(piece1, piece2, tolerance=0.05):
    """
    Detect contact points between two pieces
    Returns list of contact points
    
    Parameters:
    - piece1, piece2: Burr puzzle piece dictionaries
    - tolerance: Distance threshold to consider surfaces in contact
    
    Returns:
    - List of contact points [(point1, point2), ...]
    """
    mesh1 = piece1['mesh']
    mesh2 = piece2['mesh']
    
    # Basic approximation - check vertex proximity
    contact_points = []
    
    # Check mesh1 vertices against mesh2 faces (sample a subset for efficiency)
    vertex_indices = np.random.choice(len(mesh1.vertices), min(100, len(mesh1.vertices)), replace=False)
    
    for idx in vertex_indices:
        vertex = mesh1.vertices[idx]
        closest_points, _, _ = trimesh.proximity.closest_point(mesh2, [vertex])
        closest_point = closest_points[0]
        distance = np.linalg.norm(closest_point - vertex)
        if distance < tolerance:
            contact_points.append((vertex, closest_point))
    
    return contact_points

def find_contacting_pieces(piece, other_pieces, tolerance=0.05):
    """
    Find pieces that are in contact with the given piece
    
    Parameters:
    - piece: The piece to check for contacts
    - other_pieces: List of other pieces to check against
    - tolerance: Distance threshold for contact detection
    
    Returns:
    - List of contacting pieces
    """
    contacting = []
    for other in other_pieces:
        # Skip self-comparison
        if piece['id'] == other['id']:
            continue
            
        # Quick check using bounding box proximity
        if piece['mesh'].bounding_box.intersection(other['mesh'].bounding_box):
            # More detailed check
            contacts = detect_contact_points(piece, other, tolerance)
            if contacts:
                contacting.append(other)
    
    return contacting

def calculate_friction_force(piece, contacting_pieces, movement_vector):
    """
    Calculate friction force based on contacts
    
    Parameters:
    - piece: The moving piece
    - contacting_pieces: List of pieces in contact
    - movement_vector: Intended movement direction and magnitude
    
    Returns:
    - Modified movement vector after friction
    - Boolean indicating if movement is possible
    """
    if not contacting_pieces:
        return movement_vector, True
    
    # Count contact points for a rough estimate of normal force
    contact_count = 0
    for other in contacting_pieces:
        contacts = detect_contact_points(piece, other)
        contact_count += len(contacts)
    
    # Calculate forces
    force_magnitude = np.linalg.norm(movement_vector)
    
    # Check if movement can overcome static friction
    if force_magnitude < MIN_MOVEMENT_THRESHOLD * contact_count:
        # Can't overcome static friction
        return np.zeros_like(movement_vector), False
    
    # Apply kinetic friction
    friction_factor = max(0, 1 - (KINETIC_FRICTION * contact_count / force_magnitude))
    reduced_vector = movement_vector * friction_factor
    
    return reduced_vector, True

def move_piece_with_friction(piece, translation_vec, contacting_pieces):
    """
    Move a piece accounting for friction from contacting pieces
    
    Parameters:
    - piece: The piece to move
    - translation_vec: Desired movement vector
    - contacting_pieces: List of pieces in contact
    
    Returns:
    - Modified piece after movement (or original if friction prevents movement)
    - Boolean indicating if movement was possible
    """
    from helper_geometric_functions import move_piece
    
    # Apply friction effects to movement
    reduced_vec, can_move = calculate_friction_force(piece, contacting_pieces, translation_vec)
    
    if not can_move or np.allclose(reduced_vec, 0):
        return piece, False
    
    # Move with reduced translation vector
    moved_piece = move_piece(piece, reduced_vec)
    return moved_piece, True

def check_path_clear_with_friction(moving_piece, other_pieces, translation, steps=10):
    """
    Check if a path is clear considering friction effects
    
    Parameters:
    - moving_piece: The piece to move
    - other_pieces: Other pieces in the scene
    - translation: Desired movement vector
    - steps: Number of intermediate steps to check
    
    Returns:
    - Boolean indicating if path is clear and movement is possible
    """
    from helper_geometric_functions import check_path_clear
    
    # Find pieces in contact
    contacting_pieces = find_contacting_pieces(moving_piece, other_pieces)
    
    # Apply friction to movement vector
    reduced_vec, can_move = calculate_friction_force(moving_piece, contacting_pieces, translation)
    
    if not can_move:
        return False
    
    # Use existing path checking with reduced vector
    return check_path_clear(moving_piece['mesh'], [p['mesh'] for p in other_pieces], reduced_vec, steps)

def get_cost_change_with_friction(piece, translation, target_offset, contacting_pieces):
    """
    Calculate cost change for a move considering friction effects
    
    Parameters:
    - piece: The piece to move
    - translation: Desired movement vector
    - target_offset: Target position for the piece
    - contacting_pieces: List of pieces in contact
    
    Returns:
    - Cost change value (smaller is better)
    """
    # Apply friction to calculate actual movement
    reduced_vec, can_move = calculate_friction_force(piece, contacting_pieces, translation)
    
    if not can_move:
        return float('inf')  # Movement not possible due to friction
    
    # Calculate cost with adjusted translation
    current_location = piece['mesh'].bounding_box.centroid
    current_cost = np.linalg.norm(current_location - target_offset)
    new_location = current_location + reduced_vec
    new_cost = np.linalg.norm(new_location - target_offset)
    diff_cost = new_cost - current_cost
    
    # Add small penalty for moves with friction (encourages smoother paths)
    friction_penalty = 0.001 * len(contacting_pieces)
    
    return np.round(diff_cost + friction_penalty, 9)