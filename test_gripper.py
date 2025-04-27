import trimesh
import numpy as np

# Create parts
thickness_tip = 0.25
thickness_finger = 0.5
fingertip = trimesh.creation.cylinder(radius=1, height=thickness_tip, sections=8)
finger = trimesh.creation.box(extents=[4, 2, thickness_finger])
palm = trimesh.creation.box(extents=[1.5, 2, 4])
test = trimesh.creation.box(extents=[4, 1.5, 2])

# Move parts
left_fingertip = fingertip.copy()
left_fingertip.apply_translation([0, 0, 1+thickness_tip/2])
left_finger = finger.copy()
left_finger.apply_translation([-2, 0, 1+thickness_tip+thickness_finger/2])

right_fingertip = fingertip.copy()
right_fingertip.apply_translation([0, 0, -1-thickness_tip/2])
right_finger = finger.copy()
right_finger.apply_translation([-2, 0, -1-thickness_tip-thickness_finger/2])

palm.apply_translation([-4-1.5/2, 0, 0])


gripper = trimesh.util.concatenate([left_finger, right_finger, left_fingertip, right_fingertip, palm])

# Create a scene
scene = trimesh.Scene()
# Add parts
scene.add_geometry(gripper, node_name="gripper")
scene.add_geometry(test, node_name="test")
# Apply a transformation to the whole gripper
T = trimesh.transformations.rotation_matrix(
    angle=np.radians(-45),   # Rotate 45 degrees
    direction=[0, 0, 1],    # Around Z axis
    point=[0, 0, 0]         # Around the origin
)

gripper.apply_transform(T)

# Show it
scene.show()

