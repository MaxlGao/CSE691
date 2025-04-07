import pybullet as p
import pybullet_data
import time
import numpy as np

# Start PyBullet in GUI mode
p.connect(p.GUI)

# Load PyBullet's default data (plane, robots, etc.)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load a plane and set gravity
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)


# Load block pieces (these could be custom URDF files)
block1 = p.loadURDF("cube_small.urdf", [0, 0, 0.1], useFixedBase=False)
block2 = p.loadURDF("cube_small.urdf", [0.2, 0, 0.1], useFixedBase=False)
block3 = p.loadURDF("cube_small.urdf", [0.1, 0.2, 0.1], useFixedBase=False)

# Optionally, assign unique constraints to mimic interlocking behavior
p.createConstraint(block1, -1, block2, -1, p.JOINT_FIXED, [0, 0, 0], [0.1, 0, 0], [0.1, 0, 0])


robot1 = p.loadURDF("kuka_iiwa/model.urdf", [-0.5, 0, 0], useFixedBase=True)
robot2 = p.loadURDF("kuka_iiwa/model.urdf", [0.5, 0, 0], useFixedBase=True)


def move_robot(robot, target_position):
    """Move the robot's end-effector to the given position using IK."""
    end_effector_index = 6  # Last joint of KUKA arm
    joint_positions = p.calculateInverseKinematics(robot, end_effector_index, target_position)
    
    # Apply joint positions to the robot
    for joint_idx in range(7):
        p.setJointMotorControl2(robot, joint_idx, p.POSITION_CONTROL, joint_positions[joint_idx])

# Example: Move robot 1 to grab block 1
target_position = [0, 0, 0.2]  # Adjust based on block position
move_robot(robot1, target_position)

for _ in range(10000):
    p.stepSimulation()
    time.sleep(1 / 240.0)  # Keep real-time speed

p.disconnect()
