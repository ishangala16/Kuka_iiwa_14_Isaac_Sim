# KUKA iiwa 14 Robot Simulation with Differential IK Control

This project simulates the **KUKA iiwa 14** robotic arm using **NVIDIA Isaac Sim**. The robot is controlled via **Differential Inverse Kinematics (IK)** to reach randomly generated target positions. The simulation includes a gripper, a work table, and a NIST board for a dynamic robotics environment.

## Features
- **Differential IK Control**: Moves the robot's end-effector to target positions.
- **Randomized Targets**: Dynamic target generation within a specified range.
- **Gripper Control**: Opens and closes based on proximity to the target.
- **Visualization**: Red sphere marker for target position.

## Requirements
- NVIDIA Isaac Sim
- Python libraries: `torch`, `numpy`, `scipy`
- USD assets for the robot, table, and NIST board.

  
