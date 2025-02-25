import torch
import numpy as np
from omni.isaac.lab.app import AppLauncher
import argparse
from scipy.spatial.transform import Rotation
import random

parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
from omni.isaac.lab.sim import SimulationCfg, SimulationContext
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.controllers import DifferentialIKControllerCfg, DifferentialIKController

def design_scene():
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    
    # KUKA iiwa 14 configuration
    iiwa_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="disassembly_proj_ws/parts/iiwa14_v3.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "A1": 0.0,
                "A2": 0.0,
                "A3": 0.0,
                "A4": 0.0,
                "A5": 0.0,
                "A6": 0.0,
                "A7": 0.0,
                "finger_joint": 0.0,
            },
            pos=(-0.22769, 0.63779, 0.74864),
            rot=Rotation.from_euler("xyz", [-90, 0, 0], degrees=True).as_quat(),
        ),
        actuators={
            "iiwa_arm": ImplicitActuatorCfg(
                joint_names_expr=["A[1-7]"],
                effort_limit=300.0,
                velocity_limit=1.5,
                stiffness=800.0,
                damping=80.0,
            ),
            "robotiq_gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                effort_limit=1000,
                velocity_limit=8,
                stiffness=1000,
                damping=100,
            )
        }
    )

    robot = Articulation(iiwa_cfg)
    
    cfg_table = sim_utils.UsdFileCfg(usd_path="disassembly_proj_ws/parts/SM_TableWorkSecurity.usd")
    cfg_table.func("/World/Table", cfg_table)

    nist_board_euler = [0, 0, -180]
    nist_board_quat = Rotation.from_euler("xyz", nist_board_euler, degrees=True).as_quat()

    # Reorder to Isaac Sim's expected (w, x, y, z) format
    nist_board_quat = [nist_board_quat[3], nist_board_quat[0], nist_board_quat[1], nist_board_quat[2]]

    print("Quaternion in (w, x, y, z) format:", nist_board_quat)
    cfg_nist_board = sim_utils.UsdFileCfg(
        usd_path="disassembly_proj_ws/parts/NIST_BOARD_v3.usd",
        scale=(0.01, 0.01, 0.01),  # Scale the board by 1.5x in all dimensions
    )
    cfg_nist_board.func("/World/NIST_BOARD_v3", cfg_nist_board,translation=(0.00056, 0.30029, 0.7475),orientation=nist_board_quat)

    # Add distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=5000.0,
        color=(1.0, 1.0, 1.0),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(0, 0, 5))
    
    return robot

def randomize_target_position():
    # Generate random target position within a specified range
    x = random.uniform(-0.5, 0.5)  # Random x within a range
    y = random.uniform(0.0, 0.5)   # Random y within a range
    z = random.uniform(1.0, 1.5)   # Random z within a range
    return torch.tensor([x, y, z], device="cuda")

def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)

    # Add camera and camera light
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    cfg_camera_light = sim_utils.DiskLightCfg(
        intensity=5000.0,
        color=(1.0, 1.0, 1.0),
    )
    cfg_camera_light.func("/World/CameraLight", cfg_camera_light, translation=[2.0, 2.0, 2.0])

    robot = design_scene()
    target_pos = randomize_target_position()

    cfg_target_marker = sim_utils.SphereCfg(
        radius=0.05,  # Size of the marker
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red color
    )
    cfg_target_marker.func("/World/TargetMarker", cfg_target_marker, translation=target_pos.cpu().numpy())

    # Initialize controller
    ik_controller = DifferentialIKController(
        cfg=DifferentialIKControllerCfg(
            command_type="position",  # Use absolute position control
            ik_method="dls",
            ik_params={"lambda_val": 0.01},
        ),
        num_envs=1,
        device=sim.device
    )

    sim.reset()
    tolerance = 0.02  # Tolerance for reaching the target
    gripper_open_pos = 0.0  # Fully open
    gripper_close_pos = 0.8  # Fully closed
    gripper_target_pos = gripper_open_pos  # Start with gripper open
    
    while simulation_app.is_running():
        robot.update(sim.current_time)
        # Get current end-effector state
        ee_idx = robot.body_names.index("link_ee")
        ee_pos = robot.data.body_state_w[0, ee_idx, :3]
        ee_quat = robot.data.body_state_w[0, ee_idx, 3:7]
        
        # Compute IK control
        ik_controller.set_command(
            target_pos.unsqueeze(0),  # Target position
            ee_quat=ee_quat.unsqueeze(0),  # Current end-effector orientation
        )
        joint_targets = ik_controller.compute(
            ee_pos=ee_pos.unsqueeze(0),
            ee_quat=ee_quat.unsqueeze(0),
            jacobian=robot.root_physx_view.get_jacobians()[:, ee_idx, :, :],
            joint_pos=robot.data.joint_pos,
        )
        
        # Apply joint targets
        joint_targets[0, 7] = gripper_target_pos
        robot.set_joint_position_target(joint_targets.squeeze(0))
        # print(joint_targets)
        robot.write_data_to_sim()

        # Compute and print position error
        position_error = torch.norm(ee_pos - target_pos).item()
        if position_error < tolerance:
            print("Target reached!")
            # target_pos = randomize_target_position()
            # cfg_target_marker.func("/World/TargetMarker", cfg_target_marker, translation=target_pos.cpu().numpy())
            gripper_target_pos = gripper_close_pos
            # break

        # Step simulation
        sim.step()

    simulation_app.close()

if __name__ == "__main__":
    main()