import torch
import numpy as np
from omni.isaac.lab.app import AppLauncher
import argparse
from scipy.spatial.transform import Rotation

# Create argparser and launch app
parser = argparse.ArgumentParser(description="KUKA iiwa Reaching Task")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Main simulation setup"""
from omni.isaac.lab.sim import SimulationCfg, SimulationContext
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.controllers import DifferentialIKControllerCfg, DifferentialIKController

def design_scene():
    """Setup the simulation scene with robot and target"""
    # Ground plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/Ground", cfg_ground)

    # KUKA iiwa 14 configuration
    iiwa_cfg = ArticulationCfg(
        prim_path="/World/iiwa",
        spawn=sim_utils.UsdFileCfg(
            usd_path="disassembly_proj_ws/parts/iiwa14_v3.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
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
            },
            pos=(0.0, 0.0, 0.0),
            rot=Rotation.from_euler("xyz", [0, 0, 0], degrees=True).as_quat(),  # Proper quaternion
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["A[1-7]"],
                effort_limit=300.0,
                velocity_limit=1.5,
                stiffness=800.0,
                damping=80.0,
            )
        }
    )
    return Articulation(iiwa_cfg)

def main():
    """Main execution flow"""
    # Initialize simulation
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.5])

    # Add a point light attached to the camera
    cfg_camera_light = sim_utils.DiskLightCfg(
        intensity=5000.0,  # Adjust brightness
        color=(1.0, 1.0, 1.0),  # White light
        radius=5.0,  # Light falloff radius
    )
    cfg_camera_light.func("/World/CameraLight", cfg_camera_light, translation=[2.0, 2.0, 2.0])  # Start near camera

    # Spawn scene elements
    robot = design_scene()
    target_pos = torch.tensor([0.5, 0.3, 1.0], device=sim.device)  # Reach target in XYZ

    # Create target visualization
    cfg_target = sim_utils.SphereCfg(
        radius=0.05,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    cfg_target.func("/World/Target", cfg_target, translation=target_pos.cpu().numpy())

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

    # Simulation loop
    sim.reset()
    tolerance = 0.02  # Target position tolerance (meters)
    
    while simulation_app.is_running():
        # Update robot data
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
        robot.set_joint_position_target(joint_targets.squeeze(0))
        robot.write_data_to_sim()
        
        # Check reaching status
        position_error = torch.norm(ee_pos - target_pos).item()
        if position_error < tolerance:
            print("Target reached!")
            break

        # Step simulation
        sim.step()

    # Final simulation steps to stabilize
    for _ in range(100):
        sim.step()
    
    simulation_app.close()

if __name__ == "__main__":
    main()