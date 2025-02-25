
import argparse
import torch
import numpy as np
from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from omni.isaac.lab.sim import SimulationCfg, SimulationContext
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg

def design_scene():
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    
    # KUKA iiwa 14 configuration
    iiwa_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="disassembly_proj_ws/parts/iiwa14_gripper.usd",
            activate_contact_sensors=True,
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
                "left_inner_knuckle_joint": 0.0,
                "left_inner_finger_joint": 0.0,
                "right_outer_knuckle_joint": 0.0,
                "right_inner_knuckle_joint": 0.0,
                "right_inner_finger_joint": 0.0
                

            },
            pos=(-0.22769, 0.63779, 0.74864),
            rot=(180.0, 0.0, 0.0, 1.0),
        ),
        actuators={
        "iiwa_arm": ImplicitActuatorCfg(
            joint_names_expr=["A[1-7]"],
            effort_limit=300.0,  # Nm (adjust based on your specs)
            velocity_limit=1.5,  # rad/s
            stiffness=400.0,
            damping=40.0,
        ),
        "robotiq_gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint",
                    "left_inner_knuckle_joint",
                    "left_inner_finger_joint",
                    "right_outer_knuckle_joint",
                    "right_inner_knuckle_joint",
                    "right_inner_finger_joint"],
                effort_limit=100.0,
                velocity_limit=2,
                stiffness=800.0,
                damping=80.0,
            )
        }
    )

    robot = Articulation(iiwa_cfg)
    
    cfg_table = sim_utils.UsdFileCfg(usd_path="disassembly_proj_ws/parts/SM_TableWorkSecurity.usd")
    cfg_table.func("/World/Table", cfg_table)

    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=5000.0,
        color=(1.0, 1.0, 1.0),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(0, 0, 5))
    

    return robot


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)


    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    robot = design_scene()

    # Play the simulator
    sim.reset()
    
    # Set initial joint positions
    q_default_arm = [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]  # Neutral pose
    q_default_gripper = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    robot.set_joint_position_target(torch.tensor(q_default_arm + q_default_gripper, device=sim.device))
    robot.write_data_to_sim()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    gripper_speed = 3  # Open/close cycle speed
    max_gripper_angle = 0.8  # From URDF limit

    
    # Simulate physics
    while simulation_app.is_running():
        t = torch.tensor(sim.current_time, device=sim.device) 
        # print("this is t",t)  
        arm_frequencies = torch.tensor([2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8], device=sim.device)
        q_target_arm = 0.5 * torch.sin(t * arm_frequencies)
        q_target_arm[3] += -1.57  # Joint 4 offset
        q_target_arm[5] += 1.57   # Joint 6 offset

        # Gripper control (sinusoidal open/close)
        gripper_pos = max_gripper_angle * (0.5 * torch.sin(t * gripper_speed) + 0.5)

        q_target_gripper = torch.tensor([
            gripper_pos.item(),           # finger_joint
            gripper_pos.item(),           # left_inner_knuckle_joint (mimic 1x)
            -gripper_pos.item(),          # left_inner_finger_joint (mimic -1x)
            gripper_pos.item(),           # right_outer_knuckle_joint (mimic 1x)
            gripper_pos.item(),           # right_inner_knuckle_joint (mimic 1x)
            -gripper_pos.item()           # right_inner_finger_joint (mimic -1x)
        ], device=sim.device)
        
        q_target_full = torch.cat([q_target_arm, q_target_gripper])

        robot.set_joint_position_target(q_target_full)
        robot.write_data_to_sim()
        
        # Step simulation using proper timestep access
        sim.step()
        robot.update(sim.current_time)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
