
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
        )
        }
    )

    robot = Articulation(iiwa_cfg)
    
    cfg_table = sim_utils.UsdFileCfg(usd_path="disassembly_proj_ws/parts/SM_TableWorkSecurity.usd")
    cfg_table.func("/World/Table", cfg_table)

    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

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
    # Initialize robot
    # robot = Articulation("/World/Robot")
    # robot.initialize()
    
    # Set initial joint positions
    q_default = [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0,0.0,0.0,0.19,0.24,0,0]  # Neutral pose
    robot.set_joint_position_target(torch.tensor(q_default, device=sim.device))
    robot.write_data_to_sim()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    
    # Simulate physics
    while simulation_app.is_running():
        t = torch.tensor(sim.current_time, device=sim.device) 
        # print("this is t",t)  
        frequencies = torch.tensor([2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8], device=sim.device)
        
        # Compute joint targets using tensor operations
        q_target = 0.5 * torch.sin(t * frequencies)
        # Apply offset to middle joints
        q_target[3] += -1.57  # 4th joint (-π/2 offset)
        q_target[5] += 1.57   # 6th joint (+π/2 offset)
        
        robot.set_joint_position_target(q_target)
        robot.write_data_to_sim()
        
        # Step simulation using proper timestep access
        sim.step()
        robot.update(sim.current_time)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
