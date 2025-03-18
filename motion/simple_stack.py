# tbd: current mode: i, o, direct joint offset, i for 6 finger, o for first three joint
#   intend to use hand_off for end effort, which would need rpmflow (ik solver)
#   see target_follow example.

from omni.isaac.core.tasks import BaseTask

from omni.isaac.examples.ur10hand_stack.base_sample import BaseSample
from omni.isaac.ur10hand.controllers.stacking_controller import StackingController
import numpy as np
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.ur10hand import UR10 
from omni.isaac.franka import Franka 
from omni.isaac.core.utils.types import ArticulationAction
import carb
import omni
import weakref
import os, sys
import torch
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.ur10hand.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
print('xxxxx torch ', torch.__version__)

sys.path.append("/home/student/Documents/IsaacLab/source/extensions/omni.isaac.lab")
#from omni.isaac.lab.devices import Se3Keyboard
#from omni.isaac.franka.tasks import Stacking

class UR10Playing(BaseTask):
    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])
        self._task_achieved = False
        return
    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        self._cube = scene.add(DynamicCuboid(prim_path="/World/random_cube",
                                            name="fancy_cube",
                                            position=np.array([0.3, 0.3, 0.3]),
                                            scale=np.array([0.0415, 0.0415, 0.0415]),
                                            color=np.array([0, 0, 1.0])))

        #exts/omni.isaac.universal_robots/omni/isaac/universal_robots/tasks/bin_filling.py
        self._ur10_asset_path = "/home/student/Documents/tang/VisionRobot/isaac_example/ur10_bin_filling2.usd"
        add_reference_to_stage(usd_path=self._ur10_asset_path, prim_path="/World/Scene")
        self._franka = scene.add(UR10(prim_path="/World/Scene/ur10",
                                        name="fancy_franka",
                                        position=np.array([0., 0., 0.5])))
        # scene.add add a robot obj to whatever is @prim_path, it must match the sub prim if the prim is not empty

        return

    def get_observations(self):
        cube_position, _ = self._cube.get_world_pose()
        current_joint_positions = self._franka.get_joint_positions()
        observations = {
            self._franka.name: {
                "joint_positions": current_joint_positions,
            },
            self._cube.name: {
                "position": cube_position,
                "goal_position": self._goal_position
            }
        }
        return observations



class SimpleStack(BaseSample):
    def __init__(self, world=None, exthandle=None) -> None:
        super().__init__()
        self._world = world
        self.exthandle=exthandle # exthandle passed from simple_stack_extension
        self.dbg=False
        self.ik_conti=False
        self._controller = None
        self._articulation_controller = None
        self.finger_off=[0.,0.,0.,0.,0., 0., 0.,0.,0.,0.]
        self.arm_off=[0.,0.,0.,0.,0., 0.]
        self.hand_off=[0.,0.,0.]
        self.root_pose_0=torch.tensor([[0., 0., 0.],[0.,0.,0.]], device='cuda')
        self.pose_flag=False
        self.f_indx=1
        self.handmode='i'
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input= carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args: self._on_keyboard_event(event, *args),
            )

    def setup_scene(self):
        self._world = self.get_world()
        #self._world.add_task(UR10Playing(name="stacking_task"))
        print('robots ', self._world.scene._scene_registry.robots)

        return

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            print('ur10 keyboard event', event.input.name)
            if event.input.name == "KEY_0":
                self.f_indx=0
            if event.input.name == "KEY_1":
                self.f_indx=1
            if event.input.name == "KEY_2":
                self.f_indx=2
            if event.input.name == "KEY_3":
                self.f_indx=3
            if event.input.name == "KEY_4":
                self.f_indx=4
            if event.input.name == "KEY_5":
                self.f_indx=5
            if event.input.name == "KEY_7": #hand orientation rot x
                self.f_indx=7
            if event.input.name == "KEY_8": #hand orientation rot y
                self.f_indx=8
            if event.input.name == "KEY_9":
                #reset all to 0
                for i in range(6):
                    self.finger_off[i]=0.
            if event.input.name == "I":
                self.handmode='i'
            if event.input.name == "O":
                self.handmode='o'
            if event.input.name == "A":
                self.handmode='a'
            if event.input.name == "P":
                self.handmode='p'
            if event.input.name == "D":
                self.dbg=True
            if self.handmode == 'i' and event.input.name == "U":
                self.finger_off[self.f_indx]= 0.05
                self.pose_flag=True
            if self.handmode == 'i' and event.input.name == "J":
                self.finger_off[self.f_indx]= -0.05
                self.pose_flag=True
            if self.handmode == 'a' and event.input.name == "U":
                self.arm_off[self.f_indx]= 0.05
                self.pose_flag=True
            if self.handmode == 'a' and event.input.name == "J":
                self.arm_off[self.f_indx]= -0.05
                self.pose_flag=True
            if self.handmode == 'o' and event.input.name == "U":
                self.hand_off[self.f_indx]= 0.05
                self.pose_flag=True
            if self.handmode == 'o' and event.input.name == "MINUS":
                self.hand_off[self.f_indx]= 0.0
                self.pose_flag=True
            if self.handmode == 'o' and event.input.name == "J":
                self.hand_off[self.f_indx]= -0.05
                self.pose_flag=True
        return True


    def setup_post_load(self):
        print('robots ', self._world.scene._scene_registry.robots)
        self._franka_task = self._world.get_task(name="stacking_task")
        #self._task_params = self._franka_task.get_params()
        print("xxx self.world ", self._world)
        self.my_franka = self._world.scene.get_object("fancy_franka") #object name is not the prim path
        print("xxx self.world ", self.my_franka)
        self._controller = RMPFlowController(name="target_follower_controller", robot_articulation=self.my_franka)

        self._articulation_controller = self.my_franka.get_articulation_controller()
        #self.teleop_interface = Se3Keyboard(pos_sensitivity=0.04, rot_sensitivity=0.08)
        #self.teleop_interface.add_callback("L", reset_sim)

        #callback to start tang stuff after load the scene
        print('vvv exthandle ', self.exthandle)
        #self.exthandle.startup_tang()
        return

    def call_physics_step(self, delta_p, delta_r, step_size):
        self._on_stacking_physics_step(delta_p, step_size)

    def _on_stacking_physics_step(self,delta_p, step_size):
        observations = self._world.get_observations()
        #bservation  {'fancy_franka': {'joint_positions': array([ 18 joints...], dtype=float32)}, 'fancy_cube': {'position': array([x,y,z], dtype=float32), 'goal_position': array([-0.3    , -0.3    ,  0.02575])}}
        #print('observation ', observations)
        ee_pose_w = self.my_franka._end_effector.get_world_pose() #tuple (pose, quat)
        if self.dbg:
            self.dbg=False
            print('xxx eepose ', ee_pose_w[0])

        actions = ArticulationAction(joint_positions=np.array([0.0, -0.5, 0.54, 0. , 0.0, 0 , 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.]))
        actions.joint_positions=observations['fancy_franka']['joint_positions']
        if self.pose_flag and not self.handmode=='o':
            #self.hand_off[self.f_indx]+= 0.05
            self.pose_flag=False
            if self.handmode=='i':#finger
                actions.joint_positions[self.f_indx+6] +=self.finger_off[self.f_indx]
            if self.handmode=='a':#arm joints
                actions.joint_positions[self.f_indx] +=self.arm_off[self.f_indx]
            self._articulation_controller.apply_action(actions)
            print("xxx actions ", actions)

        if self.handmode=='o' and self.pose_flag: #IK, end_effort
            self.pose_flag=False
            self.ik_conti=True
            self.ee_pose_t_p = ee_pose_w[0].copy()
            self.ee_pose_t_q = ee_pose_w[1].copy()
            self.ee_pose_t_p += self.hand_off
            print('xxx hand_off ', self.hand_off)
            print('xxx eepose target ', self.ee_pose_t_p)
            actions_ik = self._controller.forward(target_end_effector_position=self.ee_pose_t_p, target_end_effector_orientation=self.ee_pose_t_q )
            self._articulation_controller.apply_action(actions_ik)
        if self.handmode=='o' and self.ik_conti:
           #IK, ongoing, end_effort
            print('xxx eepose curr/target conti ', ee_pose_w[0], self.ee_pose_t_p)
            print(' pose diff ', np.linalg.norm(ee_pose_w[0]-self.ee_pose_t_p))
            actions_ik = self._controller.forward(target_end_effector_position=self.ee_pose_t_p, target_end_effector_orientation=self.ee_pose_t_q )
            self._articulation_controller.apply_action(actions_ik)
            if np.linalg.norm(ee_pose_w[0]-self.ee_pose_t_p)<0.02:
                print(' pose diff ', np.linalg.norm(ee_pose_w[0]-self.ee_pose_t_p))
                self.ik_conti=False
                print("IK complete!")

        if self.handmode=='p': #IK phone, end_effort
            self.ee_pose_t_p = ee_pose_w[0].copy()
            self.ee_pose_t_q = ee_pose_w[1].copy()
            self.ee_pose_t_p += np.array(delta_p)
            print('www eepose target ', self.ee_pose_t_p)
            actions_ik = self._controller.forward(target_end_effector_position=self.ee_pose_t_p, target_end_effector_orientation=self.ee_pose_t_q )
            self._articulation_controller.apply_action(actions_ik)

        return

    async def _on_stacking_event_async(self):
        world = self.get_world()
        #world.add_physics_callback("sim_step", self._on_stacking_physics_step)
        await world.play_async()
        return

    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
        return

    def world_cleanup(self):
        return
