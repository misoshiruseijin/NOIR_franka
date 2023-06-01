import time

import numpy as np
import math

import sys
sys.path.insert(1, "/home/eeg/deoxys_control/deoxys")

from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to_custom
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig, transform_utils
from deoxys.utils.config_utils import (get_default_controller_config, verify_controller_config)
from deoxys.utils.input_utils import input2action

from deoxys.utils.log_utils import get_deoxys_example_logger
import utils.transformation_utils as U
import json

import redis

logger = get_deoxys_example_logger()

"""
At default reset joint angles:
    eef_ori = array([
        [ 0.99941218,  0.01790704,  0.02890312],
        [ 0.01801811, -0.9998216 , -0.00358691],
        [ 0.02883373,  0.00410558, -0.99957578]
    ])
       
    eef_pos = array([0.45775618, 0.03207872, 0.26534091])
    axis_angle = [3.13730597 0.02814467 0.04520512]

"""
class PrimitiveSkill:

    def __init__(
        self,
        # interface_config="config/charmander.yml",
        controller_type,
        controller_config,
        robot_interface,
        waypoint_height=0.25,
        workspace_limits=None,
        idx2skill=None,
        enable_eeg_interrupt=False,
        use_high_reset_pos=False,
        ):
        """
        Args: 
            interface_config (str) : path to robot interface config yaml file
            controller_type (str) : name of controller. defaults to OSC_POSE
            waypoint_height (float) : height of waypoint used in multi-step skills (pick, place, push)
            workspace_limits (dict) : {"x" : (x lower limit, x upper limit), "y" : (y lower limit, y upper limit), "z" : (z lower limit, z upper limit) }
            yaw_limits (tuple of floats) : (yaw lower limit, yaw upper limit)
            idx2skill (dict) : maps skill index (int) in one-hot skill-selection action vector to skill name (str). If unspecified, uses default settings
        """

        # for now, just don't pass in idx2skill - TODO may need to change if we end up doing EF
        assert idx2skill is None, "Inputting idx2skill is not allowed at the moment"
        self.robot_interface = robot_interface
        self.controller_type = controller_type
        self.controller_config = controller_config

        # robot home position, waypoint height, etc.
        self.from_top_reset_eef_pos = [0.45775618, 0.03207872, 0.35534091]
        self.from_side_reset_eef_pos = [0.45775618, 0.25018698, 0.26756592]
        self.from_top_quat = [0.9998506, 0.00906314, 0.01459545, 0.00192735] # quat when eef is pointing straight down 
        self.from_side_quat = [0.508257, 0.49478495, -0.49082687, 0.5059166] # quat when eef is pointing to right of robot 
        self.from_side_quat2 = [0.6941543, 0.02806479, 0.01957926, 0.7190123] # used for pick from side 2
        self.from_diag_quat = [0.9250823, 0.01997216, 0.01295406, 0.37901983]
        self.horizontal_quat = [0.6939173, 0.7193049, 0.02093083, 0.02532321]

        if use_high_reset_pos:
            from_top = [0.01644747, -0.44936592, -0.0032551, -1.98056433, -0.03537027, 1.52042939, 0.82159965]
            out_of_way_top = [-np.pi/2, -0.69435092, -0.0111128, -2.31718587, -0.00535662, 1.63639478, 0.6249819]
        else:
            from_top = [0.07263956, -0.34306933, -0.01955571, -2.45878116, -0.01170808, 2.18055725, 0.84792026]
            out_of_way_top = [-np.pi/2, -0.34306933, -0.01955571, -2.45878116, -0.01170808, 2.18055725, 0.84792026]
        self.reset_joint_positions = {
            "from_top" : from_top,
            "from_side" : [0.45222925, 0.3912074, 0.41882967, -2.10946937, -0.89842106, 0.98800324, 0.41594728],
            "out_of_way_top" : out_of_way_top,
            "out_of_way_side" : [-np.pi/2, 0.3912074, 0.41882967, -2.10946937, -0.89842106, 0.98800324, 0.41594728],
        }
        self.waypoint_height = waypoint_height # height of waypoint in pick, place, push skills
        if workspace_limits is not None:
            self.workspace_limits = workspace_limits
        else: 
            self.workspace_limits = {
                "x" : (0.3, 0.7),
                "y" : (-0.30, 0.25),
                "z" : (0.0, 0.25)
            }

        # executable skills
        self.skills = {
            "pick_from_top" : self._pick_from_top,
            "pick_from_top2" : self._pick_from_top2,
            "pick_from_side" : self._pick_from_side,
            "pick_book" : self._pick_book,
            "pick_from_side2" : self._pick_from_side2,
            "place_from_top" : self._place_from_top,
            "place_from_side" : self._place_from_side,
            "place_from_diag" : self._place_from_diag,
            "push_x" : self._push_x,
            "push_y" : self._push_y,
            "push_book" : self._push_book,
            "push_z" : self._push_z,
            "wipe_xy" : self._wipe_xy,
            "wipe_y" : self._wipe_y,
            "draw_x" : self._draw_x,
            "screw" : self._screw,
            "pour_from_top" : self._pour_from_top,
            "pour_from_side" : self._pour_from_side,
            "pour" : self._pour,
            "iron" : self._iron,
            "pull_x" : self._pull_x,
            "pull_y" : self._pull_y,
            "pull_up_and_right1" : self._pull_up_and_right1,
            "pull_up_and_left1" : self._pull_up_and_left1,
            "pull_up_and_right2" : self._pull_up_and_right2,
            "pull_up_and_left2" : self._pull_up_and_left2,
            "erase" : self._erase,
            "grate" : self._grate,
            "reset_joints" : self._reset_joints,
        }

        with open('config/skill_config.json') as json_file:
            self.skill_dict = json.load(json_file)
            self.num_skills = len(self.skill_dict.keys())

        # helper skills used internally by executable skills
        self.helper_skills = {
            "move_to" : {
                "num_params" : 9,
                "skill" : self._move_to,
            },
            "gripper_action" : {
                "num_params" : 1,
                "skill" : self._gripper_action,
            },
            "pause" : {
                "num_params" : 2,
                "skill" : self._pause,
            },
            "rehome" : {
                "num_params" : 9,
                "skill" : self._rehome,
            }
        }

        if idx2skill is None:
            self.idx2skill = { skill["default_idx"] : name for name, skill in self.skill_dict.items()}
        else:
            for name in idx2skill.values():
                assert name in self.skills.keys(), f"Error with skill {name}. Skill name must be one of {self.skills.keys()}"
            self.idx2skill = idx2skill    
    
        # self.num_skills = len(self.idx2skill)
        self.max_num_params = max([self.skill_dict[skill_name]["num_params"] for skill_name in self.idx2skill.values()])

        # interruption flag
        self.interrupt = False # this is set to true when the human sends a signal to interrupt a skill during execution
        self.allow_interrupt = True # if human is allowed to interrupt (prevents interruption during rehome)
        self.rehome_pos = self.from_top_reset_eef_pos # position to rehome to for the skill currently being executed
        self.rehome_quat = self.from_top_quat # quat to rehome to for the skill currently being executed
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], -1.0)

        # TODO remove this after testing
        # self.r = redis.Redis()
        # self.INTERRUPT_KEY = "interrupt"
        # self.r.set(self.INTERRUPT_KEY, "False")

        self.enable_eeg_interrupt = enable_eeg_interrupt # whether to enable interrupt using signal from eeg server

    def execute_skill(self, action):
        """
        Executes skill. This is the main function that should be called from outside this class to execute a skill
        Args:
            action : skill selection vector concatenated with params vector
        """
        # get the skill to execute
        skill_idx = np.argmax(action[:self.num_skills])
        # skill = self.skills[self.idx2skill[skill_idx]]["skill"] 
        skill = self.skills[self.idx2skill[skill_idx]]
        skill_name = self.idx2skill[skill_idx]

        # extract params and execute
        params = action[self.num_skills:]
        print(f"Executing skill {skill_name} with params {params}")

        skill(params)
        return skill_name

    def _execute_sequence(self, sequence):
        """
        Executes a sequence of skills. Used internally in each skill.

        Args:
            sequence (list of lists) : [ [skill name 1, param 1], [skill name 2, param 2], ...]
        """
        # make sure the interrupt signal received before skill execution begins is ignored
        # self.r.set(self.INTERRUPT_KEY, "False") # TODO remove or replace 
        print("======= starting skill execution ======")
        for item in sequence:
            # TODO - test this            
            # if human interrupts, stop execution and rehome
            if self.interrupt:
                self.interrupt = False
                # self.r.set(self.INTERRUPT_KEY, "False") # TODO remove or replace
                gripper_action = self._get_gripper_state() # get the current gripper state
                print("-------- Human Interrupt. Rehoming -----------")
                # self._rehome(np.concatenate([self.rehome_pos, self.rehome_quat, [gripper_action, 1]]))
                self._rehome(params=np.append(self.rehome_q[:7], gripper_action))
                print("-------- Finished Rehoming -----------")
                self.allow_interrupt = True
                break
                
            skill_name, params = item
            skill = self.helper_skills[skill_name]["skill"]
            skill(params=params)


    ########################################################################################
    ################################## EXECUTABLE SKILLS ###################################
    ########################################################################################
    
    def _pick_from_top(self, params):
        """
        Picks up object at specified position from top and rehomes

        Args:
            params (3-tuple of floats) : [goal_pos]
        """
        # define waypoints
        goal_pos = params[:3]
        waypoint = np.array([params[0], params[1], self.waypoint_height])
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], 1.0) 
         
        sequence = [
            [ "move_to", np.concatenate([waypoint, self.from_top_quat, [-1, 1]]) ], # to waypoint
            [ "move_to", np.concatenate([goal_pos, self.from_top_quat, [-1, 1]]) ], # to pick pos
            [ "gripper_action", [1] ], # close gripper
            [ "move_to", np.concatenate([waypoint, self.from_top_quat, [1, 0]]) ], # to waypoint
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],
        ]
        self._execute_sequence(sequence)

    def _pick_from_top2(self, params):
        """
        Picks up object at specified position from top and rehomes

        Args:
            params (3-tuple of floats) : [goal_pos]
        """
        # define waypoints
        goal_pos = params[:3]
        waypoint = np.array([params[0], params[1], self.waypoint_height])
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], 1.0) 
         
        sequence = [
            [ "move_to", np.concatenate([waypoint, self.horizontal_quat, [-1, 1]]) ], # to waypoint
            [ "move_to", np.concatenate([goal_pos, self.horizontal_quat, [-1, 1]]) ], # to pick pos
            [ "gripper_action", [1] ], # close gripper
            [ "move_to", np.concatenate([waypoint, self.horizontal_quat, [1, 0]]) ], # to waypoint
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],
        ]
        self._execute_sequence(sequence)

    def _pick_from_side(self, params):
        """
        Picks up object at specified position from side and rehomes

        Args:
            params (3-tuple of floats) : [goal_pos]
        """
        # define waypoints
        goal_pos = params[:3]
        waypoint_side = np.array([params[0], params[1] + 0.2, params[2]])
        waypoint_above = np.array([params[0], params[1], self.waypoint_height])
        self.rehome_q = np.append(self.reset_joint_positions["from_side"], 1.0)

        sequence = [
            [ "move_to", np.concatenate([waypoint_side, self.from_side_quat, [-1, 1]]) ], # to waypoint
            [ "move_to", np.concatenate([goal_pos, self.from_side_quat, [-1, 1]]) ], # to pick pos
            [ "gripper_action", [1] ], # close gripper
            [ "move_to", np.concatenate([waypoint_above, self.from_side_quat, [1, 0]]) ], # to waypoint
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],
        ]
        self._execute_sequence(sequence)

    def _pick_book(self, params):
        """
        Picks up object at specified position from side and rehomes

        Args:
            params (2-tuple of floats) : [goal_pos(xy)]
        """
        # define waypoints
        goal_pos = params[:2]
        goal_pos = [params[0], params[1], 0.17]
        waypoint_side = np.array([goal_pos[0], goal_pos[1] + 0.25, goal_pos[2]])
        waypoint_above = np.array([params[0], params[1], self.waypoint_height])
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], 1.0)

        sequence = [
            [ "move_to", np.concatenate([waypoint_side, self.from_side_quat2, [-1, 1]]) ], # to waypoint
            [ "move_to", np.concatenate([goal_pos, self.from_side_quat2, [-1, 1]]) ], # to pick pos
            [ "gripper_action", [1] ], # close gripper
            [ "move_to", np.concatenate([waypoint_above, self.from_side_quat2, [1, 0]]) ], # to waypoint
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],
        ]
        self._execute_sequence(sequence)

    def _pick_from_side2(self, params):
        """
        Picks up object at specified position from side and rehomes

        Args:
            params (3-tuple of floats) : [goal_pos]
        """
        # define waypoints
        goal_pos = params[:3]
        waypoint_side = np.array([goal_pos[0], goal_pos[1] + 0.25, goal_pos[2]])
        waypoint_above = np.array([params[0], params[1], self.waypoint_height])
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], 1.0)

        sequence = [
            [ "move_to", np.concatenate([waypoint_side, self.from_side_quat2, [-1, 1]]) ], # to waypoint
            [ "move_to", np.concatenate([goal_pos, self.from_side_quat2, [-1, 1]]) ], # to pick pos
            [ "gripper_action", [1] ], # close gripper
            [ "move_to", np.concatenate([waypoint_above, self.from_side_quat2, [1, 0]]) ], # to waypoint
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],
        ]
        self._execute_sequence(sequence)

    def _place_from_top(self, params):
        """
        Places object at specified location with gripper pointing down

        Args:
            params (3-tuple of floats) : [goal_pos]
        """
        # define waypoints
        goal_pos = params[:3]
        waypoint = np.array([params[0], params[1], self.waypoint_height])
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], -1.0) 

        sequence = [
            [ "move_to", np.concatenate([waypoint, self.from_top_quat, [1, 1]]) ], # to waypoint
            [ "move_to", np.concatenate([goal_pos, self.from_top_quat, [1, 1]]) ], # to place pos
            [ "gripper_action", [-1] ], # release gripper
            [ "move_to", np.concatenate([waypoint, self.from_top_quat, [-1, 0]]) ], # to waypoint
            [ "pause", np.array([-1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],
        ]
        self._execute_sequence(sequence)

    def _place_from_side(self, params):
        """
        Places object at specified location with gripper pointing down

        Args:
            params (3-tuple of floats) : [goal_pos]
        """
        # define waypoints
        goal_pos = params[:3]
        waypoint = np.array([params[0], params[1], self.waypoint_height])
        self.rehome_q = np.append(self.reset_joint_positions["from_side"], -1.0) 

        sequence = [
            [ "move_to", np.concatenate([waypoint, self.from_side_quat, [1, 1]]) ], # to waypoint
            [ "move_to", np.concatenate([goal_pos, self.from_side_quat, [1, 1]]) ], # to place pos
            [ "gripper_action", [-1] ], # release gripper
            [ "pause", np.array([-1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],
        ]
        self._execute_sequence(sequence)
    
    def _place_from_diag(self, params):
        """
        Places object at specified location with gripper orientation 3

        Args:
            params (3-tuple of floats) : [goal_pos]
        """
        # define waypoints
        goal_pos = params[:3]
        waypoint = np.array([params[0], params[1], self.waypoint_height])
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], -1.0) 
        sequence = [
            [ "move_to", np.concatenate([waypoint, self.from_diag_quat, [1, 1]]) ], # to waypoint
            [ "move_to", np.concatenate([goal_pos, self.from_diag_quat, [1, 1]]) ], # to place pos
            [ "gripper_action", [-1] ], # release gripper
            [ "move_to", np.concatenate([waypoint, self.from_top_quat, [-1, 0]]) ], # to waypoint
            [ "pause", np.array([-1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],
        ]
        self._execute_sequence(sequence)
    
    def _push_z(self, params):
        """
        Start from specified position with gripper pointing down, pushes in z direction until end_z, then rehomes

        Args: 
            params (3-tuple of floats) : [start_pos(xy), end_z]
        """

        start_pos = [params[0], params[1], self.waypoint_height]
        gripper_action = 1 # gripper is closed

        goal_pos = [start_pos[0], start_pos[1], params[2]]
        waypoint_above = [start_pos[0], start_pos[1], self.waypoint_height]
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], 1.0) 

        sequence = [
            [ "move_to", np.concatenate([start_pos, self.from_top_quat, [gripper_action, 1]]) ], # to start pos
            [ "move_to", np.concatenate([goal_pos, self.from_top_quat, [gripper_action, 1]]) ], # move in z by dz
            [ "pause", [gripper_action, 1.0] ], # pause for 1 sec
            [ "move_to", np.concatenate([waypoint_above, self.from_top_quat, [gripper_action, 0]]) ], # to waypoint
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],
        ]
        self._execute_sequence(sequence)

    def _push_xy(self, params):
        """
        Start from specified position with gripper pointing down, pushes in x and y direction by specified delta, then rehomes

        Args: 
            params (6-tuple of floats) : [start_pos, dx, dy, yaw_angle[deg]]
        """
        start_pos = params[:3]
        dx = params[3]
        dy = params[4]
        yaw = params[5]
        gripper_action = 1 # gripper is closed

        goal_pos = [start_pos[0] + dx, start_pos[1] + dy, start_pos[2]]
        waypoint_above = [start_pos[0], start_pos[1], self.waypoint_height]
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], 1.0) 

        # convert euler to quat
        from_top_euler = U.quat2euler(self.from_top_quat)
        goal_euler = np.array([from_top_euler[0], from_top_euler[1], np.radians(yaw)]) # update yaw component
        goal_quat = U.euler2quat(goal_euler)

        sequence = [
            [ "move_to", np.concatenate([start_pos, goal_quat, [gripper_action, 1]]) ], # to start pos
            [ "move_to", np.concatenate([goal_pos, goal_quat, [gripper_action, 0]]) ], # move by delta
            [ "move_to", np.concatenate([waypoint_above, goal_quat, [gripper_action, 0]]) ], # to waypoint
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],        
        ]
        self._execute_sequence(sequence)

    def _push_x(self, params):
        """
        Start from specified position with gripper pointing down, pushes in x direction to the end of the workspace, then rehomes

        Args: 
            params (6-tuple of floats) : [start_pos]
        """
        start_pos = params[:3]
        gripper_action = 1 # gripper is closed

        goal_pos = [0.75, start_pos[1], start_pos[2]]
        # goal_pos = [self.workspace_limits["x"][1]+0.05, start_pos[1], start_pos[2]]
        waypoint_above = [start_pos[0], start_pos[1], self.waypoint_height]
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], 1.0) 

        sequence = [
            [ "gripper_action", [1.] ], # close gripper
            [ "move_to", np.concatenate([start_pos, self.from_top_quat, [gripper_action, 1]]) ], # to start pos
            [ "move_to", np.concatenate([goal_pos, self.from_top_quat, [gripper_action, 0]]) ], # move by delta
            [ "move_to", np.concatenate([waypoint_above, self.from_top_quat, [gripper_action, 0]]) ], # to waypoint
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],        
        ]
        self._execute_sequence(sequence)
    
    def _pull_x(self, params):
        """
        Start from specified position with gripper pointing down, pushes in x direction to the end of the workspace, then rehomes

        Args: 
            params (6-tuple of floats) : [start_pos]
        """
        start_pos = params[:3]
        gripper_action = 1 # gripper is closed

        goal_pos = [self.workspace_limits["x"][0]-0.1, start_pos[1], start_pos[2]]
        waypoint_above = [start_pos[0], start_pos[1], self.waypoint_height]
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], 1.0) 

        sequence = [
            [ "gripper_action", [1.] ], # close gripper
            [ "move_to", np.concatenate([start_pos, self.from_top_quat, [gripper_action, 1]]) ], # to start pos
            [ "move_to", np.concatenate([goal_pos, self.from_top_quat, [gripper_action, 0]]) ], # move by delta
            [ "move_to", np.concatenate([waypoint_above, self.from_top_quat, [gripper_action, 0]]) ], # to waypoint
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],        
        ]
        self._execute_sequence(sequence)

    def _pull_y(self, params):
        """
        Start from specified position with gripper pointing down, pushes in x direction to the end of the workspace, then rehomes

        Args: 
            params (6-tuple of floats) : [start_pos]
        """
        start_pos = params[:3]
        gripper_action = 1 # gripper is closed

        goal_pos = [start_pos[0], start_pos[1]-0.35, start_pos[2]]
        waypoint_above = [start_pos[0], start_pos[1], self.waypoint_height]
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], 1.0) 

        sequence = [
            [ "gripper_action", [1.] ], # close gripper
            [ "move_to", np.concatenate([start_pos, self.horizontal_quat, [gripper_action, 1]]) ], # to start pos
            [ "move_to", np.concatenate([goal_pos, self.horizontal_quat, [gripper_action, 0]]) ], # move by delta
            [ "move_to", np.concatenate([waypoint_above, self.horizontal_quat, [gripper_action, 0]]) ], # to waypoint
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],        
        ]
        self._execute_sequence(sequence)

    def _pull_up_and_left1(self, params):
        # define waypoints
        goal_pos = params[:3]
        waypoint = np.array([params[0], params[1], self.waypoint_height])
        waypoint2 = np.array([params[0], params[1]-0.25, 0.35])
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], -1.0) 
         
        sequence = [
            [ "move_to", np.concatenate([waypoint, self.from_top_quat, [-1, 1]]) ], # to waypoint
            [ "move_to", np.concatenate([goal_pos, self.from_top_quat, [-1, 1]]) ], # to pick pos
            [ "gripper_action", [1] ], # close gripper
            [ "move_to", np.concatenate([waypoint2, self.from_top_quat, [1, 0]]) ], # pull
            [ "gripper_action", [-1]], # release gripper
            [ "pause", np.array([-1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],
        ]
        self._execute_sequence(sequence)

    def _pull_up_and_right1(self, params):
        # define waypoints
        goal_pos = params[:3]
        waypoint = np.array([params[0], params[1], self.waypoint_height])
        waypoint2 = np.array([params[0], params[1]+0.25, 0.35])
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], -1.0) 
         
        sequence = [
            [ "move_to", np.concatenate([waypoint, self.from_top_quat, [-1, 1]]) ], # to waypoint
            [ "move_to", np.concatenate([goal_pos, self.from_top_quat, [-1, 1]]) ], # to pick pos
            [ "gripper_action", [1] ], # close gripper
            [ "move_to", np.concatenate([waypoint2, self.from_top_quat, [1, 0]]) ], # pull
            [ "gripper_action", [-1]], # release gripper
            [ "pause", np.array([-1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],
        ]
        self._execute_sequence(sequence)

    def _pull_up_and_left2(self, params):
        # define waypoints
        goal_pos = params[:3]
        waypoint = np.array([params[0], params[1], self.waypoint_height])
        waypoint2 = np.array([params[0], params[1]-0.25, 0.35])
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], -1.0) 
         
        sequence = [
            [ "move_to", np.concatenate([waypoint, self.horizontal_quat, [-1, 1]]) ], # to waypoint
            [ "move_to", np.concatenate([goal_pos, self.horizontal_quat, [-1, 1]]) ], # to pick pos
            [ "gripper_action", [1] ], # close gripper
            [ "move_to", np.concatenate([waypoint2, self.horizontal_quat, [1, 0]]) ], # pull
            [ "gripper_action", [-1]], # release gripper
            [ "pause", np.array([-1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],
        ]
        self._execute_sequence(sequence)

    def _pull_up_and_right2(self, params):
        # define waypoints
        goal_pos = params[:3]
        waypoint = np.array([params[0], params[1], self.waypoint_height])
        waypoint2 = np.array([params[0], params[1]+0.25, 0.35])
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], -1.0) 
         
        sequence = [
            [ "move_to", np.concatenate([waypoint, self.horizontal_quat, [-1, 1]]) ], # to waypoint
            [ "move_to", np.concatenate([goal_pos, self.horizontal_quat, [-1, 1]]) ], # to pick pos
            [ "gripper_action", [1] ], # close gripper
            [ "move_to", np.concatenate([waypoint2, self.horizontal_quat, [1, 0]]) ], # pull
            [ "gripper_action", [-1]], # release gripper
            [ "pause", np.array([-1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],
        ]
        self._execute_sequence(sequence)


    def _push_book(self, params):
        """
        Start from specified position with gripper pointing down, pushes in y direction by 25cm, then rehomes

        Args: 
            params (2-tuple of floats) : [start_pos(xy)]
        """
        start_pos = params[:2]
        start_pos = [start_pos[0], start_pos[1], 0.175]
        gripper_action = 1 # gripper is closed


        goal_pos = [start_pos[0], start_pos[1]+0.25, start_pos[2]]
        waypoint_above = [start_pos[0], start_pos[1], self.waypoint_height]
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], 1.0) 

        sequence = [
            [ "gripper_action", [1.] ], # close gripper
            [ "move_to", np.concatenate([start_pos, self.from_top_quat, [gripper_action, 1]]) ], # to start pos
            [ "move_to", np.concatenate([goal_pos, self.from_top_quat, [gripper_action, 0]]) ], # move in y
            [ "move_to", np.concatenate([waypoint_above, self.from_top_quat, [gripper_action, 0]]) ], # to waypoint
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],        
        ]
        self._execute_sequence(sequence)

    def _push_y(self, params):
        """
        Start from specified position with gripper pointing down, pushes in y direction by 25cm, then rehomes

        Args: 
            params (3-tuple of floats) : [start_pos(xy)]
        """
        start_pos = params[:3]
        gripper_action = 1 # gripper is closed


        goal_pos = [start_pos[0], start_pos[1]+0.35, start_pos[2]]
        waypoint_above = [start_pos[0], start_pos[1], self.waypoint_height]
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], 1.0) 

        sequence = [
            [ "gripper_action", [1.] ], # close gripper
            [ "move_to", np.concatenate([start_pos, self.from_top_quat, [gripper_action, 1]]) ], # to start pos
            [ "move_to", np.concatenate([goal_pos, self.from_top_quat, [gripper_action, 0]]) ], # move in y
            [ "move_to", np.concatenate([waypoint_above, self.from_top_quat, [gripper_action, 0]]) ], # to waypoint
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],        
        ]
        self._execute_sequence(sequence)

    def _erase(self, params):
        """
        Start from specified position with gripper pointing down, pushes in x direction to the end of the workspace, then rehomes

        Args: 
            params (6-tuple of floats) : [start_pos]
        """
        x, y, z =  params[:3]
        delta_y = 0.03
        end_x = 0.75
        start_pos1 = np.array([x, y, z])
        start_pos2 = np.array([x, y+delta_y, z])
        start_pos3 = np.array([x, y+2*delta_y, z])
        end_pos1 = np.array([end_x, y, z])
        end_pos2 = np.array([end_x, y+delta_y, z])
        end_pos3 = np.array([end_x, y+2*delta_y, z])
        wp_h = np.array([0,0,0.1])
        gripper_action = 1 # gripper is closed
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], 1.0) 
        sequence = [
            [ "gripper_action", [1.] ], # close gripper
            [ "move_to", np.concatenate([start_pos1+wp_h, self.from_top_quat, [gripper_action, 1]]) ], # to start pos
            [ "move_to", np.concatenate([start_pos1, self.from_top_quat, [gripper_action, 1]]) ], # to start pos
            [ "move_to", np.concatenate([end_pos1, self.from_top_quat, [gripper_action, 0]]) ], # move by delta
            [ "move_to", np.concatenate([end_pos1+wp_h, self.from_top_quat, [gripper_action, 0]]) ], 
            [ "move_to", np.concatenate([start_pos2+wp_h, self.from_top_quat, [gripper_action, 1]]) ], # to start pos
            [ "move_to", np.concatenate([start_pos2, self.from_top_quat, [gripper_action, 1]]) ], # to start pos
            [ "move_to", np.concatenate([end_pos2, self.from_top_quat, [gripper_action, 0]]) ], # move by delta
            [ "move_to", np.concatenate([end_pos2+wp_h, self.from_top_quat, [gripper_action, 0]]) ], 
            [ "move_to", np.concatenate([start_pos3+wp_h, self.from_top_quat, [gripper_action, 1]]) ], # to start pos
            [ "move_to", np.concatenate([start_pos3, self.from_top_quat, [gripper_action, 1]]) ], # to start pos
            [ "move_to", np.concatenate([end_pos3, self.from_top_quat, [gripper_action, 0]]) ], # move by delta
            [ "move_to", np.concatenate([end_pos3+np.array([0,0,0.1]), self.from_top_quat, [gripper_action, 0]]) ], 
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],        
        ]
        self._execute_sequence(sequence)

    def _wipe_xy(self, params):
        """
        Wipes a surface by starting at specified position, moving on the xy plane with specified dx, dy, returns to start position, then rehomes
        
        Args:
            params (6-tuple of floats) : [start_pos, dx, dy, yaw_angle[deg]]
        """
        start_pos = params[:3]
        dx = params[3]
        dy = params[4]
        yaw = params[5]
        gripper_action = 1 # gripper is closed

        end_pos = [start_pos[0] + dx, start_pos[1] + dy, start_pos[2]]
        waypoint_above = [start_pos[0], start_pos[1], self.waypoint_height]
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], 1.0) 

        # convert euler to quat
        from_top_euler = U.quat2euler(self.from_top_quat)
        goal_euler = np.array([from_top_euler[0], from_top_euler[1], np.radians(yaw)]) # update yaw component
        goal_quat = U.euler2quat(goal_euler)

        sequence = [
            [ "move_to", np.concatenate([waypoint_above, goal_quat, [gripper_action, 0]]) ], # to waypoint
            [ "move_to", np.concatenate([start_pos, goal_quat, [gripper_action, 0]]) ], # to start pos
            [ "move_to", np.concatenate([end_pos, goal_quat, [gripper_action, 1]]) ], # move by delta
            [ "move_to", np.concatenate([start_pos, goal_quat, [gripper_action, 1]]) ], # back to start pos
            [ "move_to", np.concatenate([waypoint_above, goal_quat, [gripper_action, 0]]) ], # to waypoint
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ], 
        ]
        self._execute_sequence(sequence)

    def _wipe_y(self, params):
        """
        Wipes a surface by starting at specified position, moving in y direction by a fixed amount in a wiping motion, then rehomes
        
        Args:
            params (3-tuple of floats) : [start_pos]
        """
        start_pos = params[:3]
        gripper_action = 1 # gripper is closed\
        dy = 0.4

        end_pos = [start_pos[0], start_pos[1] + dy, start_pos[2]]
        waypoint_above = [start_pos[0], start_pos[1], self.waypoint_height]
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], 1.0) 

        sequence = [
            [ "move_to", np.concatenate([waypoint_above, self.from_top_quat, [gripper_action, 0]]) ], # to waypoint
            [ "move_to", np.concatenate([start_pos, self.from_top_quat, [gripper_action, 0]]) ], # to start pos
            [ "move_to", np.concatenate([end_pos, self.from_top_quat, [gripper_action, 1]]) ], # move by delta
            [ "move_to", np.concatenate([start_pos, self.from_top_quat, [gripper_action, 1]]) ], # back to start pos
            [ "move_to", np.concatenate([waypoint_above, self.from_top_quat, [gripper_action, 0]]) ], # to waypoint
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ], 
        ]
        self._execute_sequence(sequence)

    def _grate(self, params):
        """
        Wipes a surface by starting at specified position, moving in y direction by a fixed amount in a wiping motion, then rehomes
        
        Args:
            params (3-tuple of floats) : [start_pos]
        """
        start_pos = params[:3]
        gripper_action = 1 # gripper is closed\
        dy = 0.1

        end_pos = [start_pos[0], start_pos[1] + dy, start_pos[2]]
        wp1 = [start_pos[0], start_pos[1], 0.2]
        wp2 = [end_pos[0], end_pos[1], 0.2]
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], 1.0) 


        n_scrapes = 5
        sequence = []

        for _ in range(n_scrapes):
            sequence.append([ "move_to", np.concatenate([wp1, self.from_top_quat, [gripper_action, 0]]) ]), # to above start position
            sequence.append([ "move_to", np.concatenate([start_pos, self.from_top_quat, [gripper_action, 1]]) ])
            sequence.append([ "move_to", np.concatenate([end_pos, self.from_top_quat, [gripper_action, 1]]) ])
            sequence.append([ "move_to", np.concatenate([wp2, self.from_top_quat, [gripper_action, 1]]) ])

        sequence.append([ "pause", np.array([1.0, 0.5]) ])
        sequence.append([ "rehome", self.rehome_q ])
        self._execute_sequence(sequence)

    def _draw_x(self, params):
        """
        Draws X centered at specified 3d position

        Args:
            params (3-tuple of floats) : [goal_pos]
        """
        # NOTE - z = 0.11 works for marker
        # TODO - find a way to make z fixed (physical mount on marker??)
        line_len = 0.1
        center_pos = params[:3] # center position in xyz
        l1_start = center_pos + np.array([-0.5*line_len*math.sin(math.pi/4), -0.5*line_len*math.cos(math.pi/4), 0])
        l1_wp1 = [l1_start[0], l1_start[1], self.waypoint_height]
        l1_end = center_pos + np.array([0.65*line_len*math.sin(math.pi/4), 0.65*line_len*math.cos(math.pi/4), 0])
        l1_wp2 = [l1_end[0], l1_end[1], self.waypoint_height]
        l2_start = center_pos + np.array([-0.5*line_len*math.sin(math.pi/4), 0.5*line_len*math.cos(math.pi/4), 0])
        l2_wp1 = [l2_start[0], l2_start[1], self.waypoint_height]
        l2_end = center_pos + np.array([0.65*line_len*math.sin(math.pi/4), -0.65*line_len*math.cos(math.pi/4), 0])
        l2_wp2 = [l2_end[0], l2_end[1], self.waypoint_height]
        goal_quat = self.from_top_quat
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], 1.0) 

        sequence = [
            [ "move_to", np.concatenate([l1_wp1, goal_quat, [1, 1]]) ], # to line1 waypoint1
            [ "move_to", np.concatenate([l1_start, goal_quat, [1, 1]]) ], # to line1 start pos
            [ "move_to", np.concatenate([l1_end, goal_quat, [1, 0]]) ], # draw line1
            [ "move_to", np.concatenate([l1_wp2, goal_quat, [1, 0]]) ], # to line1 waypoint2
            [ "move_to", np.concatenate([l2_wp1, goal_quat, [1, 1]]) ], # to line2 waypoint1
            [ "move_to", np.concatenate([l2_start, goal_quat, [1, 1]]) ], # line2 start pos
            [ "move_to", np.concatenate([l2_end, goal_quat, [1, 0]]) ], # draw line2
            [ "move_to", np.concatenate([l2_wp2, goal_quat, [1, 0]]) ], # to line2 waypoint2
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ], 
        ]
        self._execute_sequence(sequence)

    def _screw(self, params):
        """
        Grab a cap at specified position and screw it
        
        Args:
            params (4-tuple of floats) : [grasp_pos, screw angle[deg]]
        """
        # define waypoints
        grasp_pos = params[:3]
        waypoint = np.array([params[0], params[1], self.waypoint_height])
        angle = params[3]

        # get goal quat
        from_top_euler = U.quat2euler(self.from_top_quat)
        goal_euler = np.array([from_top_euler[0], from_top_euler[1], np.radians(angle)]) # update yaw component
        goal_quat = U.euler2quat(goal_euler)
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], -1.0) 

        sequence = [
            [ "move_to", np.concatenate([waypoint, self.from_top_quat, [-1, 1]]) ], # to waypoint
            [ "move_to", np.concatenate([grasp_pos, self.from_top_quat, [-1, 1]]) ], # to grasp pos
            [ "gripper_action", [1] ], # close gripper
            [ "move_to", np.concatenate([grasp_pos, goal_quat, [1, 1]]) ], # screw
            [ "gripper_action", [-1] ], # release gripper
            [ "move_to", np.concatenate([waypoint, self.from_top_quat, [-1, 0]]) ], # to waypoint
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ], 
        ]
        self._execute_sequence(sequence)
    
    def _pour(self, params):
        """
        Pouring motion. Tilts end effector by fixed amount while maintaining end effector position at specified location.
        Assumes robot is holding a relevant object from the top

        Args:
            params (3-tuple of floats) : [eef_pos] 
        """
        # define waypoints
        pour_pos = params[:3]
        waypoint = np.array([params[0], params[1], self.waypoint_height])

        # goal quat
        # goal_quat = [-0.47400907, -0.00888116, 0.0487085, 0.87912697]
        goal_quat = [-0.26835766,  0.0294289,   0.01814095,  0.9626988]
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], 1.0) 

        sequence = [
            [ "move_to", np.concatenate([waypoint, self.from_top_quat, [1, 0]]) ], # to waypoint
            [ "move_to", np.concatenate([pour_pos, self.from_top_quat, [1, 1]]) ], # to grasp pos
            [ "move_to", np.concatenate([pour_pos, goal_quat, [1, 1]]) ], # tilt
            [ "pause", np.array([1.0, 3.0]) ], # pause to let content out
            [ "move_to", np.concatenate([pour_pos, self.from_top_quat, [1, 0]]) ], # rotate back
            [ "move_to", np.concatenate([waypoint, self.from_top_quat, [1, 0]]) ], # to waypoint
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ], 
        ]
        self._execute_sequence(sequence)        

    def _pour_from_top(self, params):
        """
        Pouring motion. Tilts end effector by fixed amount while maintaining end effector position at specified location.
        Assumes robot is holding a relevant object from the top

        Args:
            params (3-tuple of floats) : [eef_pos] 
        """
        # define waypoints
        pour_pos = params[:3]
        waypoint = np.array([params[0], params[1], self.waypoint_height])

        # goal quat
        goal_quat = [0.8490147, 0.01116254, -0.5269386, 0.0372193]
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], 1.0) 

        sequence = [
            [ "move_to", np.concatenate([waypoint, self.from_top_quat, [1, 0]]) ], # to waypoint
            [ "move_to", np.concatenate([pour_pos, self.from_top_quat, [1, 1]]) ], # to grasp pos
            [ "move_to", np.concatenate([pour_pos, goal_quat, [1, 1]]) ], # tilt
            [ "pause", np.array([1.0, 2.0]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "move_to", np.concatenate([pour_pos, self.from_top_quat, [1, 0]]) ], # rotate back
            [ "move_to", np.concatenate([waypoint, self.from_top_quat, [1, 0]]) ], # to waypoint
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ], 
        ]
        self._execute_sequence(sequence)

    def _pour_from_side(self, params):
        """
        Pouring motion. Tilts end effector by fixed amount while maintaining end effector position at specified location.
        Assumes robot is holding a relevant object from the side

        Args:
            params (3-tuple of floats) : [eef_pos] 
        """
        # define waypoints
        pour_pos = params[:3]
        waypoint = np.array([params[0], params[1], self.waypoint_height])

        # goal quat
        goal_quat = [0.6963252, 0.7174281, 0.01201747, 0.01685036]
        self.rehome_q = np.append(self.reset_joint_positions["from_side"], 1.0) 

        sequence = [
            [ "move_to", np.concatenate([waypoint, self.from_side_quat, [1, 0]]) ], # to waypoint
            [ "move_to", np.concatenate([pour_pos, self.from_side_quat, [1, 1]]) ], # to grasp pos
            [ "move_to", np.concatenate([pour_pos, goal_quat, [1, 1]]) ], # tilt
            [ "move_to", np.concatenate([pour_pos, self.from_side_quat, [1, 0]]) ], # rotate back
            # [ "move_to", np.concatenate([waypoint, self.from_side_quat, [1, 0]]) ], # to waypoint
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ],
        ]

        self._execute_sequence(sequence)

    def _iron(self, params):
        """
        Same as wipe_y but with fixed z for ironing task

        Args:
            params (2-tuple of floats) : [start_pos(xy)]
        """
        start_pos = [params[0], params[1], 0.09]
        gripper_action = 1 # gripper is closed\
        dy = 0.3

        end_pos = [start_pos[0], start_pos[1] + dy, start_pos[2]]
        waypoint_above = [start_pos[0], start_pos[1], self.waypoint_height]
        self.rehome_q = np.append(self.reset_joint_positions["from_top"], 1.0) 

        sequence = [
            [ "move_to", np.concatenate([waypoint_above, self.from_top_quat, [gripper_action, 0]]) ], # to waypoint
            [ "move_to", np.concatenate([start_pos, self.from_top_quat, [gripper_action, 0]]) ], # to start pos
            [ "move_to", np.concatenate([end_pos, self.from_top_quat, [gripper_action, 1]]) ], # move by delta
            [ "move_to", np.concatenate([start_pos, self.from_top_quat, [gripper_action, 1]]) ], # back to start pos
            [ "move_to", np.concatenate([waypoint_above, self.from_top_quat, [gripper_action, 0]]) ], # to waypoint
            [ "pause", np.array([1.0, 0.5]) ], # add short pause to prevent sudden stop from swithing controllers
            [ "rehome", self.rehome_q ], 
        ]
        self._execute_sequence(sequence)

    def _reset_joints(self, params=None):
        """
        Resets joints to fixed position.

        Args:
            params (8-tuple of floats) : [reset joint positions, gripper_action]
        """
        no_param = False
        if isinstance(params, list):
            if len(params) == 0:
                no_param = True
        elif isinstance(params, np.ndarray):
            if params.shape[0] == 0:
                no_param = True
        if params is None or no_param:
            reset_joint_positions = np.append(self.reset_joint_positions["from_top"], -1.0)
            print("HERE!")
        else:
            reset_joint_positions = params[:8]
        # print("reset joint pos", reset_joint_positions)
        reset_joints_to_custom(self.robot_interface, reset_joint_positions)   

    """
    Skill helper functions:
        _move_to is interruptable while others are not
    """
    def _move_to(self, params, step_size=0.015, deg_step_size=5):
        """
        Moves end effector to goal position and orientation.

        Args: 
            params (9-tuple of floats) : [goal_pos, goal_quat, gripper_state, finetune]
        """

        while self.robot_interface.state_buffer_size == 0:
            logger.warn("Robot state not received")
            time.sleep(0.5)

        # extract  params
        action = params[:-1]
        action = np.clip(action, -1, 1)
        goal_pos = action[:3]
        goal_orn = action[3:7]
        gripper_action = params[7]
        finetune = True if params[8] > 0 else False

        fine_tune_dist = 0.05 # start fine tuning after distance to goal is within this value

        tran_inter, ori_inter = self.interpolate_poses(goal_pos, goal_orn, step_size=step_size, step_size_deg=deg_step_size) # num_step should between [10, 30], the larger the slower


        for i in range(len(tran_inter)):
            
            self._check_for_interrupt()
            if self.interrupt:
                break

            trans = tran_inter[i]
            ori = U.mat2quat(ori_inter[i])

            for _ in range(3): # 3 = number of times each waypoint is executed. smaller = faster but less accurate, larger = slower but more accurate
                new_action = self.poses_to_action(trans, ori)
                new_action = np.concatenate((new_action, [gripper_action]))
                self.robot_interface.control(
                    controller_type=self.controller_type,
                    action=new_action,
                    controller_cfg=self.controller_config,
                )
            
            cur_quat, cur_pos = self.robot_interface.last_eef_quat_and_pos
            cur_pos = cur_pos.flatten()
            pos_error = goal_pos - cur_pos
            if np.linalg.norm(pos_error) < fine_tune_dist:
                break

        # fine tune
        if finetune and not self.interrupt:
            tran_inter, ori_inter = self.interpolate_poses(action[:3], action[3:7], step_size=0.005, step_size_deg=deg_step_size) # num_step should between [10, 30], the larger the slower
            for i in range(len(tran_inter)):

                self._check_for_interrupt()
                if self.interrupt:
                    break
                trans = tran_inter[i]
                ori = U.mat2quat(ori_inter[i])

                for _ in range(3): # 3 = number of times each waypoint is executed. smaller = faster but less accurate, larger = slower but more accurate
                    new_action = self.poses_to_action(trans, ori)
                    new_action = np.concatenate((new_action, [gripper_action]))
                    self.robot_interface.control(
                        controller_type=self.controller_type,
                        action=new_action,
                        controller_cfg=self.controller_config,
                    )

    def _gripper_action(self, params):
        """
        Closes or opens gripper
        
        Args:
            params : -1 to open, 1 to close
        """
        action = np.zeros(7)
        action[-1] = params[0]

        for _ in range(30):
            self.robot_interface.control(
                controller_type=self.controller_type,
                action=action,
                controller_cfg=self.controller_config,
            )
    
    def _rehome(self, params):
        """
        Returns to home position with gripper pointing down. Finetuning step is turned off as default

        Args:
            params (8-tuple of floats) : [reset joint positions, gripper_action]
        """
        gripper_action = params[7]
        self._reset_joints(params=params)

        # if gripper action has not been completed, take gripper action
        if not self._get_gripper_state() == gripper_action:
            print("HERE")
            self._gripper_action([gripper_action])
            
    def _rehome_pos_quat(self, params):
        """
        NOTE : Deprecated. Use _rehome() instead
        Returns to home position with gripper pointing down. Finetuning step is turned off as default

        Args:
            params (9-tuple of floats) : [gripper_pos, gripper_quat, gripper_action, finetune]
                finetune : 1 for True, 0 for False
        """
        
        self._move_to(params=params, step_size=0.015)
            
        final_quat, final_pos = self.robot_interface.last_eef_quat_and_pos
        print("rehome pos error", params[:3] - final_pos.flatten())

    def _pause(self, params):
        """
        pause in place for specified number of seconds

        Args:
            params (2-tuple of floats) : [gripper_action, seconds to pause]
        """
        gripper_action = params[0]
        sec = params[1]
        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_action]

        start_time = time.time()

        while time.time() < start_time + sec:
            self.robot_interface.control(
                controller_type=self.controller_type,
                action=action,
                controller_cfg=self.controller_config,
            )

    def _get_gripper_state(self, thresh=0.055):
        """
        Checks whether gripper is closed (used during interrupt)
        Args:
            thresh (float) : gripper is considered closed if gripper opening is less than this value
        """
        last_gripper_width = self.robot_interface.last_gripper_q
        return 1 if last_gripper_width < thresh else -1

    def _get_eef_pos(self,):
        last_state = np.array(self.robot_interface.last_state.O_T_EE).reshape(4,4).T
        eef_pos = last_state[:-1,-1]
        return eef_pos

    # def _check_for_interrupt(self):
    #     """
    #     Check if human interrupt is received and valid
    #     """
    #     # TODO - replace this with actual signal
    #     if self.allow_interrupt:
    #         interrupt_signal = self.r.get(self.INTERRUPT_KEY).decode()
    #         if interrupt_signal == "False":
    #             self.interrupt = False
    #         elif interrupt_signal == "True":
    #             self.interrupt = True
    #             self.allow_interrupt = False
    #             print("rcvd interrupt", self.interrupt)

    def _check_for_interrupt(self):
        """
        Check if human interrupt is received and valid
        """
        if not self.enable_eeg_interrupt:
            return
        if self.allow_interrupt:
            from flask_client import check_interrupt
            self.interrupt = check_interrupt()
            if self.interrupt:
                self.allow_interrupt = False
                print("rcvd interrupt", self.interrupt)

    """
    Param unnormalization and Motion Interpolation functions
    """
    def unnormalize_params(self, action): # TODO add case for push 1d's
        # NOTE not used. All params are raw values in NOIR
        """
        Unnormalizes parameters from [-1, 1] to raw values

        Args:
            action : one-hot skill selection vector concatenated with params

        Returns: 
            action : action with unnormalized params
        """
        # find out which skill is called
        action = action.copy()
        skill_idx = np.argmax(action[:self.num_skills])
        skill_name = self.idx2skill[skill_idx]
        params = action[self.num_skills:]

        if skill_name == "push":
            params[0] = ( ((params[0] + 1) / 2 ) * (self.workspace_limits["x"][1] - self.workspace_limits["x"][0]) ) + self.workspace_limits["x"][0]
            params[1] = ( ((params[1] + 1) / 2 ) * (self.workspace_limits["y"][1] - self.workspace_limits["y"][0]) ) + self.workspace_limits["y"][0]
            params[2] = ( ((params[2] + 1) / 2 ) * (self.workspace_limits["z"][1] - self.workspace_limits["z"][0]) ) + self.workspace_limits["z"][0]
            params[3] = ( ((params[3] + 1) / 2 ) * (self.workspace_limits["x"][1] - self.workspace_limits["x"][0]) ) + self.workspace_limits["x"][0]
            params[4] = ( ((params[4] + 1) / 2 ) * (self.workspace_limits["y"][1] - self.workspace_limits["y"][0]) ) + self.workspace_limits["y"][0]
            params[5] = ( ((params[5] + 1) / 2 ) * (self.workspace_limits["z"][1] - self.workspace_limits["z"][0]) ) + self.workspace_limits["z"][0]
            if self.use_yaw:
                params[6] = ( ((params[6] + 1) / 2 ) * (self.yaw_limits[1] - self.yaw_limits[0]) ) + self.yaw_limits[0]

        elif skill_name == "push_x":
            params[0] = ( ((params[0] + 1) / 2 ) * (self.workspace_limits["x"][1] - self.workspace_limits["x"][0]) ) + self.workspace_limits["x"][0]
            params[1] = ( ((params[1] + 1) / 2 ) * (self.workspace_limits["y"][1] - self.workspace_limits["y"][0]) ) + self.workspace_limits["y"][0]
            params[2] = ( ((params[2] + 1) / 2 ) * (self.workspace_limits["z"][1] - self.workspace_limits["z"][0]) ) + self.workspace_limits["z"][0]
            params[3] = ( ((params[3] + 1) / 2 ) * (self.workspace_limits["x"][1] - self.workspace_limits["x"][0]) ) + self.workspace_limits["x"][0]
            if self.use_yaw:
                params[4] = ( ((params[6] + 1) / 2 ) * (self.yaw_limits[1] - self.yaw_limits[0]) ) + self.yaw_limits[0]

        elif skill_name == "push_y":
            params[0] = ( ((params[0] + 1) / 2 ) * (self.workspace_limits["x"][1] - self.workspace_limits["x"][0]) ) + self.workspace_limits["x"][0]
            params[1] = ( ((params[1] + 1) / 2 ) * (self.workspace_limits["y"][1] - self.workspace_limits["y"][0]) ) + self.workspace_limits["y"][0]
            params[2] = ( ((params[2] + 1) / 2 ) * (self.workspace_limits["z"][1] - self.workspace_limits["z"][0]) ) + self.workspace_limits["z"][0]
            params[3] = ( ((params[3] + 1) / 2 ) * (self.workspace_limits["y"][1] - self.workspace_limits["y"][0]) ) + self.workspace_limits["y"][0]
            if self.use_yaw:
                params[4] = ( ((params[6] + 1) / 2 ) * (self.yaw_limits[1] - self.yaw_limits[0]) ) + self.yaw_limits[0]

        else:
            params[0] = ( ((params[0] + 1) / 2 ) * (self.workspace_limits["x"][1] - self.workspace_limits["x"][0]) ) + self.workspace_limits["x"][0]
            params[1] = ( ((params[1] + 1) / 2 ) * (self.workspace_limits["y"][1] - self.workspace_limits["y"][0]) ) + self.workspace_limits["y"][0]
            params[2] = ( ((params[2] + 1) / 2 ) * (self.workspace_limits["z"][1] - self.workspace_limits["z"][0]) ) + self.workspace_limits["z"][0]
            if self.use_yaw:
                params[3] = ( ((params[3] + 1) / 2 ) * (self.yaw_limits[1] - self.yaw_limits[0]) ) + self.yaw_limits[0]

        return np.concatenate([action[:self.num_skills], params])

    def interpolate_poses(self, target_pos, target_rot=None, num_steps=None, step_size=None, step_size_deg=5):
        assert num_steps is None or step_size is None
        last_robot_state = self.robot_interface._state_buffer[-1]
        last_gripper_state = self.robot_interface._gripper_state_buffer[-1]
        ee_pose = np.array(last_robot_state.O_T_EE).reshape((4, 4)).T
        if np.count_nonzero(ee_pose.reshape(-1)) == 0:
            raise Exception("GOT ZERO EE POSE")
        start_pos = ee_pose[:3, 3]
        start_rot = ee_pose[:3, :3]
        target_rot = U.quat2mat(target_rot)
    

        rot_step_size = np.radians(step_size_deg) # degrees per step

        if num_steps is None:
            # calculate number of steps in terms of translation
            delta_pos = target_pos - start_pos
            if np.linalg.norm(delta_pos) > 0:
                num_steps_pos = math.ceil(np.linalg.norm(delta_pos) / step_size)
            else:
                num_steps_pos = 1
            # calculate number of steps in terms of rotation
            rot_angle = np.arccos((np.trace(np.dot(start_rot, np.transpose(target_rot))) - 1) / 2)
            if rot_angle >= np.radians(rot_step_size):
                num_steps_rot = int(np.ceil(rot_angle / rot_step_size))  # 2 degree for one step
            else:
                num_steps_rot = 1
            
            num_steps = max(num_steps_rot, num_steps_pos)
            # print("rot angle", rot_angle)
            # print(f'num_steps_pos: {num_steps_pos}')
            # print(f'num_steps_rot: {num_steps_rot}')
            # print("num steps", num_steps)

        tran_inter = self.interpolate_tranlations(start_pos, target_pos, num_steps)
        ori_inter = self.interpolate_rotations(start_rot, target_rot, num_steps)

        return tran_inter, ori_inter

    def interpolate_tranlations(self, T1, T2, num_steps, perturb=False):
        delta_pos = T2 - T1
        pos_step_size = delta_pos / num_steps
        grid = np.arange(num_steps).astype(np.float64)
        if perturb:
            # move the interpolation grid points by up to a half-size forward or backward
            perturbations = np.random.uniform(
                low=-0.5,
                high=0.5,
                size=(num_steps - 2,),
            )
            grid[1:-1] += perturbations
        pos_steps = np.array([T1 + grid[i] * pos_step_size for i in range(num_steps)])

        # add in endpoint
        pos_steps = np.concatenate([pos_steps, T2[None]], axis=0)

        return pos_steps[1:]

    def interpolate_rotations(self, R1, R2, num_steps):
        """
        Interpolate between 2 rotation matrices.
        """
        q1 = U.mat2quat(R1)
        q2 = U.mat2quat(R2)
        rot_steps = np.array([U.quat2mat(self.quat_slerp(q1, q2, tau=(float(i) / num_steps))) for i in range(num_steps)])

        # add in endpoint
        rot_steps = np.concatenate([rot_steps, R2[None]], axis=0)

        return rot_steps[1:]
    
    def quat_slerp(self, q1, q2, tau):
        """
        Adapted from robosuite.
        """
        if tau == 0.0:
            return q1
        elif tau == 1.0:
            return q2
        d = np.dot(q1, q2)
        if abs(abs(d) - 1.0) < np.finfo(float).eps * 4.:
            return q1
        if d < 0.0:
            # invert rotation
            d = -d
            q2 *= -1.0
        angle = math.acos(np.clip(d, -1, 1))
        if abs(angle) < np.finfo(float).eps * 4.:
            return q1
        isin = 1.0 / math.sin(angle)
        q1 = q1 * math.sin((1.0 - tau) * angle) * isin
        q2 = q2 * math.sin(tau * angle) * isin
        q1 = q1 + q2
        return q1
    
    def poses_to_action(self, target_pos, target_rot=None, max_dpos=0.05, max_drot=0.2):
        """
        Takes a starting eef pose and target controller pose and returns a normalized action that
        corresponds to the desired controller target.
        """
        last_robot_state = self.robot_interface._state_buffer[-1]
        last_gripper_state = self.robot_interface._gripper_state_buffer[-1]
        ee_pose = np.array(last_robot_state.O_T_EE).reshape((4, 4)).T
        if np.count_nonzero(ee_pose.reshape(-1)) == 0:
            raise Exception("GOT ZERO EE POSE")
        start_pos = ee_pose[:3, 3]
        start_rot = ee_pose[:3, :3]

        delta_position = target_pos - start_pos
        delta_position = np.clip(delta_position / max_dpos, -1., 1.)
        if target_rot is None:
            return delta_position

        target_rot = U.quat2mat(target_rot)

        # use the OSC controller's convention for delta rotation
        delta_rot_mat = target_rot.dot(start_rot.T)
        delta_quat = U.mat2quat(delta_rot_mat)
        delta_rotation = U.quat2euler(delta_quat)
        delta_rotation = np.clip(delta_rotation / max_drot, -1., 1.)
        return np.concatenate([delta_position, delta_rotation])
