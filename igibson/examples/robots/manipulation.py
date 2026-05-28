import argparse
import logging
import os
import random
import sys
from sys import platform

import numpy as np
import pybullet as p
import yaml

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings

from igibson.external.pybullet_tools.utils import quat_from_euler
from igibson.objects.articulated_object import URDFObject
from igibson.render.profiler import Profiler
from igibson.scenes.empty_scene import EmptyScene
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_category_path, get_ig_model_path
from igibson.utils.constants import MAX_INSTANCE_COUNT, MAX_CLASS_COUNT
from igibson.utils.vision_utils import segmentation_to_rgb, randomize_colors

from igibson.robots import Fetch as Fetch_iGibson

from igibson.external.pybullet_tools.utils import (
    get_max_limits,
    get_min_limits,
    get_sample_fn,
    joints_from_names,
    set_joint_positions,
)
from igibson.utils.utils import l2_distance, parse_config, restoreState
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper


class Arguments:
    def __init__(self):
        self.n_images = 10
        self.behavior = ["push_objects"]
        self.take_images = True
        self.n_objects = 10


args = Arguments()

fetch_yaml = """
scene: igibson
scene_id: Beechwood_0_int
hide_robot: false
build_graph: true
load_texture: true
pybullet_load_texture: true
trav_map_type: no_obj
trav_map_resolution: 0.1
trav_map_erosion: 3
should_open_all_doors: true

# domain randomization
texture_randomization_freq: null
object_randomization_freq: null

robot:
  name: Fetch
  action_type: continuous
  action_normalize: true
  proprio_obs:
    - eef_0_pos
    - eef_0_quat
    - trunk_qpos
    - arm_0_qpos_sin
    - arm_0_qpos_cos
    - gripper_0_qpos
    - grasp_main
  reset_joint_pos: null
  base_name: null
  scale: 1.0
  self_collision: true
  rendering_params: null
  grasping_mode: physical
  rigid_trunk: false
  default_trunk_offset: 0.365
  default_arm_pose: vertical
  controller_config:
    base:
      name: DifferentialDriveController
    arm_0:
      name: InverseKinematicsController
      kv: 2.0
    gripper_0:
      name: MultiFingerGripperController
      mode: binary
    camera:
      name: JointController
      use_delta_commands: False

# sensor spec
output: [rgb, depth, scan, occupancy_grid]
# image
# Primesense Carmine 1.09 short-range RGBD sensor
# http://xtionprolive.com/primesense-carmine-1.09
fisheye: false
image_width: 512
image_height: 512
vertical_fov: 90
# depth
depth_low: 0.35
depth_high: 3.0
# scan
# SICK TIM571 scanning range finder
# https://docs.fetchrobotics.com/robot_hardware.html
# n_horizontal_rays is originally 661, sub-sampled 1/3
n_horizontal_rays: 220
n_vertical_beams: 1
laser_linear_range: 25.0
laser_angular_range: 220.0
min_laser_dist: 0.05
laser_link_name: laser_link

# Objects to prevent from loading into the scene
not_load_object_categories: [
    breakfast_table,
    pool_table,
    straight_chair,
    armchair,
    coffee_table,
    console_table,
    pedestal_table,
    gaming_table,
    bed
]

# sensor noise
depth_noise_rate: 0.0
scan_noise_rate: 0.0

# visual objects
visible_target: false
visible_path: false

collision_ignore_link_a_ids: [0, 1, 2]  # ignore collisions with these robot links
"""


class Fetch:
    """A wrapper class for igibson.robots.Fetch that keeps track of manipulation-useful data

    Attributes
    ----------
    config : dict
        Configuration dictionary for Fetch's robot instance, read from configs/fetch.yaml
    reference : igibson.robots.Fetch
        The reference to the iGibson Fetch instance in the simulator
    """

    robot_joint_names = [
        "r_wheel_joint",
        "l_wheel_joint",
        "torso_lift_joint",
        "head_pan_joint",
        "head_tilt_joint",
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "forearm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
        "r_gripper_finger_joint",
        "l_gripper_finger_joint",
    ]

    arm_joints_names = [
        "torso_lift_joint",
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "forearm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
    ]

    arm_default_joint_positions = (
        0.10322468280792236,
        -1.414019864768982,
        1.5178184935241699,
        0.8189625336474915,
        2.200358942909668,
        2.9631312579803466,
        -1.2862852996643066,
        0.0008453550418615341,
    )

    robot_default_joint_positions = (
            [0.0, 0.0]
            + [arm_default_joint_positions[0]]
            + [0.0, 0.0]
            + list(arm_default_joint_positions[1:])
            + [0.01, 0.01]
    )

    arm_out_of_way_position = [0.30, -1.40, -0.21, 2.75, 1.69, -1.57, 1.86, 2.27]

    def __init__(self):
        """Prepares Fetch instance for import
        """
        config = yaml.load(fetch_yaml, Loader=yaml.FullLoader)
        self.config = config["robot"]
        self.config.pop("name")

        self.moving = False

    def setup_import(self, simulator, position=(0, 1, 0), orientation=(0, 0, -90), env=None):
        """A function to finish setup, import, and still Fetch in a given simulation with a given physical state

        Parameters
        ----------
        simulator : igibson.simulator.Simulator
            The iGibson simulator to load the Fetch instance into
        position : tuple[float, float, float]
            The global XYZ position to spawn the Fetch instance into
        orientation : tuple[float, float, float]
            The Pitch, Roll, and Yaw values (euler) to spawn the Fetch instance into
        env : igibson.envs.iGibsonEnv
            If using an iGibsonEnv, the environment to load the Fetch instance into
        """
        # Instantiate robot instance
        if not env:
            logging.warning("Motion Planning not supported in EmptyScene")
            self.reference = Fetch_iGibson(**self.config)

            # Import with pybullet
            simulator.import_object(self.reference)
        else:
            self.reference = env.robots[0]

            # Used for planning arm motion
            self.motion_planner = MotionPlanningWrapper(env, optimize_iter=0, full_observability_2d_planning=True,
                                                        collision_with_pb_2d_planning=True,
                                                        visualize_2d_planning=False, visualize_2d_result=False,
                                                        fine_motion_plan=True, arm_mp_algo="birrt")

        self.simulator = simulator
        self.env = env

        self.position = position
        self.orientation = orientation

        # Get pybullet body id
        body_ids = self.reference.get_body_ids()
        assert len(body_ids) == 1, "Fetch robot is expected to be single-body."
        self.body_id = body_ids[0]

        # Set position and orientation
        self.reset_pos_orn(position, orientation)

        # Fill in joint information
        self.arm_joint_ids = joints_from_names(self.body_id, self.arm_joints_names)
        self.all_joint_ids = joints_from_names(self.body_id, self.robot_joint_names)

        self.max_limits = get_max_limits(self.body_id, self.all_joint_ids)
        self.min_limits = get_min_limits(self.body_id, self.all_joint_ids)

        self.rest_position = self.robot_default_joint_positions
        self.joint_range = list(np.array(self.max_limits) - np.array(self.min_limits))
        self.joint_range = [item + 1 for item in self.joint_range]
        self.joint_damping = [0.1 for _ in self.joint_range]
        self.robot_arm_indices = [self.robot_joint_names.index(arm_joint_name)
                                  for arm_joint_name in self.arm_joints_names]
        # Stabilization
        self.reference.reset()
        self.reference.keep_still()

    def accurate_calculate_inverse_kinematics(self, target_pos, threshold, max_iter):
        """A function to calculate an arm position that puts Fetch's gripper at target_pos

        Parameters
        ----------
        target_pos : tuple[float, float, float]
            The target position of Fetch's gripper
        threshold : float
            The distance from the calculated position of Fetch's gripper to the intended target position that will
            indicate a solution has been found
        max_iter : int
            The maximum amount of iterations the Inverse Kinematics function may take to find a solution.

        Returns
        -------
        joint_poses : list[float]
            A list of floats whose positions correspond to joint IDs in arm_joint_ids, and whose values correspond to
            positions of those arm joints
        """
        # Save initial robot pose
        state_id = p.saveState()

        eef_link_id = self.reference.eef_links[self.reference.default_arm].link_id

        max_attempts = 10
        solution_found = False
        joint_poses = None
        for attempt in range(1, max_attempts + 1):
            logging.debug("Attempt {} of {}".format(attempt, max_attempts))
            # Get a random robot pose to start the IK solver iterative process
            # We attempt from max_attempt different initial random poses
            sample_fn = get_sample_fn(self.body_id, self.arm_joint_ids)
            sample = np.array(sample_fn())
            # Set the pose of the robot there
            set_joint_positions(self.body_id, self.arm_joint_ids, sample)

            it = 0
            # Query IK, set the pose to the solution, check if it is good enough repeat if not
            while it < max_iter:

                joint_poses = p.calculateInverseKinematics(
                    self.body_id,
                    eef_link_id,
                    target_pos,
                    lowerLimits=self.min_limits,
                    upperLimits=self.max_limits,
                    jointRanges=self.joint_range,
                    restPoses=self.rest_position,
                    jointDamping=self.joint_damping,
                )
                joint_poses = np.array(joint_poses)[self.robot_arm_indices]

                set_joint_positions(self.body_id, self.arm_joint_ids, joint_poses)

                dist = l2_distance(self.reference.get_eef_position(), target_pos)
                if dist < threshold:
                    solution_found = True
                    break
                logging.debug("Dist: " + str(dist))
                it += 1

            if solution_found:
                logging.debug(f"Solution found at iter: {it}, residual: {dist}")
                break
            else:
                logging.debug(f"IK attempt failed with residual {dist}, retrying")
                joint_poses = None

        restoreState(state_id)
        p.removeState(state_id)
        return joint_poses

    def move_hand_to_position(self, target_pos, body_id=-1):
        """A function to move Fetch's gripper to the passed position using Motion Planning and Inverse Kinematics

        Parameters
        ----------
        target_pos : tuple[float, float, float]
            The target position of Fetch's gripper, represented in XYZ world coordinates
        body_id : int
            A pybullet body ID corresponding to an object in the scene that is to be removed from the obstacles list
            used by iGibson's Motion Planning function to avoid objects during planning. If not set, all objects in
            the loaded environment are planned to be avoided during arm motion

        Returns
        -------
        executed_arm_motion : bool
            True if Inverse Kinematics found an arm position, and Motion Planning found a motion plan to that position,
            and if the plan was executed without fail, otherwise False
        """
        if not self.motion_planner:
            logging.warning("Motion Planning for EmptyScene not supported")
            return

        self.moving = True

        executed_arm_motion = False

        threshold = 0.03
        max_iter = 10

        joint_poses = self.accurate_calculate_inverse_kinematics(target_pos, threshold, max_iter)

        if joint_poses is not None:
            target_normal = np.array(target_pos) - np.array(joint_poses[-1])
            target_normal[2] = 0

            if body_id != -1:
                self.motion_planner.mp_obstacles.remove(body_id)

            # plan = self.motion_planner.plan_arm_push(target_pos, target_normal)
            if self.motion_planner.marker is not None:
                self.motion_planner.set_marker_position_direction(target_pos, target_normal)

            self.motion_planner.simulator_sync()
            plan = self.motion_planner.plan_arm_motion(joint_poses)

            if body_id != -1:
                self.motion_planner.mp_obstacles.append(body_id)

            if plan and len(plan) > 0:
                plan_len = len(plan)

                logging.info("Executing planned arm push")
                print("the plan has been found, waiting for push")

                for i, p in enumerate(plan):
                    set_joint_positions(self.body_id, self.arm_joint_ids, p)
                    self.env.step(None)
                    logging.debug(f"Arm push step {i} / {plan_len}")

                logging.debug(f"Arm push step {plan_len} / {plan_len}")
                set_joint_positions(self.body_id, self.arm_joint_ids, joint_poses)
                # self.motion_planner.interact(target_pos, target_normal)



                logging.info("End of execution")

                executed_arm_motion = True
            else:
                logging.warning("MP couldn't find path to pushing location")

        self.reference.keep_still()

        self.moving = False

        return executed_arm_motion

    def prepare_for_picture(self):
        """A function to move Fetch's arm out of its Field of View, and to reset Fetch's position, in order to capture a
           clean image from its camera
        """
        self.hide_mp_marker()

        set_joint_positions(self.body_id, self.arm_joint_ids, self.arm_out_of_way_position)

        self.reset_pos_orn()
        # Look forward and slightly down
        # self.set_joints(["head_tilt_joint", "head_pan_joint"], [0.4, 0.0])

    def get_joints(self, joint_names):
        """A function to get a tuple of joint IDs corresponding to the provided names

        Parameters
        ----------
        joint_names : list[str]
            A list of strings containing joint names

        Returns
        -------
        tuple[int]
            A tuple of joint IDs corresponding to the passed joint names
        """
        return joints_from_names(self.body_id, joint_names)

    def set_joints(self, joint_names, values):
        """A function to set the positions of joints corresponding to the provided names

        Parameters
        ----------
        joint_names : list[str]
            A list of strings containing joint names
        values : list[float]
            A list of joint positions corresponding to each name in the passed joint_names
        """
        set_joint_positions(self.body_id, self.get_joints(joint_names), values)

    def hide_mp_marker(self):
        """A function to hide the marker that Fetch's MotionPlanningWrapper uses to indicate the target position for
           Fetch's gripper
        """
        if self.motion_planner.marker is not None:
            self.motion_planner.set_marker_position_direction([0, 0, -1000], [0, 0, -1])

    def reset_pos_orn(self, position=None, orientation=None):
        """A function to set Fetch's position and orientation to a passed position and orientation, or its original
           position and orientation

        Parameters
        ----------
        position : tuple[float, float, float]
            The global XYZ position to move Fetch to, if None
        orientation : tuple[float, float, float]
            The Pitch, Roll, and Yaw values (euler) to set Fetch's rotation to
        """
        if position is None:
            position = self.position
        if orientation is None:
            orientation = self.orientation

        self.reference.set_position(position)
        self.reference.set_orientation(euler_to_quaternion(orientation))



# This is where a lot of the meat is
def env_loop(env, fetch, spawned_objects, logger):
    simulator = env.simulator

    experiment_successful = True

    # Setup Fetch's Motion Planner
    fetch.motion_planner.setup_arm_mp()

    # Image output setup
    needs_new_picture = True

    counter = 0
    picture_counter = 0
    move_hand_fails = 0
    while True:
        counter += 1
        with Profiler("Environment step", logger=logger):
            # Remove fallen objects from selectable objects
            fallen = []
            for i in range(len(spawned_objects)):
                obj = spawned_objects[i]
                if obj.get_position_orientation()[0][2] < 0.5:
                    fallen.append(obj)
            for obj in fallen:
                spawned_objects.remove(obj)

            # Pick an object to move, move the end effector to it with minimal collisions along the way
            if "push_objects" in args.behavior and counter % 5 == 0 and not fetch.moving:
                if not len(spawned_objects):
                    logging.error("All spawned objects have been destroyed, exiting")
                    experiment_successful = False
                    break

                # Pick object
                target_obj = random.choice(spawned_objects)
                target_pos = target_obj.get_position_orientation()[0]
                
                # Start motion planning, ignore collision for target object because we want to push it
                needs_new_picture = fetch.move_hand_to_position(target_pos, target_obj.get_body_ids()[0])

            if not needs_new_picture:
                move_hand_fails += 1
                if move_hand_fails >= 100:
                    experiment_successful = False
                    break
            else:
                move_hand_fails = 0

            # Step environment
            state, reward, done, info = env.step(None)

            logging.info("Moving arm back")
            # Move arm back to default position
            fetch.prepare_for_picture()

            # This doesn't matter
            if args.take_images and needs_new_picture:
                env.step(None)

                picture_counter += 1
                logging.info(f"Taking picture {picture_counter}")
                # take_pictures(picture_counter, simulator, args, images_subdir, colors_is)

                needs_new_picture = False

            if picture_counter >= args.n_images:
                break

            if fetch.reference.get_position_orientation()[0][2] < -10:
                experiment_successful = False
                logging.error("Robot fell below floor")
                break

            # Exit
            if done:
                logging.debug("Episode finished after {} time steps".format(counter + 1))
                break

    return experiment_successful


def euler_to_quaternion(rotation):
    """A function to convert Euler rotation values to a Quaternion

    Parameters
    ----------
    rotation : tuple[float, float, float]
        Tuple containing a pitch, roll, and yaw to convert to a Quaternion
    Returns
    -------
    tuple : tuple[float, float, float, float]
        A tuple containing the equivalent quaternion [x, y, z, w]
    """
    mutable_rotation = list(rotation)

    # Convert integer (degree) rotations to radians
    for i in range(3):
        if mutable_rotation[i] == int(mutable_rotation[i]):
            mutable_rotation[i] *= 0.0174533

    return quat_from_euler(mutable_rotation)


def setup_scene(env, fetch, object_indeces=None, n=5):
    random_categories = [
        'watch',
        'eggplant',
        'alarm',
        'steak',
        'ring',
        'pasta',
        'cupcake',
        'cup',
        'cucumber',
        'package',
        'newspaper',
        'mushroom',
        'mug',
        'mousetrap',
        'mouse',
        'tea_bag',
        'teapot'
    ]
    """A list of possible object types to import"""

    category_scales = {
        'watch': 0.03,
        'eggplant': 1.5,
        'alarm': 0.25,
        'steak': 0.2,
        'ring': 0.03,
        'saw': 0.01,
        'pasta': 0.3,
        'cupcake': 0.17,
        'cup': 0.3,
        'cucumber': 5,
        'package': 0.05,
        'newspaper': 0.075,
        'mushroom': 1.5,
        'mug': 0.7,
        'mousetrap': 0.01,
        'mouse': 1,
        'tea_bag': 1.75,
        'teapot': 1.2,
    }
    """A map of sizes for each of the available object types"""

    breakfast_table = {
        "category": "breakfast_table",
        "model": "1b4e6f9dd22a8c628ef9d976af675b86",
        "pos": (0, 0.2, 1),
        "orn": (0, 0, 0),
    }
    """A dictionary storing values for the standard breakfast table URDFObject"""

    """A function to spawn the table and manipulation objects, and relocate the Fetch instance accordingly

    Parameters
    ----------
    env : igibson.envs.iGibsonEnv
        The iGibson environment to load the table and objects into
    fetch : fetch_irvl.Fetch
        The Fetch instance to place in front of the spawned table
    object_indeces : list
        A list of indeces of the supported iGibson objects to spawn. To generate random objects, leave empty
    n : int
        The number of random objects to spawn if and only if object_indeces is an empty list

    Returns
    -------
    tuple : tuple[URDFObject, list[URDFObject]]
        A tuple containing the spawned table and a list of the spawned objects
    """
    if object_indeces is None:
        object_indeces = []

    simulator = env.simulator

    # Spawn table
    category = breakfast_table["category"]

    # If the specific model is given, we use it. If not, we select one randomly
    table_model = breakfast_table["model"]

    # Create the full path combining the path for all models and the name of the model
    model_path = get_ig_model_path(category, table_model)
    filename = os.path.join(model_path, table_model + ".urdf")

    # Create a unique name for the object instance
    name_offset = 0
    if not isinstance(simulator.scene, EmptyScene):
        while f"{category}_{name_offset}" in simulator.scene.objects_by_name.keys():
            name_offset += 1
    obj_name = f"{category}_{name_offset}"

    # Load the specs of the object categories, e.g., common scaling factor
    avg_category_spec = get_ig_avg_category_specs()

    # Create and import the table
    table_obj = URDFObject(
        filename,
        name=obj_name,
        category=category,
        model_path=model_path,
        texture_randomization=False,
        overwrite_inertial=True,
        scale=np.array([1.6, 1.6, 1.6]),
    )
    simulator.import_object(table_obj)

    table_body_id = table_obj.get_body_ids()[0]

    # table_obj.set_position_orientation(breakfast_table["pos"], quat_from_euler(breakfast_table["orn"]))

    reset_success = False

    ig_floor = simulator.scene.get_random_floor()
    p1 = simulator.scene.get_random_point(ig_floor)[1]
    p2 = simulator.scene.get_random_point(ig_floor)[1]
    shortest_path, geodesic_distance = simulator.scene.get_shortest_path(ig_floor, p1[:2], p2[:2], entire_path=True)

    i = 0
    while not reset_success:
        while i + 3 >= len(shortest_path):
            p1 = simulator.scene.get_random_point(ig_floor)[1]
            p2 = simulator.scene.get_random_point(ig_floor)[1]
            shortest_path, geodesic_distance = simulator.scene.get_shortest_path(ig_floor, p1[:2], p2[:2],
                                                                                 entire_path=True)
            i = 0

        table_pos = np.append(shortest_path[i], p1[-1] + 0.1)
        robot_pos = (table_pos[0], table_pos[1] + 0.7, table_pos[2])

        env.land(table_obj, table_pos, breakfast_table["orn"])
        env.land(fetch.reference, robot_pos, fetch.orientation)

        table_obj.set_position_orientation(table_pos, euler_to_quaternion(breakfast_table["orn"]))
        fetch.reset_pos_orn(robot_pos, fetch.orientation)

        reset_success = env.test_valid_position(table_obj, table_pos, breakfast_table["orn"]) \
                        and env.test_valid_position(fetch.reference, robot_pos, fetch.orientation) \
                        and len(list(p.getContactPoints(bodyA=table_body_id))) <= 9 \
                        and len(list(p.getContactPoints(bodyA=fetch.body_id))) <= 6
        i += 3
        for _ in range(0, 100):
            p.stepSimulation()

    fetch.position = fetch.reference.get_position_orientation()[0]
    fetch.reset_pos_orn()

    env.land(table_obj, table_pos, breakfast_table["orn"])
    env.land(fetch.reference, robot_pos, fetch.orientation)

    p.changeDynamics(table_body_id, -1, mass=0)
    table_obj.set_velocities([(0, 0)])

    # move robot out of the way before spawning objects
    fetch.prepare_for_picture()

    # Viewport setup
    if simulator.viewer is not None:
        simulator.viewer.initial_pos = [fetch.position[0] + 1, fetch.position[1] - 0.9, fetch.position[2] + 1]
        simulator.viewer.initial_view_direction = [-0.7, 0, -0.7]
        simulator.viewer.reset_viewer()

    table_aabb = p.getAABB(table_body_id)
    table_aabb_min, table_aabb_max = table_aabb[0], table_aabb[1]

    # Spawn objects
    spawned = []

    if object_indeces:
        for idx in object_indeces:
            obj_category = random_categories[idx]

            spawned.append(URDFObject(filename=get_random_URDF_object(obj_category),
                                      category=category,
                                      scale=np.array([category_scales[obj_category] * 3])))
    # else:
    #     for i in range(n):
    #         obj_category = np.random.choice(random_categories)
    #         spawned.append(
    #             URDFObject(filename=get_random_URDF_object(obj_category),
    #                        category=obj_category,
    #                        scale=np.array([category_scales[obj_category]] * 3)))

    # Import and set position, orientation of objects
    for obj in spawned:
        obj_category = obj.category
        name_offset = 0
        if not isinstance(simulator.scene, EmptyScene):
            while f"{obj_category}_{name_offset}" in simulator.scene.objects_by_name.keys():
                name_offset += 1
        obj.name = f"{obj_category}_{name_offset}"
        simulator.import_object(obj)

        x_dist = (table_aabb_max[0] - table_aabb_min[0]) / 3
        y_dist = (table_aabb_max[1] - table_aabb_min[1]) / 3
        obj.set_position_orientation([np.random.uniform(table_pos[0] - x_dist, table_pos[0] + x_dist),
                                      np.random.uniform(table_pos[1] - y_dist, table_pos[1] + y_dist),
                                      table_aabb_max[2] + 0.5],
                                     [0, 0, 0, 1])
        for i in range(5):
            env.step(None)

    return table_obj, spawned


def get_random_URDF_object(category):
    """A function to randomly select an existing URDF object from provided category in the existing Igibson database

    Parameters
    ----------
    category : str
        A string that contains name of the category that this function will randomly select an object from

    Returns
    -------
    str : str
        A string containing the full path of the selected URDF model
    """
    # Create a path lead to the database that can be used to locate all models
    model_database_path = os.path.join(igibson.ig_dataset_path, "objects")
    # Category path
    category_path = os.path.join(model_database_path, category)

    # get all models under that folder
    models_collection = os.listdir(category_path)

    # select a random model to use
    model_path = os.path.join(category_path, np.random.choice(models_collection))

    # get the URDF file
    urdf_path = [_ for _ in os.listdir(model_path) if _.endswith(".urdf")]

    # return the full path  of a randomly selected URDF model
    return os.path.join(model_path, np.random.choice(urdf_path))


def main():
    settings = MeshRendererSettings(enable_shadow=False, msaa=False, texture_scale=0.5)

    env = iGibsonEnv(yaml.load(fetch_yaml, Loader=yaml.FullLoader), mode="gui_interactive", rendering_settings=settings)
    simulator = env.simulator

    fetch = Fetch()

    fetch.setup_import(simulator, env=env)

    table, spawned_objects = setup_scene(env, fetch, object_indeces=[], n=args.n_objects)

    fetch.motion_planner.setup_arm_mp()

    robot_pos_orn = fetch.reference.get_position_orientation()
    env.reset()
    fetch.reference.set_position_orientation(*robot_pos_orn)

    experiment_successful = env_loop(env, fetch, spawned_objects, logging.root)


if __name__ == "__main__":
    main()
