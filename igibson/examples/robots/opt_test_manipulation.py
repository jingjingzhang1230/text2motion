import argparse
import logging
import os
import random
import sys
from sys import platform

from pathlib import Path
from importlib.resources import read_text
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

import numpy as np
import pybullet as p
import yaml

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.objects.visual_marker import VisualMarker

from igibson.external.pybullet_tools.utils import quat_from_euler
from igibson.objects.articulated_object import URDFObject
from igibson.render.profiler import Profiler
from igibson.scenes.empty_scene import EmptyScene
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_category_path, get_ig_model_path
from igibson.utils.constants import MAX_INSTANCE_COUNT, MAX_CLASS_COUNT
from igibson.utils.vision_utils import segmentation_to_rgb, randomize_colors

from igibson.utils.constants import ViewerMode

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
        self.n_objects = 2 # only 3 objects are spawned


args = Arguments()

# import yaml config
cfg_path = os.path.join(igibson.configs_path, "manipulation_scene_5.yaml")
fetch_yaml = Path(cfg_path).read_text(encoding="utf-8")

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
        self.plan_counter = 0
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
                                                        visualize_2d_planning=True, visualize_2d_result=True,
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

        max_attempts = 100
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

    def get_color(self, i):
                        if i<0 or i>4:
                            raise ValueError("Color code out of range.")
                    
                        # color: green, blue, red, black, white
                        if i == 1:
                            color_code = [0,0,1,0.5]
                        elif i == 0:
                            color_code = [0,1,0,0.5]
                        elif i == 2:
                            color_code = [1,0,0,0.5]
                        elif i == 3:
                            color_code = [0,0,0,0.5]
                        elif i == 4:
                            color_code = [1,1,1,0.5]
                    
                        return color_code

    def move_hand_to_position(self, env, target_pos, body_id=-1):
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
            self.plan = plan
            if body_id != -1:
                self.motion_planner.mp_obstacles.append(body_id)

        
            if plan and len(plan) > 0:
                plan_len = len(plan)
                self.sample_number = np.array(env.config["sample_number"])
                self.plan_counter += 1
               
                print(str(self.plan_counter) + " path has been found, remaining " + str(self.sample_number - self.plan_counter) + " path.")
                logging.info("Executing planned arm push")
                

                # print("The plan has been found, waiting for push")
                
                logging.debug(f"Arm push step {plan_len} / {plan_len}")
                set_joint_positions(self.body_id, self.arm_joint_ids, joint_poses)
                # self.motion_planner.interact(target_pos, target_normal)

                logging.info("End of execution")
                
                
                executed_arm_motion = True

                # input("Arm motion plan shown. Press any key to continue...")

        else:
                # logging.warning("MP couldn't find path to pushing location")
        
                self.reference.keep_still()
                self.moving = False

        return executed_arm_motion


    def prepare_for_picture(self):
        """A function to move Fetch's arm out of its Field of View, and to reset Fetch's position, in order to capture a
           clean image from its camera
        """
        # self.hide_mp_marker()

        set_joint_positions(self.body_id, self.arm_joint_ids, self.arm_out_of_way_position)

        self.reset_pos_orn()
        # Look forward and slightly down
        self.set_joints(["head_tilt_joint", "head_pan_joint"], [0.4, 0.0])

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

# end of class of Fetch



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
    all_plans = []
    

    # respawn the target obj(first obj)
    # Execute the push
    # Only reset after a successful push, and only once per new plan


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
                # target_obj = random.choice(spawned_objects)
                # We always pick the first obj in the list
                target_obj = (spawned_objects[0])

                def wait_ticks(env, n=5):
                    for _ in range(n):
                        env.simulator.step() 
                def reset_obj(env):
                    reset_pos = [env.config["x_position"][0], env.config["y_position"][0], 0.6]
                    env.land(target_obj, reset_pos, [0, 0, 0])
                    wait_ticks(env, 5)
                    fetch.prepare_for_picture()
                
                target_pos = target_obj.get_position_orientation()[0]


                # Start motion planning, ignore collision for target object because we want to push it
                while fetch.plan_counter < (np.array(env.config["sample_number"])):
                    needs_new_picture = fetch.move_hand_to_position(
                        env, target_pos, target_obj.get_body_ids()[0]
                    )
                    
                    if fetch.plan is not None:
                        all_plans.append(fetch.plan)
                    if needs_new_picture == True:
                        reset_obj(env=env)
                

                def extract_features(path):
                        path_array = np.array(path)  # Convert to numpy array
                        if path_array.ndim == 1:  # If the path has only one point, reshape it
                            path_array = path_array.reshape(1, -1)
                        features = []
                        for dim in range(path_array.shape[1]):  # Iterate over dimensions (x, y, z)
                            dim_values = path_array[:, dim]
                            features.extend([
                                np.mean(dim_values),  # Mean
                                np.std(dim_values),   # Standard deviation
                                np.min(dim_values),   # Min value
                                np.max(dim_values)    # Max value
                            ])
                        return features

                def clustering(all_plans, num_clusters = 2):
                    features = [extract_features(path) for path in all_plans]
                    # Standardize the features
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(features)
                    # Perform KMeans clustering
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                    labels = kmeans.fit_predict(scaled_features)

                    # Identify the most centered path for each cluster
                    most_centered_paths = []
                    for cluster_label in range(num_clusters):  # Number of clusters
                        cluster_indices = np.where(labels == cluster_label)[0]  # Indices of paths in this cluster
                        cluster_features = scaled_features[cluster_indices]     # Features of paths in this cluster
                        centroid = kmeans.cluster_centers_[cluster_label]       # Centroid of this cluster
                        distances = cdist(cluster_features, [centroid])         # Distance from each path to centroid
                        closest_index = cluster_indices[np.argmin(distances)]   # Index of the closest path
                        most_centered_paths.append(all_plans[closest_index])    # Append the most centered path
                    return most_centered_paths

                # visualization

                most_centered_paths = clustering(all_plans, num_clusters = 2)
                for j in range(len(most_centered_paths)): 
                    markers = []
                    marker_idx = 0
                    marker_color = fetch.get_color(j)
                    for i, n in enumerate(most_centered_paths[j]):
                        set_joint_positions(fetch.body_id, fetch.arm_joint_ids, n)
                        fetch.env.step(None)
                        # logging.debug(f"Arm push step {i} / {fetch.plan_len}")
                        # print(f"Arm push step {i} / {plan_len}")
                        # for every 5 steps, draw a marker
                        if i % 5 == 0:
                            way_point_task_space  = fetch.reference.get_eef_position()
                            markers.append(VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.03,rgba_color=marker_color))
                            # print(marker_idx)
                            env.simulator.import_object(markers[marker_idx])
                            markers[marker_idx].set_position([way_point_task_space[0], way_point_task_space[1], way_point_task_space[2]])
                            marker_idx += 1        
            if fetch.plan_counter >= (np.array(env.config["sample_number"])):
                reset_obj(env=env)
                # make the change visible
                for _ in range(5):
                    try:
                        env.simulator.step()        # iGibson step (advances physics + render)
                    except AttributeError:
                        p.stepSimulation()          # fallback if needed
                #logging.info("Moving arm back")
                print("There are " + str(fetch.plan_counter) + " path has been found, exiting ...")
                input("Press ENTER to exit...")
                experiment_successful = False
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

def setup_scene(env, fetch, object_indeces=None, n=2):
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
        'alarm': 0.5,
        'steak': 0.2,
        'ring': 0.03,
        'saw': 0.01,
        'pasta': 0.3,
        'cupcake': 0.2,
        'cup': 0.3,
        'cucumber': 5,
        'package': 0.008,
        'newspaper': 0.075,
        'mushroom': 2.2,
        'mug': 0.7,
        'mousetrap': 0.01,
        'mouse': 1.3,
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

        # random spwaned
        # table_pos = np.append(shortest_path[i], p1[-1] + 0.1)

        # fixed table position
        table_pos = np.array(env.config["table_pos"], dtype=float)


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

    print("The table is spawned at " + str(table_pos))

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
        # simulator.viewer.initial_pos = [fetch.position[0] + 1, fetch.position[1] - 0.9, fetch.position[2] + 1]
        #simulator.viewer.initial_pos = [-3.1, -4, 1.5]
        simulator.viewer.initial_pos = np.array(env.config["viewer_pos"])
        #simulator.viewer.initial_view_direction = [-0.5, 0.6, -0.6]
        simulator.viewer.initial_view_direction = np.array(env.config["viewer_direction"])
        simulator.viewer.reset_viewer()

    table_aabb = p.getAABB(table_body_id)
    table_aabb_min, table_aabb_max = table_aabb[0], table_aabb[1]



    # Spawn objects
    spawned = []
        # insert the desired spwaned objects here:
  
    desired_category = (env.config["desired_category"])
    for obj_category in desired_category:
        spawned.append(
            URDFObject(filename=get_random_URDF_object(obj_category),
                        category=obj_category,
                        scale=np.array([category_scales[obj_category]] * 3)))
        
    # Import and set position, orientation of objects
    for obj in spawned:
        obj_category = obj.category
        name_offset = 0
        if not isinstance(simulator.scene, EmptyScene):
            while f"{obj_category}_{name_offset}" in simulator.scene.objects_by_name.keys():
                name_offset += 1
        obj.name = f"{obj_category}_{name_offset}"
        simulator.import_object(obj)

    # np.random.uniform(a,b) : take a uniform distribution for a to b, choose a number randomly from this range.
    # 每次run 都会生成一个新的table_pos，所以必须重新获取位置      

    # call spawn position from yaml
    # fixed object position & order
    # z change to 0.8 prevent falling
    counter = 1
    for obj in spawned:
            if counter == 1:
                obj.set_position_orientation([env.config["x_position"][0], env.config["y_position"][0], table_aabb_max[2] + 0.2],
                                            [0, 0, 0, 0.6])
                print("The first obj is spawned.")
                counter += 1
            elif counter == 2:
                obj.set_position_orientation([env.config["x_position"][1], env.config["y_position"][1], table_aabb_max[2] + 0.2],
                                            [0, 0, 0, 0.6])
                print("The second obj is spawned.")
                counter += 1
            elif counter == 3:
                obj.set_position_orientation([env.config["x_position"][2], env.config["y_position"][2], table_aabb_max[2] + 0.2],
                                            [0, 0, 0, 0.6])
                print("The third obj is spawned.")
            
            
    # reinforcement learning
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

# convert back to dict
# cfg = yaml.safe_load(fetch_yaml)

def main():
    settings = MeshRendererSettings(enable_shadow=False, msaa=False, texture_scale=0.5)

    env = iGibsonEnv(yaml.load(fetch_yaml, Loader=yaml.FullLoader), mode="gui_interactive", rendering_settings=settings)
    simulator = env.simulator

    fetch = Fetch()

    fetch.setup_import(simulator, env=env)

    table_obj, spawned_objects = setup_scene(env, fetch, object_indeces=[], n=args.n_objects)

    fetch.motion_planner.setup_arm_mp()

    robot_pos_orn = fetch.reference.get_position_orientation()
    env.reset()
    fetch.reference.set_position_orientation(*robot_pos_orn)


    experiment_successful = env_loop(env, fetch, spawned_objects, logging.root)

   

if __name__ == "__main__":
    main()
