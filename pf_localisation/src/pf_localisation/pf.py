from geometry_msgs.msg import Pose, PoseArray, Quaternion
from . pf_base import PFLocaliserBase
import math
import rospy
import numpy as np

from . util import rotateQuaternion, getHeading
from random import random

import time

def rel_error(x, y):
    """Computes the relative error between 2 values or vectors."""
    x, y = np.array(x), np.array(y)
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def log_to_file(fname, msgs, append=False):
    """
    If append = False, it will create a new log file with name fname and content msgs
    If append = True, it will append the messages (msgs) to the end of the file called fname

    :Args:
        | fname (str): the name of the file
        | msgs (list): list of messages
        | append (bool): if True new file, if False append at the end of the file 
    """
    if append:
        mode = 'a'
    else:
        mode = 'w'
    with open(f'logs/{fname}', mode) as f:
        for msg in msgs:
            f.write(msg)

class PFLocaliser(PFLocaliserBase):
    """Extension of the base Particle Localisation class.

    This class fully implements the resampling algorithms for the
    particle filter using MCL and AMCL approaches.
    """

    def __init__(self):
        """Initializes the Particle Filter Localiser.

        This method assigns optimal values to achieve the best
        localization performance. Values have been explored and tested.

        Kidnap MCL attributes
        ---------------------
        * KIDNAP_THRESHOLD - threshold above which to spread particles
        * clustered - whether the particles are all together
        * ws_last_eval - `20` last max weights

        Odometry model parameters
        -------------------------
        * ODOM_ROTATION_NOISE - rotation noise
        * ODOM_TRANSLATION_NOISE - x axis (forward) noise
        * ODOM_DRIFT_NOISE - y axis (side-to-side) noise

        Sensor model parameters
        -----------------------
        * NUMBER_PREDICTED_READINGS - number of readings to predict
        """
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()

        self.ADAPTIVE = False
        
        # Efficiency
        self.UPDATE_IF_STOPPED = False
        self.NUM_SKIP_UPDATES = 2
        self.num_last_updates = 0
        self.last_scan = None

        # Kidnap MCL attributes
        self.KIDNAP_THRESHOLD = 100 if self.ADAPTIVE else 2000
        self.WS_LAST_FUNC = lambda x: np.mean(np.square(x))
        self.ws_last_eval = [-1] * 10
        self.clustered = True
        
        # ----- Set motion model parameters
        self.ODOM_ROTATION_NOISE = .1
        self.ODOM_TRANSLATION_NOISE = .15
        self.ODOM_DRIFT_NOISE = .25

        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 70 if self.ADAPTIVE else 200

        # ----- Log Parameters
        self.log_init_timev = 0

        # ----- Log Constants (Control which los you want to generate)
        self.log_weights = False
        self.log_probabilities = False
        self.log_loc_error_time = True
        self.log_loc_error_samples = False

    def calculate_loc_erro(self, x, y, new_heading):
        dif_x = x - self.prev_odom_x
        dif_y = y - self.prev_odom_y
        dif_heading = new_heading - self.prev_odom_heading
        
        distance_travelled = math.sqrt(dif_x*dif_x + dif_y*dif_y)
        direction_travelled = abs(math.atan2(dif_y, dif_x))

        return distance_travelled + direction_travelled

    def _get_noisy_pose(self, pose):
        """Adds noise to one particle to generate a new pose estimate.

        This method simply draws values from Gaussian distribution
        controlled by the mean of the pose and the variance of the
        initialized noise constants.

        Args:
            pose (geometry_msgs.msg.Pose): the original pose estimate
        Return:
            (geometry_msgs.msg.Pose): noisy pose of the particle
        """
        # Initialize pose estimate
        pose_hat = Pose()

        # Samples from Gaussian with variance based on sampling noise
        x_hat = np.random.normal(pose.position.x, self.ODOM_TRANSLATION_NOISE)
        y_hat = np.random.normal(pose.position.y, self.ODOM_DRIFT_NOISE)
        theta_hat = np.random.normal(scale=self.ODOM_ROTATION_NOISE)

        # Assign noisy pose estimates
        pose_hat.position.x = x_hat
        pose_hat.position.y = y_hat
        pose_hat.orientation = rotateQuaternion(pose.orientation, theta_hat)

        return pose_hat
    
    def _generate_random_poses(self, num_poses=None):
        """Generates random poses uniformly across the map.

        This method generates `num_poses` random poses within the
        boundaries of a map. All poses face randomly.

        Args:
            num_poses (int): the amount of poses to generate. If not
                provided, `self.NUMBER_PREDICTED_READINGS` is used.
        Return:
            (geometry_msgs.msg.PoseArray): random particle poses
        """
        # Set default value
        if num_poses is None:
            num_poses = self.NUMBER_PREDICTED_READINGS
        
        # Initialize random particle array
        poses_uniform = PoseArray()

        # Define map constants
        width = self.occupancy_map.info.width
        height = self.occupancy_map.info.height
        origin = self.occupancy_map.info.origin.position
        resolution = self.occupancy_map.info.resolution

        # While we don't have `num_poses` particles
        while len(poses_uniform.poses) < num_poses:
            y = np.random.randint(height)
            x = np.random.randint(width)

            # If the random position is within map boundaries
            if self.occupancy_map.data[y * width + x] != -1:
                pose = Pose()
                theta = np.random.uniform(-np.pi, np.pi)
                pose.position.x = x * resolution + origin.x
                pose.position.y = y * resolution + origin.y
                pose.orientation = rotateQuaternion(Quaternion(w=1), theta)

                poses_uniform.poses.append(pose)

        return poses_uniform

    def _resample_mcl_base(self, poses, ws):
        """Performs Stochastic Universal Resampling.

        This is the core resampling method for the MCL algorithm. All
        it does is simply goes through the CDF of weight distribution
        and resamples those particles with the highest weights.

        Note:
            Passed weights, i.e., `ws` must sum up to `1`.
        
        Args:
            poses (geometry_msgs.msg.PoseArray): the particle poses
            ws (list): the list of importance weights for each pose
        Return:
            (geometry_msgs.msg.PoseArray): resampled particle poses
        """
        # Initialize resampled particle array
        poses_resampled = PoseArray()

        # Initialize the algorithm
        i = 0
        M_inv = 1 / len(ws)
        cdf = np.cumsum(ws)
        us = [M_inv - np.random.uniform(0, M_inv)]

        # Filter by contributions of particles
        for _ in ws:
            while us[-1] > cdf[i]: i += 1
            us.append(us[-1] + M_inv)
            poses_resampled.poses.append(self._get_noisy_pose(poses.poses[i]))

        return poses_resampled

    def _resample_mcl(self, poses, ws):
        """Performs full MCL.

        The method utilizes helper function to resample particles in a
        Stochastic Universal way. It also deals with a kidnapped robot
        problem by monitoring a trailing sequence of max weight values
        and causing the particle cloud to spread if the average of the
        sequence is below `self.KIDNAP_THRESHOLD`.

        Args:
            poses (geometry_msgs.msg.PoseArray): the particle poses
            ws (list): the list of importance weights for each pose
        Return:
            (geometry_msgs.msg.PoseArray): resampled particle poses
        """
        # Update last weight evaluations with the most recent evaluation
        # self.ws_last_eval = [self.WS_LAST_FUNC(ws)] + self.ws_last_eval[:-1]

        # Keep track of robot's certainty of the pose
        rospy.loginfo(f"Certainty: {np.mean(self.ws_last_eval):.4f} | " + \
                      f"Threshold: {self.KIDNAP_THRESHOLD}")
        
        # If weights are low, dissolve particles
        if -1 not in self.ws_last_eval and self.clustered and \
           np.mean(self.ws_last_eval) <= self.KIDNAP_THRESHOLD:
            # Reinitialize the trailing of weight eval values 
            self.ws_last_eval = list(map(lambda _: -1, self.ws_last_eval))
            poses = self._generate_random_poses()
            self.clustered = False

            return poses
        
        # Usually particles are clustered
        if np.mean(self.ws_last_eval) > self.KIDNAP_THRESHOLD:
            self.clustered = True

        # Resample the particle cloud and add noise to it
        poses = self._resample_mcl_base(poses, ws)

        return poses

    def _resample_amcl_base(self, poses, ws):
        """Performs KLD adaptive resampling.

        This is the core resampling method for the AMCL algorithm.
        Here, a dynamic size of particlecloud is determined by KL
        distance measure while particles are being sampled from a
        weighted distribution.

        Note:
            Passed weights, i.e., `ws` must sum up to `1`.

        Args:
            poses (geometry_msgs.msg.PoseArray): the particle poses
            ws (list): the list of importance weights for each pose
        :Return:
            (geometry_msgs.msg.PoseArray): resampled particle poses
        """
        # Initialize resampled particle array
        poses_resampled = PoseArray()

        # KLD sampling initialization
        MAX_NUM_PARTICLES = 500
        eps = 0.08
        z = 0.99
        Mx = 0

        # Assure no bins are prerecorded
        self.histogram.non_empty_bins.clear()

        # While not min or KLD calculated samples reached
        while len(poses_resampled.poses) < Mx or \
              len(poses_resampled.poses) < self.NUMBER_PREDICTED_READINGS:
            # Sample random pose, add it to resampled list
            pose = np.random.choice(poses.poses, p=ws)
            pose = self._get_noisy_pose(pose)
            poses_resampled.poses.append(pose)
            
            # If the pose falls into empty bin
            if self.histogram.add_if_empty(pose):
                # Number of current non-empty bins
                k = len(self.histogram.non_empty_bins)

                # Update KL distance
                if k > 1:
                    Mx = ((k - 1) / (2 * eps)) * \
                         math.pow(1 - (2 / (9 * (k - 1))) + \
                         math.sqrt(2 / (9 * (k - 1))) * z, 3)
                
                # Don't exceed the maximum allowed range
                Mx = MAX_NUM_PARTICLES if Mx > MAX_NUM_PARTICLES else Mx

        # Keep track of num particles generated
        rospy.loginfo(f"Generated {len(poses_resampled.poses)} particles")

        return poses_resampled


    def _resample_amcl(self, poses, ws):
        """Performs full AMCL (KLD version) resampling.

        The method utilizes helper function to resample particles in an
        adaptive KLD-based way.
        
        Args:
            poses (geometry_msgs.msg.PoseArray): the particle poses
            ws (list): the list of importance weights for each pose
        Return:
            (geometry_msgs.msg.PoseArray): resampled particle poses
        """
        SPREAD_THRESHOLD = 150
        
        if not self.clustered:
            poses = self._resample_mcl_base(poses, ws)
        
        poses = self._resample_amcl_base(poses, ws)

        # Keep track of robot's certainty of the pose
        rospy.loginfo(f"Certainty: {np.mean(self.ws_last_eval):.4f} | " + \
                      f"Threshold: {self.KIDNAP_THRESHOLD}")

        # If weights are low, dissolve particles
        if len(poses.poses) > SPREAD_THRESHOLD and \
           -1 not in self.ws_last_eval and self.clustered and \
           np.mean(self.ws_last_eval) <= self.KIDNAP_THRESHOLD:
            # Reinitialize the trailing of weight eval values 
            self.ws_last_eval = list(map(lambda _: -1, self.ws_last_eval))
            poses = self._generate_random_poses(len(poses.poses))
            self.clustered = False

            return poses
        
        if len(poses.poses) <= SPREAD_THRESHOLD:
            self.clustered = True
            
        return poses


    def _filter_naive(self, estimates, num_keep=None):
        """Naively filters the best (closest to mean) value estimates.

        Args:
            estimates (list): list of values which estimate the pose
            num_keep (int):
        Return:
            (list): list of best estimates based on `num_keep`
        """
        # Num best particles to keep
        if num_keep is None:
            num_keep = self.NUMBER_PREDICTED_READINGS // 2
        
        # Sorted indeces of `estimates - mean`
        ind = np.argsort(np.abs(estimates - np.mean(estimates)))

        return np.take(estimates, ind)[:num_keep]


    def initialise_particle_cloud(self, initialpose):
        """
        Set particle cloud to initialpose plus noise

        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        self.particlecloud can be initialised here. Initial pose of the robot
        is also set here.
        
        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """        
        # Map parameters
        width = self.occupancy_map.info.width
        height = self.occupancy_map.info.height
        origin = self.occupancy_map.info.origin.position
        resolution = self.occupancy_map.info.resolution

        # Initialize histogram based on map parameters
        self.histogram = self._Histogram(width * resolution,
                                         height * resolution, origin)
        
        # Shorthand for initial pose
        pose0 = initialpose.pose.pose

        # Check the initial pose
        rospy.loginfo("Initial pose: " + \
                      f"[ x {pose0.position.x:.4f} " + \
                      f"| y {pose0.position.y:.4f} " + \
                      f"| theta {getHeading(pose0.orientation):.4f} ]")

        # Set up initial pose array
        initial_poses = PoseArray()

        # Generate a list of noisy particles
        for _ in range(self.NUMBER_PREDICTED_READINGS):
            initial_poses.poses.append(self._get_noisy_pose(pose0))

        self.log_init_time = time.time()

        return initial_poses

 
    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update
        """
        # If the sensor data is the same, don't update particle cloud
        if not self.UPDATE_IF_STOPPED and self.last_scan is not None:
            if rel_error(scan.ranges, self.last_scan.ranges) < 1e-9:
                return

        # Save the previous scan and increment num last updates
        self.last_scan = scan
        self.num_last_updates += 1

        # Skip an update every `NUM_SKIP_UPDATES` times for efficiency
        if self.num_last_updates <= self.NUM_SKIP_UPDATES:
            return
        else:
            self.num_last_updates = 0

        # Generate importance weights based on scan readings
        ws = [self.sensor_model.get_weight(scan, pose)
              for pose in self.particlecloud.poses]

        # Update last weight evaluations with the most recent evaluation
        self.ws_last_eval = [self.WS_LAST_FUNC(ws)] + self.ws_last_eval[:-1]

        # Log Weights (Used for Kidnapped Problem)
        if self.log_weights:
            log_list = []
            log_list.append(f"Max: {np.max(ws):.4f}\n")
            log_list.append(f"Min: {np.min(ws):.4f}\n")
            log_list.append(f"Mean: {np.mean(ws):.4f}\n")
            for i in range(0, len(ws)):
                log_list.append(f"[{i}]: {ws[i]:.4f}\n")
            log_to_file(f"weights.log", log_list, append=False)

        # Weights should sum up to 1
        ws /= np.sum(ws)

        # Resample either using MCL or AMCL algorithm
        resample = self._resample_amcl if self.ADAPTIVE else self._resample_mcl
        self.particlecloud = resample(self.particlecloud, ws)

        # Log Probabilties
        if self.log_probabilities:
            log_list = ["Range No. | Obs. Range | Pre. Range | Probability\n"]
            j = 0
            pose = self.estimate_pose()
            for i, obs_bearing in self.sensor_model.reading_points:
                # ----- For each range...
                obs_range = scan.ranges[i]
                
                # ----- Laser reports max range as zero, so set it to range_max
                if (obs_range <= 0.0):
                    obs_range = self.sensor_model.scan_range_max 
                
                # ----- Compute the range according to the map
                map_range = self.sensor_model.calc_map_range(pose.position.x, pose.position.y,
                                        getHeading(pose.orientation) + obs_bearing)
                pz = self.sensor_model.predict(obs_range, map_range)
                log_list.append(f"[{j}]: {obs_range} {map_range} {pz}\n")
                j += 1

            log_to_file("probabilities.log", log_list)


    def estimate_pose(self):
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        
        Create new estimated pose, given particle cloud
        E.g. just average the location and orientation values of each of
        the particles and return this.
        
        Better approximations could be made by doing some simple clustering,
        e.g. taking the average location of half the particles after 
        throwing away any which are outliers

        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
        """
        # Initialize estimated position
        pose_hat = Pose()

        # Initialize lists of pose estimates
        pxs, pys, oxs, oys, ozs, ows = [], [], [], [], [], []

        # Generate lists for each pose estimate
        for pose in self.particlecloud.poses:
            pxs.append(pose.position.x)
            pys.append(pose.position.y)
            oxs.append(pose.orientation.x)
            oys.append(pose.orientation.y)
            ozs.append(pose.orientation.z)
            ows.append(pose.orientation.w)

        # Filter the best estimates and average the result
        pose_hat.position.x = np.mean(self._filter_naive(pxs))
        pose_hat.position.y = np.mean(self._filter_naive(pys))
        pose_hat.orientation.x = np.mean(self._filter_naive(oxs))
        pose_hat.orientation.y = np.mean(self._filter_naive(oys))
        pose_hat.orientation.z = np.mean(self._filter_naive(ozs))
        pose_hat.orientation.w = np.mean(self._filter_naive(ows))

        # Localization Error based on Time
        if self.log_loc_error_time:
            log_list = []
            t = time.time() - self.log_init_time
            error_total = self.calculate_loc_erro(pose_hat.position.x, pose_hat.position.y, pose_hat.orientation.z)
            log_list.append(f"{t} {error_total}\n")
            log_to_file("loc_error_time_our_amcl.log", log_list, append=True)

        # Localization Error based on Number of Samples
        if self.log_loc_error_samples:
            log_list = []
            t = time.time() - self.log_init_time
            error_total = self.calculate_loc_erro(pose_hat.position.x, pose_hat.position.y, pose_hat.orientation.z)
            log_list.append(f"{t} {error_total}\n")
            log_to_file("loc_error_samples_400.log", log_list, append=True)

        return pose_hat

    class _Histogram():
        """
        Histogram inner class used for managing bins.

        There are 3 types of bins: in x axis, in y axis and in theta
        axis. The bins can be thought of as cells for a volume where
        particles belong to based on their parameters being within a
        specific interval.
        """
        def __init__(self, width, height, origin, num_bins=(400, 400, 1)):
            """Initializes the histogram object.

            Args:
                width (float): the width of the map
                height (float): the height of the map
                origin ()
                num_bins (tuple): the number of bins in each dimension
            """
            # Number of bins in each dimension
            size_x, size_y, size_theta = num_bins

            # Get size intervals
            self.x_bins = np.linspace(origin.x, origin.x + width, size_x + 1)
            self.y_bins = np.linspace(origin.y, origin.y + height, size_y + 1)
            self.theta_bins = np.linspace(-np.pi, np.pi, size_theta + 1)

            # Keep track of requests for particles belonging to bins
            self.non_empty_bins = []


        def add_if_empty(self, pose):
            """
            Adds a bin to a `non_empty_bins` list if passed pose belongs
            to that bin.
            
            Args:
                pose (geometry_msgs.msg.Pose): the particle pose
            Return:
                (bool): `True` if bin is empty, `False` otherwise
            """
            # Get x, y and theta
            x = pose.position.x
            y = pose.position.y
            theta = getHeading(pose.orientation)
            
            # Find the corresponding bin (interval)
            bin = (np.searchsorted(self.x_bins, x), 
                   np.searchsorted(self.y_bins, y),
                   np.searchsorted(self.theta_bins, theta))
            
            # Add to non-empty list if such bin is not there
            if bin in self.non_empty_bins:
                return False
            else:
                self.non_empty_bins.append(bin)
                return True