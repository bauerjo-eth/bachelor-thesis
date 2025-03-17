# This is a special node that subscribe to a topic, but also send request to a client. 
# To avoid dreadlocks, I used this example https://github.com/ros2/examples/blob/humble/rclpy/services/minimal_client/examples_rclpy_minimal_client/client_async_callback.py
# All poses sent to the controller should be in the tool frame of the robot, the controller will return global increments suitable for moveit

import rclpy
from rclpy.node import Node, MutuallyExclusiveCallbackGroup
from rclpy.clock import ROSClock
from rclpy.parameter import Parameter

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, LivelinessPolicy, HistoryPolicy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo

from us_msg.srv import InverseKinematics, SystemCmds

from us_utils.pose import Pose as PoseStruct
from us_utils.controller import PoseControl, SystemState, SystemCMD

from scipy.spatial.transform import Rotation

from copy import deepcopy
import numpy as np

import time

import open3d as o3d
import open3d.core as o3c
from cv_bridge import CvBridge

from us_utils.camera_utils import intrinsics_tensor_from_msg, signed_vector_angle, cameraPose_to_extrinsics



CAMERA_ACTIVE = True

sampling_rate = 0.005

class US(Node):

    def __init__(self, position_stiffnesses:float):
        super().__init__('us_node')

        # QoS Settings
        qos_settings = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,  # Keep only the last message
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Prioritize latency over delivery guarantee
            durability=DurabilityPolicy.VOLATILE,  # Don't store messages
            liveliness=LivelinessPolicy.AUTOMATIC,
            deadline=rclpy.duration.Duration(seconds=0.005)  # 5 ms deadline
        )

        
        # Parameters
        self.declare_parameter("applied_force", 6.0)
        self.declare_parameter("max_position_increments_X", 0.0)
        self.declare_parameter("max_position_increments_Y", 0.0)
        self.declare_parameter("max_position_increments_Z", 0.0)
        self.declare_parameter("max_force_position_increment", 0.0)
        self.declare_parameter("max_orientation_increment", 0.0)
        
        # Callback & Events
        self.cmd_server_callback = MutuallyExclusiveCallbackGroup()
        self.timer_callback = MutuallyExclusiveCallbackGroup()
        self.subscriber_eeCartesianPoses = MutuallyExclusiveCallbackGroup()
        self.IK_controller_callback = MutuallyExclusiveCallbackGroup()
        self.cmd_publisher_callback = MutuallyExclusiveCallbackGroup()
        self.pcd_subscriber_callback = MutuallyExclusiveCallbackGroup()

        # Subscribers
        self.subscription_eeCartesianPoses = self.create_subscription(
            Pose,
            'measured_cartesian_pose',
            self.update_eeCartesianPoses,
            qos_settings,
            callback_group=self.subscriber_eeCartesianPoses)
        self.subscription_eeCartesianPoses
        self.subscription_jointPoses = self.create_subscription(
            JointState,
            'measured_joint_states',
            self.update_jointPoses,
            qos_settings)
        self.subscription_jointPoses 

        # Service
        self.cmd_service = self.create_service(SystemCmds, 'cmd_trajectory', self.cmd_received, callback_group=self.cmd_server_callback)

        # Publisher 
        self.jointPose_publisher = self.create_publisher(JointState, 'cmd_joint_states', qos_settings, callback_group=self.cmd_publisher_callback)
        
        # Clients
        # IK
        self.inverseKinematics_client = self.create_client(InverseKinematics, 'IK_solver', callback_group=self.IK_controller_callback)
        while not self.inverseKinematics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.inverseKinematics_client_request = InverseKinematics.Request()

        # Get intial pose values
        self.trajectory = np.array([])
        self.eePose_received = False
        self.jointPoses_received = False

        # Init structures 
        self.system_state = SystemState()
        self.system_state = SystemState()
        self.system_cmd = SystemCMD()
        self.pose_increment = PoseStruct()            

        # Init controller
        self.pose_controller = PoseControl(position_stiffnesses)

        self.trajectory_in_progress = False

        # Start timer
        self.fsm_timer = self.create_timer(sampling_rate, self.fsmTimer_callback, callback_group=self.timer_callback)

        ###################################################################
        ## PCD processing for the camera orientation
        ###################################################################
        if CAMERA_ACTIVE:
            self.bridge = CvBridge()

            # Init subscribers
            self.sub_depth = self.create_subscription(msg_Image, '/cam/rl/aligned_depth_to_color/image_raw', self.imageDepthCallback, 1)
            self.sub_depth_info = self.create_subscription(CameraInfo, '/cam/rl/aligned_depth_to_color/camera_info', self.imageDepthInfoCallback, 1)
            self.sub_color = self.create_subscription(msg_Image, '/cam/rl/color/image_rect_raw', self.imageColorCallback, 1)
            self.sub_color_info = self.create_subscription(CameraInfo, '/cam/rl/color/camera_info', self.imageColorInfoCallback, 1)
            # Stupid other variables
            self.x_unit = np.array([1.0, 0.0, 0.0])
            self.y_unit = np.array([0.0, 1.0, 0.0])
            self.z_unit = np.array([0.0, 0.0, 1.0])

            self.device = o3d.core.Device("CUDA:0")
            self.depth_scale = 1000.0
            self.depth_max = 0.5
            self.voxel_size = 0.01
            self.height_pixels = (0, 480)
            self.width_pixels = (0, 566)

            self.bounding_box_extend = o3d.core.Tensor([0.05, 0.10, 0.25], o3d.core.Dtype.Float64, self.device)

            self.reset_angle_manager()
            self.timer_integrate = self.create_timer(0.5, self.timer_integrate_callback)

            # Necessary transformations
            self.us_to_camera = np.eye(4)  # Create identity matrix
            self.us_to_camera[:3, :3] = Rotation.from_rotvec([0.0, 0.0, np.pi]).as_matrix()
            self.us_to_camera[:3, 3] = -np.array([0.0, 0.0, 0.225]) + np.array([0.079, 0.0, 0.070])  # us tip to kuka attach, to camera

        self.get_logger().info('us server started')

    def cmd_received(self, request, response):
        self.get_logger().info("Received cmd_trajectory")

        if self.eePose_received:
            # Extract the trajectory
            self.get_logger().info("Processing the trajectory")
            self.trajectory = []
            for tr_msg in request.trajectory:
                cmd = SystemCMD()
                cmd.from_msg(tr_msg)

                self.trajectory.append(cmd)

            self.trajectory_idx = 0
            self.trajectory_size = len(self.trajectory)

            self.update_cmd(self.trajectory[self.trajectory_idx])
            self.trajectory_in_progress = True

            self.get_logger().info("Finished processing the trajectory, starting motion")

            while self.trajectory_in_progress:
                rclpy.spin_once(self)

            self.get_logger().info("Trajectory finished")
            response.success = True

        else:
            response.success = False
            self.get_logger().info("Did not receive the initial pose")

        return response

    def update_eeCartesianPoses(self, msg):
        # self.get_logger().info(f"Received Cartesian Pose")
        self.system_state.measured_pose = PoseStruct()
        self.system_state.measured_pose.from_msg(msg, 0.001)
        # First time, initialize the current value too
        if not self.eePose_received:
            self.system_state.set_pose = deepcopy(self.system_state.measured_pose)
            self.eePose_received = True

    def update_jointPoses(self, msg):
        # self.set_joint_state = msg.position
        if not self.jointPoses_received:
            self.set_joint_state = msg.position
            self.jointPoses_received = True

    async def fsmTimer_callback(self):
        if self.trajectory_in_progress:
            if self.eePose_received and self.jointPoses_received:
                self.get_logger().info("trajectory_idx: " + str(self.trajectory_idx))
                if self.trajectory_idx < self.trajectory_size:
                    # Update the system_cmd from ros parameters
                    self.update_system_cmd_parameters()

                    # Update the systeme_cmd rotation with the camera orientation
                    if CAMERA_ACTIVE and self.system_cmd.orientation_from_camera:
                        # Get future position and orientation
                        # Remove force control from system_cmd
                        noForce_system_cmd = deepcopy(self.system_cmd)
                        noForce_system_cmd.applied_force = -1.0
                        # Use a system state with no Z offset due to the force control
                        noForce_system_state = deepcopy(self.system_state)
                        noForce_system_state.set_pose.position = self.system_state.measured_pose.position
                        # Predict the displacement only in position and rotation
                        _, future_pose_increment = self.pose_controller.pose_control(noForce_system_state, noForce_system_cmd)
                        future_pose = self.pose_controller.add_poses(self.system_state.measured_pose, future_pose_increment)
                        # Send request to the camera orientation service
                        future_orientation_result = self.get_future_orientation_from_pcd(future_pose)
                        system_cmd_camera = deepcopy(self.system_cmd)
                        system_cmd_camera.goal_pose.orientation = future_orientation_result
                        reached, self.pose_increment = self.pose_controller.pose_control(self.system_state, system_cmd_camera)
                    else:
                        # Calculate the position increments
                        reached, self.pose_increment = self.pose_controller.pose_control(self.system_state, self.system_cmd)
          
                    if reached:
                        self.trajectory_idx += 1
                        if self.trajectory_idx < self.trajectory_size:
                            self.update_cmd(self.trajectory[self.trajectory_idx])
                    else:
                        await self.controller()

                else:
                    self.reset_increments()
                    self.trajectory_in_progress = False
            else:
                self.get_logger().info("Did not receive the initial pose")
                
    def update_cmd(self, cmd: SystemCMD):
        self.reset_increments()
        self.system_cmd = cmd
        self.set_parameters([
                Parameter('applied_force', Parameter.Type.DOUBLE, cmd.applied_force),
                Parameter('max_position_increments_X', Parameter.Type.DOUBLE, cmd.max_local_position_increments[0]),
                Parameter('max_position_increments_Y', Parameter.Type.DOUBLE, cmd.max_local_position_increments[1]),
                Parameter('max_position_increments_Z', Parameter.Type.DOUBLE, cmd.max_local_position_increments[2]),
                Parameter('max_force_position_increment', Parameter.Type.DOUBLE, cmd.max_force_position_increment),
                Parameter('max_orientation_increment', Parameter.Type.DOUBLE, cmd.max_orientation_increment)])
        
        # Set the goal poste relative to the current end-effector position
        if self.system_cmd.position_coordinate_frame == "local":
            self.system_cmd.goal_pose.position = self.system_state.set_pose.position + self.system_state.set_pose.orientation.apply(cmd.goal_pose.position)
        elif self.system_cmd.position_coordinate_frame == "global":
            self.system_cmd.goal_pose.position = self.system_state.set_pose.position + cmd.goal_pose.position

        if self.system_cmd.orientation_coordinate_frame == "local":
            self.system_cmd.goal_pose.orientation = self.system_state.set_pose.orientation * cmd.goal_pose.orientation

        self.pose_controller.reached.reset_reachedFlags()

        # if CAMERA_ACTIVE and self.system_cmd.reset_surface_angles_storage:
        #     self.angle_manager.reset_surface()

    def update_system_cmd_parameters(self): 
        X_increments = self.get_parameter('max_position_increments_X').get_parameter_value().double_value
        Y_increments = self.get_parameter('max_position_increments_Y').get_parameter_value().double_value
        Z_increments = self.get_parameter('max_position_increments_Z').get_parameter_value().double_value
        self.system_cmd.max_local_position_increments = np.array([X_increments, Y_increments, Z_increments])
        self.system_cmd.max_orientation_increment = self.get_parameter('max_orientation_increment').get_parameter_value().double_value
        self.system_cmd.applied_force = self.get_parameter("applied_force").get_parameter_value().double_value
        self.system_cmd.max_force_position_increment = self.get_parameter("max_force_position_increment").get_parameter_value().double_value

    def reset_increments(self):
        self.pose_increment = PoseStruct()

    async def controller(self):
        new_pose = self.pose_controller.add_poses(self.system_state.set_pose, self.pose_increment)

        jointState_result = await self.IK_client(new_pose.to_msg())

        if jointState_result.ik_joint_angles != []:
            cmd = JointState()
            cmd.header.stamp = ROSClock().now().to_msg()
            cmd.name = ["lbr_joint_0", "lbr_joint_1", "lbr_joint_2", "lbr_joint_3", "lbr_joint_4", "lbr_joint_5", "lbr_joint_6"]
            cmd.effort = [0.0] * 7
            cmd.position = jointState_result.ik_joint_angles
            self.jointPose_publisher.publish(cmd)

            # Update poses
            self.system_state.set_pose = new_pose
            self.set_joint_state = jointState_result.ik_joint_angles
           
            return True
        else:
            self.get_logger().info("Did not receive IK")
            return False
        
    async def IK_client(self, desired_pose):
        # Convert cartesian pose to joint space
        self.inverseKinematics_client_request.joint_angles = self.set_joint_state
        self.inverseKinematics_client_request.poses_request = desired_pose
        future = self.inverseKinematics_client.call_async(self.inverseKinematics_client_request)
        reponse = await future
        return reponse

    def get_future_orientation_from_pcd(self, future_pose: PoseStruct)->Rotation:
        if self.pcd_surface.is_empty():
            self.get_logger().warn("The point cloud is empty. Returning the original orientation.")
            return future_pose.orientation

        future_pose_position_tensor = o3d.core.Tensor(future_pose.position, o3d.core.Dtype.Float64, self.device)
        future_pose_orientation_tensor = o3d.core.Tensor(future_pose.orientation.as_matrix(), o3d.core.Dtype.Float64, self.device)
        bounding_box = o3d.t.geometry.OrientedBoundingBox(future_pose_position_tensor, future_pose_orientation_tensor, self.bounding_box_extend)

        # Crop the point cloud to the bounding box in a thead-safe way
        try:
            pcd_below_US = self.pcd_surface.crop(bounding_box)
        except IndexError as e:
            self.get_logger().warn("Bounding box is out of bounds: " + str(e))
            return future_pose.orientation

        # Extract the normals as numpy array
        normals = pcd_below_US.point.normals.cpu().numpy()
        surface_normal = np.mean(normals, axis=0)
        if np.isnan(surface_normal).any():
            self.get_logger().warn("There are NaN values in the surface normal. Returning the original orientation.")
            return future_pose.orientation
        surface_normal /= np.linalg.norm(surface_normal)  # Normalize the vector
        surface_normal = -surface_normal  # Invert the normal to point opposite to the camera

        # Rotate around the local X-axis to align the YZ projection
        surface_normal_local = future_pose.orientation.inv().apply(surface_normal)
        surface_normal_yz = np.array([0, surface_normal_local[1], surface_normal_local[2]])

        # Calculate the angle between z and surface_normal_yz
        angle_x = signed_vector_angle(self.z_unit, surface_normal_yz, self.x_unit)
        if np.abs(angle_x) < np.deg2rad(1.0):
            rotation_x = Rotation.identity()
        else:
            rotation_x = Rotation.from_rotvec(self.x_unit * angle_x)

        # start_time = time.time()
        # Rotate around the local Y-axis to align with v2
        surface_normal_local_2 = rotation_x.inv().apply(surface_normal_local)
        surface_normal_xz = np.array([surface_normal_local_2[0], 0, surface_normal_local_2[2]])

        # Calculate the angle between v1_rotated_xz and v2_xz
        angle_y = signed_vector_angle(self.z_unit, surface_normal_xz, self.y_unit)
        if np.abs(angle_y) < np.deg2rad(1.0):
            rotation_y = Rotation.identity()
        else:
            rotation_y = Rotation.from_rotvec(self.y_unit * angle_y)

        # Apply this second rotation to the rotated v1
        final_rotation = rotation_y * rotation_x

        # Back to the global frame
        future_orientation = future_pose.orientation * final_rotation

        return future_orientation
    
    def imageColorCallback(self, data):
        self.cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        self.cv_image = np.ascontiguousarray(self.cv_image[self.height_pixels[0]:self.height_pixels[1]+1, 
                                                           self.width_pixels[0]:self.width_pixels[1]+1])
        
    def imageDepthCallback(self, data):
        self.cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
        self.cv_depth = np.ascontiguousarray(self.cv_depth[self.height_pixels[0]:self.height_pixels[1]+1, 
                                                           self.width_pixels[0]:self.width_pixels[1]+1])

    def imageDepthInfoCallback(self, cameraInfo):
        if self.depth_intrinsics is None:
            self.get_logger().info("Depth intrinsics received.")
            self.depth_intrinsics = intrinsics_tensor_from_msg(cameraInfo, self.width_pixels, self.height_pixels)

    def imageColorInfoCallback(self, cameraInfo):
        if self.color_intrinsics is None:
            self.get_logger().info("Color intrinsics received.")
            self.color_intrinsics = intrinsics_tensor_from_msg(cameraInfo, self.width_pixels, self.height_pixels)
    
    def init_done(self):
        return self.depth_intrinsics is not None and self.color_intrinsics is not None and self.cv_depth is not None and self.cv_image is not None and self.eePose_received

    def timer_integrate_callback(self):
        if self.init_done():
            camera_position = self.system_state.measured_pose.transformation_matrix() @ self.us_to_camera
            self.integrate(camera_position)
            self.pcd_surface = self.vbg.extract_point_cloud()
            if self.pcd_surface.is_empty():
                self.get_logger().warn("The point cloud is empty.")
                return
            else:
                self.pcd_surface.estimate_normals(30, 1.4*self.voxel_size)
            self.pcd_surface.orient_normals_towards_camera_location(o3d.core.Tensor(camera_position[:3, 3], o3d.core.Dtype.Float64, self.device))

    def integrate(self, camera_position: np.ndarray):
        # https://www.open3d.org/docs/release/tutorial/t_reconstruction_system/integration.html
        # Do not start before all values get initialized
        if not self.init_done():
            return
        # Convert the depth image, extrinsics and intrinsics to Open3D tensor
        depth = o3d.t.geometry.Image(self.cv_depth).to(self.device)
        extrinsic = cameraPose_to_extrinsics(camera_position)
        frustum_block_coords = self.vbg.compute_unique_block_coordinates(
            depth, self.depth_intrinsics, extrinsic, self.depth_scale,
            self.depth_max)
        
        color = o3d.t.geometry.Image(self.cv_image).to(self.device)
        self.vbg.integrate(frustum_block_coords, depth, color, self.depth_intrinsics,
                           self.color_intrinsics, extrinsic, self.depth_scale,
                           self.depth_max)
        

    
    def reset_angle_manager(self):
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=self.voxel_size,
            block_resolution=16,
            block_count=5000,
            device=self.device)
        self.pcd_surface = o3d.t.geometry.PointCloud()
        
        self.depth_intrinsics = None
        self.color_intrinsics = None
        self.cv_depth = None
        self.cv_image = None
        self.received_measured_pose = False

    

def main(args=None):
    rclpy.init(args=args)


    # Position stiffnesses
    position_stiffnesses = 100.0
    us_process = US(position_stiffnesses)
    rclpy.spin(us_process)

    us_process.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
            