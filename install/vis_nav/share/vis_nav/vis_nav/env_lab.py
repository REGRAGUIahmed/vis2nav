#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:17:01 2023

@author: oscar
"""

import os
from os import path
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
import time
import math
import math
import random
import numpy as np
from numpy import inf
from collections import deque
from squaternion import Quaternion
import rclpy
import cv2
from cv_bridge import CvBridge
from gazebo_msgs.msg import EntityState
#import rospy
import subprocess
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
#import sensor_msgs.point_cloud2 as pc2
import point_cloud2 as pc2
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from rclpy.node import Node
from gazebo_msgs.msg import ModelState
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose

#import ros_numpy
last_odom = None
last_image = None
last_dist = None
scan_data =None
velodyne_data = np.ones(20) * 10 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goalOK = False

    if -6.7 < x < -6.5 and -3.4 < y < 3.6:
        goalOK = True
    
    elif 3.5 < x < 5.2 and -3.4 < y < 3.6:
        goalOK = True
        
    elif -6.4 < x < 3.5 and -3.4 < y <-2.8:
        goalOK = True
    
    elif -6.4 < x < 3.5 and 3.1 < y < 3.6:
        goalOK = True
    
    elif -1.8 < x < -1.4 and -3.4 < y < 3.6:
        goalOK = True

    return goalOK


# Function to put the laser data in bins
def binning(lower_bound, data, quantity):
    width = round(len(data) / quantity)
    quantity -= 1
    bins = []
    for low in range(lower_bound, lower_bound + quantity * width + 1, width):
        bins.append(min(data[low:low + width]))
    return np.array([bins])

#launchfile, ROS_MASTER_URI, height, width, nchannels
class GazeboEnv(Node):
    """Superclass for all Gazebo environments.
    """

    def __init__(self):
        super().__init__('env') 
        self.entity_name = 'goal'
        self.entity_dir_path='/home/regmed/dregmed/vis_to_nav/src/vis_nav/description/sdf'
        self.entity_path = os.path.join(self.entity_dir_path, 'obstacle.sdf')
        self.entity = open(self.entity_path, 'r').read()
        self.odomX = 0.0
        self.odomY = 0.0

        self.goalX = 1.0
        self.goalY = 0.0
        self.angle = 0.0
        self.upper = 10.0
        self.lower = -10.0
        # Changement de la position initial de robot mobile 
        self.model_name = 'td_robot'
        self.model_dir_path = '/home/regmed/dregmed/vis_to_nav/install/vis_nav/share/vis_nav/models/td_robot'
        self.model_path = os.path.join(self.model_dir_path, 'td_robot.sdf')
        self.model = open(self.model_path, 'r').read()
        self.modelX = 3.0
        self.modelY = 3.0
        self.orientationX = 0.0
        self.orientationY = 0.0
        self.orientationZ = 0.0
        self.orientationW = 1.0 
        #self.velodyne_data = np.ones(20) * 10
        #self.last_laser = None
        #self.last_odom = None
        #self.last_image = None
        #self.last_image_fish = None
        self.rgb_image = None
        self.original_image = None
        self.br = CvBridge()
        self.collision = 0.0
        self.last_act = 0.0
        self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
        self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
        self.x_pos_list = deque(maxlen=5)
        self.y_pos_list = deque(maxlen=5)

        '''self.set_self_state = EntityState()
        self.set_self_state.name = 'my_robot'
        self.set_self_state.pose.position.x = 4.0
        self.set_self_state.pose.position.y = 3.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0'''
        self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
        self.gaps = [[-1.6, -1.57 + 3.14 / 20]]
        for m in range(19):
            self.gaps.append([self.gaps[m][1], self.gaps[m][1] + 3.14 / 20])
        self.gaps[-1][-1] += 0.03

        '''try:
            subprocess.Popen(["roscore", "-p", ROS_MASTER_URI])
        except OSError as e:
            raise e

        print("Roscore launched!")'''

        # Launch the simulation with gym initialization
        '''rospy.init_node('gym', anonymous=True)'''

        '''if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join('/home/oscar/ws_oscar/DRL-Transformer-SimtoReal-Navigation/catkin_ws/src/gtrl/launch', launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        time.sleep(10)

        process = subprocess.Popen(["roslaunch", "-p", ROS_MASTER_URI, fullpath])
        print("Gazebo launched!")'''

        self.gzclient_pid = 0

        # Set up the ROS publishers and subscribers
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        #self.vel_pub = rospy.Publisher('/scout/cmd_vel', Twist, queue_size=1)
        self.set_state = self.create_publisher(EntityState, "gazebo/set_entity_state", 10)
        #self.set_state = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=10)
        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        #self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        #self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        #self.reset_proxy = self.create_client(Empty, "/reset_world")
        #self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        topic = 'vis_mark_array'
        self.publisher = self.create_publisher(MarkerArray, topic, 3)
        #self.publisher = rospy.Publisher(topic, MarkerArray, queue_size=3)
        topic2 = 'vis_mark_array2'
        self.publisher2 = self.create_publisher(MarkerArray, topic2, 3)
        #self.publisher2 = rospy.Publisher(topic2, MarkerArray, queue_size=1)
        topic3 = 'vis_mark_array3'
        self.publisher3 = self.create_publisher(MarkerArray, topic3, 3)
        #self.publisher3 = rospy.Publisher(topic3, MarkerArray, queue_size=1)
        topic4 = 'vis_mark_array4'
        self.publisher4 = self.create_publisher(MarkerArray, topic4, 3)
        self.reset_proxy = self.create_client(Empty, "/reset_world")
        #self.publisher4 = rospy.Publisher(topic4, MarkerArray, queue_size=1)
        #self.velodyne = self.create_subscription(PointCloud2, "/velodyne_points", self.velodyne_callback, 1)
        #self.velodyne = rospy.Subscriber('/velodyne_points', PointCloud2, self.velodyne_callback, queue_size=1)
        #self.laser = self.create_subscription(LaserScan, "/front_laser/scan", self.laser_callback, 1)
        #self.laser = rospy.Subscriber('/front_laser/scan', LaserScan, self.laser_callback, queue_size=1)
        #self.odom = self.subscription = self.create_subscription( Odometry, '/odom', self.odom_callback, 1)
        #self.odom = rospy.Subscriber('/scout/odom', Odometry, self.odom_callback, queue_size=1)
        #self.image = self.create_subscription(Image,'/camera/image_raw',self.image_callback, qos_profile_sensor_data)
        #self.image = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback, queue_size=1)
        #self.image_fish = rospy.Subscriber('/camera/fisheye/image_raw', Image, self.image_fish_callback, queue_size=1)

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)


    # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # range as state representation
    '''def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(20) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])  # * -1
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break'''

    '''def laser_callback(self, scan):
        self.last_laser = scan

    def odom_callback(self, od_data):
        self.last_odom = od_data
        

    def image_callback(self, rgb_data):
        image = self.br.imgmsg_to_cv2(rgb_data, "mono8")
        ######## Depth Image ##########
        # image = self.br.imgmsg_to_cv2(rgb_data, "passthrough")
        ######## RGB Image #########
        #self.last_image = np.expand_dims(cv2.resize(image, (128, 64)), axis=2)
        self.last_image = np.expand_dims(cv2.resize(image, (160, 128)), axis=2)

    def image_fish_callback(self, rgb_data):
        image = self.br.imgmsg_to_cv2(rgb_data, "mono8")
        self.original_image = self.br.imgmsg_to_cv2(rgb_data, "rgb8")
        self.last_image_fish = np.expand_dims(cv2.resize(image[80:400, 140:500], (160, 128)), axis=2)
        image_ = self.br.imgmsg_to_cv2(rgb_data, "rgb8")
        self.rgb_image = image_[80:400, 140:500, :]'''

    # Detect a collision from laser datav def spawn_model(self):
    
    def spawn_model(self,modelX,modelY,orientationX,orientationY,orientationZ,orientationW):
        #env.get_logger().info("spawn_entity **********************")
        model_pose1 = Pose()
        model_pose1.position.x = modelX
        model_pose1.position.y = modelY
        model_pose1.orientation.x = orientationX
        model_pose1.orientation.y = orientationY
        model_pose1.orientation.z = orientationZ
        model_pose1.orientation.w = orientationW

        # Create the service request
        req_s1 = SpawnEntity.Request()
        req_s1.name = self.model_name
        req_s1.xml = self.model
        req_s1.initial_pose = model_pose1
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.spawn_entity_client.call_async(req_s1) 
        self.get_logger().info(f'spawn Model at New position : x={modelX}, y={modelY}')

    def spawn_entity(self):
        #env.get_logger().info("spawn_entity **********************")
        goal_pose1 = Pose()
        goal_pose1.position.x = self.goalX
        goal_pose1.position.y = self.goalY
        req_s = SpawnEntity.Request()
        req_s.name = self.entity_name
        req_s.xml = self.entity
        req_s.initial_pose = goal_pose1
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.spawn_entity_client.call_async(req_s)
        #self.get_logger().info(f'spawn_entity at New goal : x={self.goalX }, y={self.goalY}')
    def delete_entity(self, entity_name):
        #env.get_logger().info("delete_entity **********************")
        req = DeleteEntity.Request()
        req.name = entity_name
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.delete_entity_client.call_async(req)
        #self.get_logger().info(f'Deleting {entity_name}')
    def calculate_observation(self, data):
        min_range = 0.5
        min_laser = 2.0
        done = False
        col = False

        for i, item in enumerate(data.ranges):
            if min_laser > data.ranges[i]:
                min_laser = data.ranges[i]
            if (min_range > data.ranges[i] > 0):
                done = True
                col = True
        return done, col, min_laser

    # Perform an action and read a new state
    def step(self, act, timestep):
        self.spawn_entity()
        vel_cmd = Twist()
        vel_cmd.linear.x = act[0]
        vel_cmd.angular.z = act[1]
        self.vel_pub.publish(vel_cmd)

        target = False
        '''rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")'''
            
        ###############################"
        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...') # type: ignore

        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/gazebo/unpause_physics service call failed")
        # #############################"

        time.sleep(0.1)
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            pass
            self.pause.call_async(Empty.Request())
        except (rclpy.ServiceException) as e: # type: ignore
            print("/gazebo/pause_physics service call failed")

        '''dataOdom = None
        while dataOdom is None:
            try:
                dataOdom = rospy.wait_for_message('/scout/odom', Odometry, timeout=0.1)
            except:
                pass

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/front_laser/scan', LaserScan, timeout=0.1)
            except:
                pass

        data_obs = None

        data_obs_fish = None
        while data_obs_fish is None:
            try:
                data_obs_fish = rospy.wait_for_message('/camera/fisheye/image_raw', Image, timeout=0.1)
            except:
                pass

        time.sleep(0.1)
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")'''

        #data = self.last_laser
        data = scan_data
        #dataOdom = self.last_odom
        dataOdom = last_odom
        #data_obs = self.last_image
        data_obs = last_image
        #data_obs_fish = self.last_image_fish
        laser_state = np.array(data.ranges[:])
        v_state = []
        v_state[:] = velodyne_data[:]
        laser_state = [v_state]

        done, col, min_laser = self.calculate_observation(data)

        # Calculate robot heading from odometry data
        self.odomX = dataOdom.pose.pose.position.x
        self.odomY = dataOdom.pose.pose.position.y
        #print(f'x= {self.odomX}  y = {self.odomY}')
        self.x_pos_list.append(round(self.odomX,2))
        self.y_pos_list.append(round(self.odomY,2))
        
        quaternion = Quaternion(
            dataOdom.pose.pose.orientation.w,
            dataOdom.pose.pose.orientation.x,
            dataOdom.pose.pose.orientation.y,
            dataOdom.pose.pose.orientation.z)
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        Dist = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))

        # Calculate the angle distance between the robots heading and heading toward the goal
        skewX = self.goalX - self.odomX
        skewY = self.goalY - self.odomY
        dot = skewX * 1 + skewY * 0
        mag1 = math.sqrt(math.pow(skewX, 2) + math.pow(skewY, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skewY < 0:
            if skewX < 0:
                beta = -beta
            else:
                beta = 0 - beta
                
        beta2 = (beta - angle)
        if beta2 > np.pi:
            beta2 = np.pi - beta2
            beta2 = -np.pi - beta2
        if beta2 < -np.pi:
            beta2 = -np.pi - beta2
            beta2 = np.pi - beta2

        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goalX
        marker.pose.position.y = self.goalY
        marker.pose.position.z = 0.0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(act[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5.0
        marker2.pose.position.y = 0.0
        marker2.pose.position.z = 0.0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(act[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5.0
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0.0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

        markerArray4 = MarkerArray()
        marker4 = Marker()
        marker4.header.frame_id = "odom"
        marker4.type = marker.CUBE
        marker4.action = marker.ADD
        marker4.scale.x = 0.1
        marker4.scale.y = 0.1
        marker4.scale.z = 0.01
        marker4.color.a = 1.0
        marker4.color.r = 1.0
        marker4.color.g = 0.0
        marker4.color.b = 0.0
        marker4.pose.orientation.w = 1.0
        marker4.pose.position.x = 5.0
        marker4.pose.position.y = 0.4
        marker4.pose.position.z = 0.0

        markerArray4.markers.append(marker4)
        self.publisher4.publish(markerArray4)

        '''Bunch of different ways to generate the reward'''
        
        r_heuristic = (self.distOld - Dist) * 20 #* math.cos(act[0]*act[1]/4)
        r_action = act[0]*2 - abs(act[1])
        r_smooth = - abs(act[1] - self.last_act)/4
              
        self.distOld = Dist

        r_target = 0.0
        r_collision = 0.0
        r_freeze = 0.0

        # Detect if the goal has been reached and give a large positive reward
        if Dist < 0.5:
            target = True
            done = True
            self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
            r_target = 100

        # Detect if ta collision has happened and give a large negative reward
        if col:
            self.collision += 1
            r_collision = -100

        if timestep>10 and self.check_list(self.x_pos_list) and self.check_list(self.y_pos_list):
            r_freeze = -1

        reward = r_heuristic + r_action + r_collision + r_target + r_smooth #+ r_freeze
        Dist  = min(Dist/15, 1.0) #max 15m away from current position
        beta2 = beta2 / np.pi
        toGoal = np.array([Dist, beta2, act[0], act[1]])
        
        ####### Depth Iamge ########
        # image = data_obs.copy()
        # image[np.isnan(image)] = 10.0
        # state = image/10
        
        ######## FishEye ####
        #state = data_obs_fish / 255
        image = np.expand_dims(cv2.resize(data_obs, (160, 128)), axis=2)
        state = image / 255
        self.last_act = act[1]
        return state, r_heuristic, r_action, r_freeze, r_collision, r_target, reward, done, toGoal, target

    def check_list(self, buffer):
        it = iter(buffer)
        try:
            first = next(it)
        except StopIteration:
            return True
        return all((abs(first-x)<0.1) for x in buffer)

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        '''rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")'''
        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        try:
            self.reset_proxy.call_async(Empty.Request())
        except rclpy.ServiceException as e: # type: ignore
            print("/gazebo/reset_simulation service call failed")
        #print(f'x= {self.odomX}  y = {self.odomY}')

        self.delete_entity(self.entity_name)
        self.change_goal()
        self.spawn_entity()
        self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
        data = scan_data
        data_obs = None
        data_obs_fish = last_image

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/gazebo/unpause_physics service call failed")
        time.sleep(0.2)
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            self.pause.call_async(Empty.Request())
        except:
            print("/gazebo/pause_physics service call failed")
            
        '''while data is None:
            print("HHHHHHHHHHHHHHHHHHHHHHHHHHH")
            self.spawn_model()'''
        laser_state = np.array(data.ranges[:])
        laser_state[laser_state == inf] = 10
        laser_state = binning(0, laser_state, 20)

        '''while data_obs_fish is None:
            try:
                data_obs_fish = rospy.wait_for_message('/camera/fisheye/image_raw', Image, timeout=0.1)
            except:
                pass'''
        
        ##### FishEye Image ####
        camera_image = data_obs_fish
        
        #image = self.br.imgmsg_to_cv2(camera_image, "mono8")
        '''print("Saluuuut")
        print(type(camera_image))'''
        image = np.expand_dims(cv2.resize(camera_image, (160, 128)), axis=2)
        
        #image = np.expand_dims(cv2.resize(image[80:400, 140:500], (160, 128)), axis=2)
        state = image/255

        ######## Depth Image ##########
        # image = self.br.imgmsg_to_cv2(camera_image, "passthrough")
        # image = image.copy()
        # image[np.isnan(image)] = 10.0
        # image = np.expand_dims(cv2.resize(image, (128, 64)), axis=2)
        # state = image/10

        '''while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            pass
            self.pause.call_async(Empty.Request())
        except (rclpy.ServiceException) as e: # type: ignore
            print("/gazebo/pause_physics service call failed")
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")'''

        Dist = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))

        skewX = self.goalX - self.odomX
        skewY = self.goalY - self.odomY

        dot = skewX * 1 + skewY * 0
        mag1 = math.sqrt(math.pow(skewX, 2) + math.pow(skewY, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skewY < 0:
            if skewX < 0:
                beta = -beta
            else:
                beta = 0 - beta
        beta2 = (beta - self.angle)

        if beta2 > np.pi:
            beta2 = np.pi - beta2
            beta2 = -np.pi - beta2
        if beta2 < -np.pi:
            beta2 = -np.pi - beta2
            beta2 = np.pi - beta2

        Dist  = min(Dist/15, 1.0) # max 15m away from current position
        beta2 = beta2 / np.pi
        toGoal = np.array([Dist, beta2, 0.0, 0.0])
        return state, toGoal

    # Place a new goal and check if its lov\cation is not on one of the obstacles
    def change_goal(self):
        if self.upper < 10:
            self.upper += 0.008
        if self.lower > -10:
            self.lower -= 0.008

        gOK = False
        #and self.goalX==self.modelX and self.goalY==self.goalY
        while not gOK :
            self.goalX = self.odomX + random.uniform(self.upper, self.lower)
            self.goalY = self.odomY + random.uniform(self.upper, self.lower)

            euclidean_dist = math.sqrt((self.goalX - self.odomX)**2 + (self.goalY - self.odomY)**2)
            if self.upper > 4 and euclidean_dist < 3:
                gOK = False
                continue
            elif self.upper > 8 and euclidean_dist < 6:
                gOK = False
                continue

            gOK = check_pos(self.goalX, self.goalY)
    def change_pose(self):
        self.angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0., 0., self.angle)
        #object_state = self.set_self_state

        x = 0.0
        y = 0.0
        chk = False
        while not chk :
            x = np.random.uniform(-7.0, 7.0)
            y = np.random.uniform(-4.0, 4.0)
            chk = check_pos(x, y)
        self.modelX = x
        self.modelY = y
        self.orientationX = quaternion.x
        self.orientationY = quaternion.y
        self.orientationZ = quaternion.z
        self.orientationW = quaternion.w
        '''if(i==1):
            self.delete_entity(self.model_name)
        #time.sleep(0.2)
        self.spawn_model()
        time.sleep(0.2)'''
        self.odomX = self.orientationX
        self.odomY = self.orientationY
        return x,y,quaternion.x,quaternion.y,quaternion.z,quaternion.w
        
                
            

    # Randomly change the location of the boxes in the environment on each reset to randomize the training environment
    def random_box(self):
        for i in range(2):
            name = 'cardboard_box_' + str(i)

            x = 0.0
            y = 0.0
            chk = False
            while not chk:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                chk = check_pos(x, y)
                d1 = math.sqrt((x - self.odomX) ** 2 + (y - self.odomY) ** 2)
                d2 = math.sqrt((x - self.goalX) ** 2 + (y - self.goalY) ** 2)
                if d1 < 1.5 or d2 < 1.5:
                    chk = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)
class Image_subscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber') 
        self.subscription = self.create_subscription(
            Image,
            '/camera1/image_raw',
            self.image_callback, qos_profile_sensor_data)
        self.subscription

    def image_callback(self, im_data):
        # print(".....................HI  image_callback!!.........................")
        global last_image
        cv_image = CvBridge().imgmsg_to_cv2(im_data, "mono8")
        last_image = cv_image
        
class Image_subscriber3(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.image_callback,
            qos_profile_sensor_data)
        self.bridge = CvBridge()
        self.image_counter = 0  # To keep track of the image files

    def image_callback(self, im_data):
        self.get_logger().info('Received image data')
        try:
            cv_image = self.bridge.imgmsg_to_cv2(im_data, "mono8")
            file_name = f'image_{self.image_counter:04d}.png'
            cv2.imwrite(file_name, cv_image)
            self.get_logger().info(f'Saved image: {file_name}')
            self.image_counter += 1
        except Exception as e:
            self.get_logger().error(f'Failed to save image: {str(e)}')
class Image_subscriber33(Node):
    def __init__(self):
        super().__init__('depth_image_saver')
        self.subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',  # Replace with your depth image topic
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        self.get_logger().info('Receiving depth image')
        try:
            global last_image
            # Convert ROS Image message to OpenCV2 image
            last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # Save the image as a .png file
            cv2.imwrite('depth_image.png', last_image)
            self.get_logger().info('Depth image saved as /tmp/depth_image.png')
        except Exception as e:
            self.get_logger().error('Could not convert depth image: %s' % str(e))

class Odom_subscriber(Node):

    def __init__(self):
        super().__init__('odom_subscriber')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.subscription

    def odom_callback(self, od_data):
        global last_odom
        last_odom = od_data

class Velodyne_subscriber(Node):

    def __init__(self):
        super().__init__('velodyne_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            "/velodyne_points",
            self.velodyne_callback,
            10)
        self.subscription

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / 20]]
        for m in range(20 - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / 20]
            )
        self.gaps[-1][-1] += 0.03

    def velodyne_callback(self, v):
        global velodyne_data
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        velodyne_data = np.ones(20) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        velodyne_data[j] = min(velodyne_data[j], dist)
                        break

class LaserScan_subscriber(Node):

    def __init__(self):
        super().__init__('laserScan_subscriber')
        self.subscription = self.create_subscription(
            LaserScan,
            '/front_laser/scan',
            self.laser_callback,
            1)
        self.subscription

    def laser_callback(self, od_data):
        global scan_data
        scan_data = od_data 