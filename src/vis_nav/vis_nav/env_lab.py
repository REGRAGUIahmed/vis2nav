#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os import path
from gazebo_msgs.srv import DeleteEntity, SpawnEntity, SetEntityState
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
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import Image, LaserScan
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose
from ament_index_python.packages import get_package_share_directory

last_odom = None
last_image = None
last_dist = None
scan_data =None
goal_pose_rviz = None
velodyne_data = np.ones(20) * 10 
trajectorie=[]


# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    if (3.7 < x < 5.5 and -3.5 < y < 4) or \
       (-4.5 < x < 4 and -3.5 < y < -2) or \
       (-3.5 < x < 3.3 and -1.5 < y < 2) or \
       (-5 < x < -4 and -3.5 < y < 0.3) or \
       (-5 < x < -4 and 2 < y < 4) or \
       (0 < x < 2.3 and 2.5 < y < 4) or x > 5 or x < -5 or y > 3.7 or y < -3:
        return False
    else:
        return True
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
        self.entity_dir_path=os.path.join(get_package_share_directory('vis_nav'), 'description/sdf')
        self.entity_path = os.path.join(self.entity_dir_path, 'goal.sdf')
        self.entity = open(self.entity_path, 'r').read()
        self.odomX = 0.0
        self.odomY = 0.0
        self.entityX = 0.0
        self.entityY = 2.5
        self.quaterX = 0.0
        self.quaterY = 0.0
        self.quaterZ = 0.0
        self.quaterW = 1.0
        self.goalX = 0.0
        self.goalY = 8.5
        self.angle = 0.0
        self.upper = 5.0 #10.0
        self.lower = -5.0 #-10.0
        self.collision = 0.0
        self.last_act = 0.0
        self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
        self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
        self.set_entity_client = self.create_client(SetEntityState, 'gazebo/set_entity_state')
        self.x_pos_list = deque(maxlen=5)
        self.y_pos_list = deque(maxlen=5)
        self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
        self.gaps = [[-1.6, -1.57 + 3.14 / 20]]
        for m in range(19):
            self.gaps.append([self.gaps[m][1], self.gaps[m][1] + 3.14 / 20])
        self.gaps[-1][-1] += 0.03

        # Set up the ROS publishers and subscribers
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
       
        #self.set_state = self.create_publisher(SetEntityState, "gazebo/set_entity_state", 10)
        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        topic = 'goal_mark_array'
        self.publisher = self.create_publisher(MarkerArray, topic, 3)
        self.reset_proxy = self.create_client(Empty, "/reset_world")

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def spawn_entity(self):
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
    def set_entity(self):
        request = SetEntityState.Request()
        request.state.name = 'scout'
        request.state.pose.position.x = self.entityX
        request.state.pose.position.y = self.entityY
        request.state.pose.position.z = 0.0  # Assuming ground level
        
        request.state.pose.orientation.x = self.quaterX
        request.state.pose.orientation.y = self.quaterY
        request.state.pose.orientation.z = self.quaterZ
        request.state.pose.orientation.w = self.quaterW       
        
        future = self.set_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error('Failed to change entity position.')
   
    def delete_entity(self, entity_name):
        req = DeleteEntity.Request()
        req.name = entity_name
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.delete_entity_client.call_async(req)
    def calculate_observation(self, data):
        min_range = 0.30
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
    
    def step(self, act, timestep):
        self.spawn_entity()
        vel_cmd = Twist()
        vel_cmd.linear.x = act[0]
        vel_cmd.angular.z = act[1]
        self.vel_pub.publish(vel_cmd)

        target = False
        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...') # type: ignore

        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(0.1)
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            pass
            self.pause.call_async(Empty.Request())
        except (rclpy.ServiceException) as e: # type: ignore
            print("/gazebo/pause_physics service call failed")
        data = scan_data
        dataOdom = last_odom
        data_obs = last_image
        laser_state = np.array(data.ranges[:])
        v_state = []
        v_state[:] = velodyne_data[:]
        laser_state = [v_state]

        done, col, min_laser = self.calculate_observation(data)

        self.odomX = dataOdom.pose.pose.position.x
        self.odomY = dataOdom.pose.pose.position.y
        self.x_pos_list.append(round(self.odomX,2))
        self.y_pos_list.append(round(self.odomY,2))
        
        quaternion = Quaternion(
            dataOdom.pose.pose.orientation.w,
            dataOdom.pose.pose.orientation.x,
            dataOdom.pose.pose.orientation.y,
            dataOdom.pose.pose.orientation.z)
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

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


        '''Bunch of different ways to generate the reward'''
        
        r_heuristic = (self.distOld - Dist) * 20 #* math.cos(act[0]*act[1]/4)
        r_action = act[0]*2 - abs(act[1])
        r_smooth = - abs(act[1] - self.last_act)/4
              
        self.distOld = Dist

        r_target = 0.0
        r_collision = 0.0
        r_freeze = 0.0

        # Detect if the goal has been reached and give a large positive reward
        #if Dist < 0.5:
        if Dist < 0.5:
            target = True
            done = True
            self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
            r_target = 120

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
            
        self.delete_entity(self.entity_name)
        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0., 0., angle)
        self.quaterX = quaternion.x
        self.quaterY = quaternion.y
        self.quaterZ = quaternion.z
        self.quaterW = quaternion.w
        x = 0
        y = 0
        chk = False
        while not chk:
            x = np.random.uniform(-4.5, 5.0)
            y = np.random.uniform(-3.0, 3.5)
            chk = check_pos(x, y)
        
        self.entityX = x
        self.entityY = y
        self.quaterX = quaternion.x
        self.quaterY = quaternion.y
        self.quaterZ = quaternion.z
        self.quaterW = quaternion.w
        
        self.set_entity()
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
            
        laser_state = np.array(data.ranges[:])
        laser_state[laser_state == inf] = 10
        laser_state = binning(0, laser_state, 20)
        
        ##### FishEye Image ####
        camera_image = data_obs_fish
        image = np.expand_dims(cv2.resize(camera_image, (160, 128)), axis=2)
        state = image/255
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

        Dist  = min(Dist/15, 1.0) 
        beta2 = beta2 / np.pi
        toGoal = np.array([Dist, beta2, 0.0, 0.0])
        time.sleep(0.2) 
        return state, toGoal

    def change_goal(self):
        if self.upper < 10:
            self.upper += 0.008
        if self.lower > -10:
            self.lower -= 0.008

        gOK = False
        while not gOK:
            self.goalX = self.odomX + random.uniform(self.upper, self.lower)
            self.goalY = self.odomY + random.uniform(self.upper, self.lower)

            euclidean_dist = math.sqrt((self.goalX - self.odomX)**2 + (self.goalY - self.odomY)**2)
            if euclidean_dist < 1.5:
                gOK = False
                continue

            gOK = check_pos(self.goalX, self.goalY)
    def change_pose(self):
        self.angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0., 0., self.angle)
        x = 0.0
        y = 0.0
        chk = False
        while not chk :
            x = np.random.uniform(-5.0, 5.0)
            y = np.random.uniform(-3.0, 4.0)
            chk = check_pos(x, y)
        self.modelX = x
        self.modelY = y
        self.orientationX = quaternion.x
        self.orientationY = quaternion.y
        self.orientationZ = quaternion.z
        self.orientationW = quaternion.w
        self.odomX = self.orientationX
        self.odomY = self.orientationY
        return x,y,quaternion.x,quaternion.y,quaternion.z,quaternion.w

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
         
class DepthImage_subscriber(Node):
    def __init__(self):
        super().__init__('depth_image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw', 
            self.listener_callback,
            10)
        self.subscription  
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            cv_image_normalized = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
            cv_image_normalized = cv_image_normalized.astype(np.uint8)
            
            global last_image
            last_image = cv_image_normalized[80:400, 140:500]  #
            
        except Exception as e:
            self.get_logger().error('Could not convert depth image: %s' % str(e))
class Image_fish_subscriber(Node):
    def __init__(self):
        super().__init__('image_fish_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  
            self.listener_callback,
            10)
        self.subscription 
        self.bridge = CvBridge()

    def listener_callback(self, rgb_data):
        try:
            image = self.bridge.imgmsg_to_cv2(rgb_data, "mono8")
            last_image_fish = image[80:400, 118:523]
            global last_image
            last_image = last_image_fish
            
        except Exception as e:
            self.get_logger().error('Could not convert fish image: %s' % str(e))

class Image_subscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber') 
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback, qos_profile_sensor_data)
        self.subscription
        self.bridge = CvBridge()
    def image_callback(self, im_data):
        global last_image
        cv_image = self.bridge.imgmsg_to_cv2(im_data, "mono8")
        last_image = cv_image
