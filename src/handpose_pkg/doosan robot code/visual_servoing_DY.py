#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ##
# @brief    [py example simple] motion simple test
# @author   Kab Kyoum Kim (kabkyoum.kim@doosan.com)

import rclpy

##-------------------------------------------------------------------------------------------------
# @autour DY
# @note   delta vector from camera to point for visual servoing with ROS2 system
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from rclpy.qos import QoSProfile
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSReliabilityPolicy
##-------------------------------------------------------------------------------------------------

import os, sys
import threading, time
import signal
import math
from dsr_msgs2.msg import *

sys.dont_write_bytecode = True
sys.path.append("./install/common2/bin/common2/imp")  # get import pass : DSR_ROBOT2.py

# --------------------------------------------------------
import DR_init
g_node = None
rclpy.init()
g_node = rclpy.create_node("global_node")
DR_init.__dsr__node = g_node
from DSR_ROBOT2 import *

# --------------------------------------------------------
# @author DY
class VisualServoingNode(Node):
  """Visual Servoing Node
  -- delta x,y,z(position error) will be obtained from camera node
  then, robot arm move to the target based on TCP(tool center position) coordinate

  Args:
      Node (_type_): _description_
  """
  def __init__(self):
    global g_node
    
    super().__init__('visual_servoing_node')
    self.declare_parameter('qos_depth', 10)
    qos_depth = self.get_parameter('qos_depth').value
    QOS_RKL10V = QoSProfile(
      reliability=QoSReliabilityPolicy.RELIABLE,
      history=QoSHistoryPolicy.KEEP_LAST,
      depth=qos_depth,
      durability=QoSDurabilityPolicy.VOLATILE
    )
    self.handpose_subscriber = g_node.create_subscription(
      Vector3,
      'camera_to_hand_vector',
      self.camera_to_hand_vector_callback,
      QOS_RKL10V
    )
    self.camera_to_hand_vector = Vector3()
    
    
    
    print(self.camera_to_hand_vector)
    self.get_logger().info(f'Subscriber ''camera_to_hand_vector'' is created')
    
    self.run()

  def camera_to_hand_vector_callback(self, msg):
    self.camera_to_hand_vector = msg
    
  def run(self):
    global g_node
    signal.signal(signal.SIGINT, self.signal_handler)
    
    set_velx(30, 20)  # set global task speed: 30(mm/sec), 20(deg/sec)
    set_accx(60, 40)  # set global task accel: 60(mm/sec2), 40(deg/sec2)
    
    velx = [100, 100]
    accx = [100, 100]

    # move to initial position
    init_pos = posj(0.0, 0.0, 90.0, 0.0, 90.0, 0.0)  # joint
    movej(init_pos, vel=30, acc=30)
    print("------------> move joint OK")
    time.sleep(1)
    
    point_RCM = Vector3()
    point_RCM.x = 0.0
    point_RCM.y = 0.0
    point_RCM.z = 300.0
    
    count = 0
    while rclpy.ok():
        count += 1
        if count >= 200:
          count = 0
        target = math.sin(0.01 * count * math.pi)
        
        # x1 = posx(target,
        #           target,
        #           target,
        #           0,
        #           0,
        #           0)
        
        dx = self.camera_to_hand_vector.x - point_RCM.x
        dy = self.camera_to_hand_vector.y - point_RCM.y
        dz = self.camera_to_hand_vector.z - point_RCM.z
        if abs(dx)>=100 or abs(dy)>=100 or abs(dz)>=100:
          dx = 0.0
          dy = 0.0
          dz = 0.0
          print("out of range to move robot!")
        x1 = posx(dx/5.,
                  dy/5.,
                  dz/5.,
                  0,
                  0,
                  0)
        # move line
        # amovej(x1, velx, accx)
        amovel(x1, velx, accx, ref=DR_TOOL, mod=DR_MV_MOD_REL)
        print(count)
        print(dx)
        print(dy)
        print(dz)
        print("------------> move line OK")


    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX good-bye!")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX good-bye!")
    
  def signal_handler(self, sig, frame):
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX signal_handler")
    global g_node
    publisher = g_node.create_publisher(RobotStop, "stop", 10)

    msg = RobotStop()
    msg.stop_mode = 1

    publisher.publish(msg)
    # sys.exit(0)
    rclpy.shutdown()
  


def main(args=None):
  # rclpy.init(args=args)
  visual_servoing_node = VisualServoingNode()
  rclpy.spin(visual_servoing_node)
  visual_servoing_node.destroy_node()
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()
  
  
  
  
# #================================================================================================
# # original Code
# #================================================================================================
# def signal_handler(sig, frame):
#     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX signal_handler")
#     global g_node
#     publisher = g_node.create_publisher(RobotStop, "stop", 10)

#     msg = RobotStop()
#     msg.stop_mode = 1

#     publisher.publish(msg)
#     # sys.exit(0)
#     rclpy.shutdown()
  
  
# def main(args=None):
#     global g_node
#     signal.signal(signal.SIGINT, signal_handler)
    
#     set_velx(30, 20)  # set global task speed: 30(mm/sec), 20(deg/sec)
#     set_accx(60, 40)  # set global task accel: 60(mm/sec2), 40(deg/sec2)

#     velx = [50, 50]
#     accx = [100, 100]

#     p2 = posj(0.0, 0.0, 90.0, 0.0, 90.0, 0.0)  # joint
#     x1 = posx(600, 600, 600, 0, 175, 0)
#     x2 = posx(600, 750, 600, 0, 175, 0)
#     # move joint
#     movej(p2, vel=100, acc=100)
#     print("------------> move joint OK")
#     time.sleep(1)
#     while rclpy.ok():
       
#         # move line
#         movel(x1, velx, accx)
#         print("------------> move line OK")
        
#         # move line
#         movel(x2, velx, accx)
#         print("------------> move line OK")


#     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX good-bye!")
#     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX good-bye!")
#     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX good-bye!")


# if __name__ == "__main__":
#     main()