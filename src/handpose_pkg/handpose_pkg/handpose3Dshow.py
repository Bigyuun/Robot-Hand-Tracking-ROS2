import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import pyrealsense2 as rs
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from functools import partial
import threading

import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSReliabilityPolicy

from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from std_msgs.msg import Header
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension

from rviz_visual_tools import RvizVisualTools

import queue
queue = queue.Queue()

class RealTimePlot3D(Node):
    def __init__(self, num_points=21):
        super().__init__('handpose3Dplot_node')
        self.declare_parameter('qos_depth', 10)
        qos_depth = self.get_parameter('qos_depth').value
        
        QOS_RKL10V = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=qos_depth,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.handpose_subscriber = self.create_subscription(
            Float64MultiArray,
            'mediapipe_hand_landmarks',
            self.handlandmarks_callback,
            QOS_RKL10V
        )
        self.rviz_visual_tools = RvizVisualTools('base_link', 'visualization_marker')

        self.num_points = num_points
        self.handlandmarks = np.zeros((21,3))
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')
        self.ax.set_xlim(-0.1, 0.1)
        self.ax.set_ylim(-0.1, 0.1)
        self.ax.set_zlim(-0.1, 0.1)
        # self.ani = None
        self.ani = FuncAnimation(self.fig, self.update_data, frames=1, interval=100)
        self.scatter = self.ax.scatter(self.handlandmarks[:, 0],
                                       self.handlandmarks[:, 1],
                                       self.handlandmarks[:, 2])
        self.update_thread = threading.Thread(target=self.update_data)
        # self.start()
        
        self.plt_thread = threading.Thread(target=self.show_plt)
        self.plt_thread.start()
        # self.start()
        # self.show_plt()
        
    def handlandmarks_callback(self, msg):
        num_rows = msg.layout.dim[0].size
        num_cols = msg.layout.dim[1].size
        self.handlandmarks = np.array(msg.data).reshape((num_rows, num_cols))
        
    def update_data(self, frames):

        colors = ['black', 'blue', 'green', 'orange', 'red', 'black']
        intervals = [4, 8, 12, 16, 20]

        while True:
            # 현재 점 삭제
            self.scatter.remove()
            self.ax.cla()
            self.ax.set_xlabel('X Label')
            self.ax.set_ylabel('Y Label')
            self.ax.set_zlabel('Z Label')

            self.scatter = self.ax.scatter(self.handlandmarks[:, 0], self.handlandmarks[:, 1], self.handlandmarks[:, 2],
                                           color='black',
                                           s=50,
                                           alpha=1)
            for i in range(len(intervals)):
                start_idx = 0 if i == 0 else intervals[i - 1] + 1
                end_idx = intervals[i]
                self.ax.plot(self.handlandmarks[start_idx:end_idx + 1, 0],
                            self.handlandmarks[start_idx:end_idx + 1, 1],
                            self.handlandmarks[start_idx:end_idx + 1, 2],
                            color=colors[i])

            return self.scatter

    def start(self):
        # 데이터 업데이트 쓰레드 시작
        self.update_thread.start()
        self.ani = FuncAnimation(self.fig, self.update_data, frames=100, interval=30)
        
    def show_plt(self):
        plt.show()
        
def main(args=None):
    rclpy.init(args=args)
    handpose3Dplot_node = RealTimePlot3D()
    rclpy.spin(handpose3Dplot_node)
    handpose3Dplot_node.destroy_node()
    rclpy.shutdown()
        
    # try:
    #     handpose3Dplot_node = RealTimePlot3D()
    #     try:
    #         rclpy.spin(handpose3Dplot_node)
    #     except KeyboardInterrupt:
    #         handpose3Dplot_node.get_logger().info('Keyboard Interrupt (SIGINT)')
    #     finally:
    #         handpose3Dplot_node.destroy_node()
    # finally:
    #     rclpy.shutdown()
        


if __name__ == '__main__':
    main()


