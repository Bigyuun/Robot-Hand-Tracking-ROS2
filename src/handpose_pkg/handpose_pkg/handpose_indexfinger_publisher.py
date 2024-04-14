import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import pyrealsense2 as rs
import time
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from functools import partial
import threading
# matplotlib.use('Agg')

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
# from rviz_visual_tools import RvizVisualTools

import queue
queue = queue.Queue()

g_hand_data = np.zeros((21, 3))
g_hand_world_data = np.zeros((21, 3))
g_hand_world_data_normalized = np.zeros((21, 3))

class RealTimePlot3D:
    def __init__(self, num_points=21):
        self.num_points = num_points
        self.data = np.random.randn(self.num_points, 3)
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
        self.scatter = self.ax.scatter(g_hand_data[:, 0], g_hand_data[:, 1], g_hand_data[:, 2])
        self.update_thread = threading.Thread(target=self.update_data)
        # self.update_thread.daemon = True
        # self.update_thread.start()
        self.start()
        self.pltshow_thread = threading.Thread(target=self.show)
        # self.pltshow_thread.daemon = True
        # self.pltshow_thread.start()        

    def update_data(self, frames):

        global g_hand_data, g_hand_data_normalized
        colors = ['black', 'blue', 'green', 'orange', 'red', 'black']
        intervals = [4, 8, 12, 16, 20]

        while True:
            # 현재 점 삭제
            self.scatter.remove()
            self.ax.cla()
            self.ax.set_xlabel('X Label')
            self.ax.set_ylabel('Y Label')
            self.ax.set_zlabel('Z Label')
            data = g_hand_data

            self.scatter = self.ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                                           color='black',
                                           s=50,
                                           alpha=1)
            for i in range(len(intervals)):
                start_idx = 0 if i == 0 else intervals[i - 1] + 1
                end_idx = intervals[i]
                self.ax.plot(data[start_idx:end_idx + 1, 0],
                        data[start_idx:end_idx + 1, 1],
                        data[start_idx:end_idx + 1, 2],
                        color=colors[i])



            return self.scatter

    def start(self):
        # 데이터 업데이트 쓰레드 시작
        self.update_thread.start()
        self.ani = FuncAnimation(self.fig, self.update_data, frames=100, interval=30)
    def show(self):
        plt.show()

class MediaPipeHandLandmarkDetectorNode(Node):
        # pass
    def __init__(self):
        super().__init__('handpose_node')
        self.declare_parameter('qos_depth', 10)
        qos_depth = self.get_parameter('qos_depth').value
        
        QOS_RKL10V = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=qos_depth,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        self.handlandmarks_publisher = self.create_publisher(
            Float64MultiArray,
            'mediapipe_hand_landmarks',
            QOS_RKL10V
        )
        
        self.handpose_publisher = self.create_publisher(
            Vector3,
            'camera_to_hand_vector',
            QOS_RKL10V
        )
        # self.rviz_visual_tools = RvizVisualTools('base_link', 'visualization_marker')
        
        self.camera_to_hand_vector = {'x':0, 'y':0, 'z':0}
        
        self.mediapipe_hand_detection()

            
    def mediapipe_hand_detection(self):
        global g_hand_data, g_hand_data_normalized
        hand_data = np.zeros((21, 3))

        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        ## for filter

        # moving average
        num_of_avg = 10
        finger_depth = np.zeros(num_of_avg)

        # smoothing
        finger_depth_prev = 0
        finger_depth_curr = 0
        filter_sensitivity = 0.3

        # Realsense 카메라 객체 생성
        pipeline = rs.pipeline()
        config = rs.config()
        # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)


        # 카메라 시작
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)

        color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        print(f'intrinsics(color): {color_intrinsics}')
        print(f'intrinsics(depth): {depth_intrinsics}')

        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.depth_units):
            depth_sensor.set_option(rs.option.depth_units, 0.001)
            print(f'depth sensor: {depth_sensor} - change depth_units to : 0.001')
            pipeline.stop()
            print(f'Reopen the camera...')
            time.sleep(1)

            profile = pipeline.start(config)
            align = rs.align(rs.stream.color)
        else:
            print('depth sensor doesn''t support changing depth_units.')

        p_time = time.time()

        with mp_hands.Hands(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7) as hands:
            while True:

                frames = pipeline.wait_for_frames()
                # align_frames = align.process(frames)
                align_frames = frames
                color_frame = align_frames.get_color_frame()
                depth_frame = align_frames.get_depth_frame()

                # depth_frame_spatial_filter = rs.spatial_filter().process(depth_frame)
                # depth_frame_temporal_filter = rs.temporal_filter().process(depth_frame)
                depth_frame = rs.hole_filling_filter().process(depth_frame)

                if not depth_frame or not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_image_origin = depth_image
                # depth_image_spatial_filter = np.asanyarray(depth_frame_spatial_filter.get_data())
                # depth_image_temporal_filter = np.asanyarray(depth_frame_temporal_filter.get_data())

                cv2.imshow('MediaPipe Hands', color_image)
                cv2.imshow('MediaPipe Hands', depth_image)
                ######################################################
                # choose image with filter
                depth_image = depth_image
                ######################################################

                # depth_image_origin = cv2.flip(depth_image_origin, 1)
                depth_image = cv2.flip(depth_image, 1)

                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                image = cv2.cvtColor(cv2.flip(color_image, 1), cv2.COLOR_BGR2RGB)
                depth_norm_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_JET)


                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = hands.process(image)
                image_height, image_width, _ = image.shape
                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                hand_label = None
                # class of hands (detected)
                if results.multi_handedness:
                    for handedness in results.multi_handedness:
                        for ids, classification in enumerate(handedness.classification):
                            index = classification.index
                            score = classification.score
                            label = classification.label
                            hand_label = label

                if hand_label == "Right":
                    # print("Right Hand")
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Here is How to Get All the Coordinates
                            # print(f"{hand_landmarks.categories.index} / score: {hand_landmarks.categories.score} / categoryname: {hand_landmarks.categories.categoryName}")

                            for ids, landmrk in enumerate(hand_landmarks.landmark):
                                cx, cy, cz = landmrk.x, landmrk.y, landmrk.z
                                g_hand_data[ids, 0] = cx
                                g_hand_data[ids, 1] = cy
                                g_hand_data[ids, 2] = cz
                                
                            handlmrks_data_1d = g_hand_data.flatten().tolist()
                            handlmrks_msg = Float64MultiArray()
                            handlmrks_msg.layout.dim = [
                                MultiArrayDimension(label='keypoint', size=21, stride=3),
                                MultiArrayDimension(label='xyz', size=3, stride=1)
                            ]
                            handlmrks_msg.data = handlmrks_data_1d
                            self.handlandmarks_publisher.publish(handlmrks_msg)
                            
                            # # NumPy 배열을 튜플의 목록으로 변환합니다.
                            # handlmrks_data_tuple = [tuple(point) for point in g_hand_data]
                            # # RViz에서 마커로 점을 나타냅니다.
                            # self.rviz_visual_tools.publishPoints(handlmrks_data_tuple, rviz_visual_tools.RED, scale=0.05)
                            
                            # 정규화
                            min_vals = np.min(g_hand_data, axis=0)
                            max_vals = np.max(g_hand_data, axis=0)
                            g_hand_data_normalized = (g_hand_data - min_vals) / (max_vals - min_vals)
                            g_hand_data_normalized[:, 2] = g_hand_data[:,2]

                            mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                # mp_drawing_styles.get_default_hand_landmarks_style(),
                                # mp_drawing_styles.get_default_hand_connections_style()
                                )

                    if results.multi_hand_world_landmarks:
                        for hand_world_landmarks in results.multi_hand_world_landmarks:
                            # Here is How to Get All the Coordinates
                            for ids, landmrk in enumerate(hand_world_landmarks.landmark):
                                # print(ids, landmrk)
                                cx, cy, cz = landmrk.x, landmrk.y, landmrk.z
                                # print(f'{ids}: {cx}, {cy}, {cz}')
                                g_hand_world_data[ids, 0] = cx
                                g_hand_world_data[ids, 1] = cy
                                g_hand_world_data[ids, 2] = cz
                else:
                    # print("No Right Hand")
                    pass
                c_time = time.time()
                fps = 1 / (c_time - p_time)
                p_time = c_time
                cv2.putText(image, f"FPS: {int(fps)}", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

                pixel_x = int(g_hand_data[8, 0] * image_width)
                pixel_y = int(g_hand_data[8, 1] * image_height)
                # print(pixel_x, pixel_y)
                if pixel_x>0 and pixel_x<image_width and pixel_y>0 and pixel_y<image_height:

                    # Applying filter
                    finger_depth_curr = depth_image[pixel_y, pixel_x]
                    filter_depth = filter_sensitivity*finger_depth_curr + (1-filter_sensitivity)*finger_depth_prev
                    # camera_to_hand_vector = rs.rs2_deproject_pixel_to_point(color_intrinsics, [pixel_x, pixel_y], filter_depth)
                    camera_to_hand_vector = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [pixel_x, pixel_y], filter_depth)
                    self.camera_to_hand_vector['x'] = camera_to_hand_vector[0]
                    self.camera_to_hand_vector['y'] = camera_to_hand_vector[1]
                    self.camera_to_hand_vector['z'] = camera_to_hand_vector[2]
                    self.publishall()
                    
                    finger_depth_prev = filter_depth

                    cv2.putText(image, f"{filter_depth:.1f} mm",
                                (pixel_x, pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    cv2.putText(image, f"{camera_to_hand_vector[0]:.1f}, {camera_to_hand_vector[1]:.1f}, {camera_to_hand_vector[2]:.1f} mm",
                                (pixel_x, pixel_y+15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    cv2.line(depth_norm_image, (pixel_x, pixel_y), (pixel_x, pixel_y), (0, 255, 0), 5)
                    cv2.putText(depth_norm_image, f"{filter_depth:.1f} mm",
                                (pixel_x, pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
                    cv2.putText(depth_norm_image, f"{camera_to_hand_vector[0]:.1f}, {camera_to_hand_vector[1]:.1f}, {camera_to_hand_vector[2]:.1f} mm",
                                (pixel_x, pixel_y+15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)


                cv2.imshow('MediaPipe Hands', image)

                # depth 표시만 gray scale
                cv2.imshow('depth image', depth_norm_image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                
    def publish_point_cloud(self, points):
        self.rviz_visual_tools.publishPoints(points, rviz_visual_tools.RED, scale=0.05)

    def publishall(self):
        msg = Vector3()
        msg.x = self.camera_to_hand_vector['x']
        msg.y = self.camera_to_hand_vector['y']
        msg.z = self.camera_to_hand_vector['z']
        self.handpose_publisher.publish(msg)
                    

def main(args=None):
    # 클래스 인스턴스 생성
    # real_time_plot = RealTimePlot3D()
    # thread_plot3d = threading.Thread(target=real_time_plot.start)
    # thread_plot3d.daemon = True
    # thread_plot3d.start()
    # print('show start')
    # plt.show()
    # print('show end')
    
    rclpy.init(args=args)
    try:
        handpose_node = MediaPipeHandLandmarkDetectorNode()
        try:
            rclpy.spin(handpose_node)
        except KeyboardInterrupt:
            handpose_node.get_logger().info('Keyboard Interrupt (SIGINT)')
        finally:
            handpose_node.destroy_node()
    finally:
        rclpy.shutdown()
            
    # thread_mediapipe = threading.Thread(target=mediapipe_hand_detection)
    # thread_mediapipe.daemon = True
    # thread_mediapipe.start()


if __name__ == '__main__':
    main()


