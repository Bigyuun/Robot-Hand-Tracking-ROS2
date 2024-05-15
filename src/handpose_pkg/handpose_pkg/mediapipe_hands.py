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

import json

from RealSense_Utilities.realsense_api.realsense_api import RealSenseCamera
from RealSense_Utilities.realsense_api.realsense_api import find_realsense
from RealSense_Utilities.realsense_api.realsense_api import frame_to_np_array

import queue

queue = queue.Queue()


class HandLandmarks():
    def __init__(self):
        self.keypoints = json.load(open("./handlandmark_keypoints.json"))
        self.hand_landmarks = {}  # tuple
        self.hand_landmarks['index'] = None
        self.hand_landmarks['score'] = None
        self.hand_landmarks['label'] = None
        self.hand_landmarks['landmarks'] = None
        self.hand_landmarks['world_landmarks'] = None  # calculate from camera instrinsic(Realsense)

        # if serveral hands is detected, hand_results will has objects of each hands.
        # e.g. [{hand_landmarks#1}, {hand_landmarks#2}, ...]
        self.hand_results = []

        pass


class MediaPipeHandLandmarkDetector(HandLandmarks):
    # pass
    def __init__(self):
        super().__init__()
        self.camera_to_hand_vector = {'x': 0, 'y': 0, 'z': 0}
        self.hand_data = np.zeros((21, 3))
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8,
                                         min_tracking_confidence=0.7)
        # smoothing (LPF-low pass filter)
        self.depth_intrinsics = None
        self.finger_depth_prev = 0
        self.finger_depth_curr = 0
        self.filter_sensitivity = 0.3

        self.color_image = None
        self.depth_image = None
        self.depth_image_prev = None
        self.start_flag = False

        self.pps = 0

    def hand_detection(self, color_frame, depth_frame, depth_intrinsic=None):

        ######################################################
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        self.depth_intrinsics = depth_intrinsic
        self.color_image = color_frame
        self.color_image = cv2.flip(self.color_image, 1)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        self.depth_image = depth_frame

        if self.start_flag == False:
            self.depth_image_filtered = self.depth_image
        else:
            self.depth_image_filtered = self.filter_sensitivity * self.depth_image + (
                    1 - self.filter_sensitivity) * self.depth_image_prev

        self.depth_image_filtered = cv2.flip(self.depth_image_filtered, 1)
        self.depth_image_prev = self.depth_image_filtered
        # depth_norm_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_JET)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        self.color_image.flags.writeable = False
        results = self.hands.process(self.color_image)
        image_height, image_width, _ = self.color_image.shape
        depth_image_height, depth_image_width = self.depth_image_filtered.shape
        if image_height != depth_image_height or image_width != depth_image_width:
            print(f'[mediapipe_hands.py | Warning] It does not match a size (H x W) of color & depth frame')
        # Draw the hand annotations on the image.
        self.color_image.flags.writeable = True
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2BGR)
        ######################################################

        self.hand_results.clear()

        if results.multi_handedness:

            s_time = time.time()

            for handedness, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                # dictionary clear
                self.hand_landmarks = {key: None for key in self.hand_landmarks}

                classification = handedness.classification[0]
                index = classification.index
                score = classification.score
                label = classification.label
                self.hand_landmarks['index'] = classification.index
                self.hand_landmarks['score'] = classification.score
                self.hand_landmarks['label'] = classification.label

                hand_data = np.zeros((21, 3))
                for idx, landmrk in enumerate(hand_landmarks.landmark):
                    cx, cy, cz = landmrk.x, landmrk.y, landmrk.z
                    hand_data[idx, 0] = cx
                    hand_data[idx, 1] = cy
                    hand_data[idx, 2] = cz

                    pixel_x = int(cx * image_width)
                    pixel_y = int(cy * image_height)

                self.hand_landmarks['landmarks'] = hand_data

                world_landmarks = self.conversion_hand_keypoint_pixel_to_point(hand_data, top_point_n=None)
                self.hand_landmarks['world_landmarks'] = world_landmarks

                # finally
                self.hand_results.append(self.hand_landmarks)

                # drawing points on image
                self.mp_drawing.draw_landmarks(
                    self.color_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    # mp_drawing_styles.get_default_hand_landmarks_style(),
                    # mp_drawing_styles.get_default_hand_connections_style()
                )

            # print(pixel_x, pixel_y)
            # if pixel_x>0 and pixel_x<image_width and pixel_y>0 and pixel_y<image_height:
            #     # Applying filter
            #     finger_depth_curr = self.depth_image_filtered[pixel_y, pixel_x]
            #     filter_depth = filter_sensitivity*finger_depth_curr + (1-filter_sensitivity)*finger_depth_prev
            #     # camera_to_hand_vector = rs.rs2_deproject_pixel_to_point(color_intrinsics, [pixel_x, pixel_y], filter_depth)
            #     camera_to_hand_vector = rs.rs2_deproject_pixel_to_point(self.depth_image_filtered, [pixel_x, pixel_y], filter_depth)
            #     self.camera_to_hand_vector['x'] = camera_to_hand_vector[0]
            #     self.camera_to_hand_vector['y'] = camera_to_hand_vector[1]
            #     self.camera_to_hand_vector['z'] = camera_to_hand_vector[2]

            #     finger_depth_prev = filter_depth
            #     cv2.line(image, (int(image_width/2), int(image_height/2)), (int(image_width/2), int(image_height/2)), (255,0,0), 5)
            #     cv2.putText(image, f"{filter_depth:.1f} mm",
            #                 (pixel_x, pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            #     cv2.putText(image, f"{camera_to_hand_vector[0]:.1f}, {camera_to_hand_vector[1]:.1f}, {camera_to_hand_vector[2]:.1f} mm",
            #                 (pixel_x, pixel_y+15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            #     cv2.line(depth_norm_image, (pixel_x, pixel_y), (pixel_x, pixel_y), (0, 255, 0), 5)
            #     cv2.putText(depth_norm_image, f"{filter_depth:.1f} mm",
            #                 (pixel_x, pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
            #     cv2.putText(depth_norm_image, f"{camera_to_hand_vector[0]:.1f}, {camera_to_hand_vector[1]:.1f}, {camera_to_hand_vector[2]:.1f} mm",
            #                 (pixel_x, pixel_y+15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

            e_time = time.time()
            self.pps = 1 / (e_time - s_time)

    def get_hand_keypoint_pixel_xy(self):
        pass

    def get_hand_keypoint_pixel_xyz(self, hand_index=0, keypoint='index_finger_tip'):
        try:
            keypoint_index = self.keypoints['keypoints'][keypoint]
            landmarks = self.hand_results[hand_index]['landmarks']
            x = landmarks[keypoint_index, 0]
            y = landmarks[keypoint_index, 1]
            z = landmarks[keypoint_index, 2]

            return x, y, z
        except Exception as e:
            print(f'[mediapipe_hands.py] Exception error: {e}')
        finally:
            pass

    def conversion_hand_keypoint_pixel_to_point(self, keypoints_array, top_point_n=None):
        """Algorithm for annotating pixel.
        Select 3(or more, optional) points closest to the camera.
        And, calculate of scale of x,y and z (distance_pixel : distance_world)
        Finally, obtain all 21 x,y and z dimension values of keypoints on real world

        Landmarks
        There are 21 hand landmarks, each composed of x, y and z coordinates.
        The x and y coordinates are normalized to [0.0, 1.0] by the image width and height, respectively.
        The z coordinate represents the landmark depth, with the depth at the wrist being the origin.
        The smaller the value, the closer the landmark is to the camera.
        The magnitude of z uses roughly the same scale as x.
        source: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#handle_and_display_results


        Author: DY
        Args:
            keypoints_array (ndarray): (21, 3) array

        Returns:
            ndarray: (21,3) world dx,dy,dz coordinates
        """

        try:
            keypoint_arr = np.reshape(keypoints_array, (21, 3))  # must be check

            if top_point_n == None:
                align_keypoint_world_arr = []
                h, w = self.depth_image_filtered.shape
                for idx, pixel_xyz in enumerate(keypoint_arr):
                    x = min(int(pixel_xyz[0] * w), w - 1)  # prevent the index overflow
                    y = min(int(pixel_xyz[1] * h), h - 1)  # prevent the index overflow
                    if x <= 0 or x >= w or y <= 0 or y >= h:
                        ''' out of index '''
                        return

                    z = self.depth_image_filtered[y, x]
                    xyz_world = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x, y], z)
                    align_keypoint_world_arr.append(xyz_world)

                return align_keypoint_world_arr

            elif isinstance(top_point_n, int):
                z_arr = keypoint_arr[:, 2]  # depth values
                min_z = np.min(z_arr)
                min_z_index = np.argmin(z_arr)
                ##########################################################
                ## select top N values and obtain scale of x,y and z
                # - start
                b_ids, b_values = self.get_bottom_n_values(z_arr, n=top_point_n)

                selected_keypoint_arr = keypoint_arr[b_ids]

                selected_keypoint_world_arr = []
                offset_min_z = 0
                h, w = self.depth_image_filtered.shape
                for idx, pixel_xyz in enumerate(selected_keypoint_arr):
                    x = min(int(pixel_xyz[0] * w), w - 1)  # prevent the index overflow
                    y = min(int(pixel_xyz[1] * h), h - 1)  # prevent the index overflow
                    if x <= 0 or x >= w or y <= 0 or y >= h:
                        ''' out of index '''
                        return

                    z = self.depth_image_filtered[y, x]
                    if idx == min_z_index:
                        offset_min_z = z
                    xyz_world = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x, y], z)
                    selected_keypoint_world_arr.append(xyz_world)

                s_x, s_y, s_z = self.get_scales_xyz_coordinate_to_world(selected_keypoint_arr=selected_keypoint_arr,
                                                                        selected_keypoint_world_arr=selected_keypoint_world_arr,
                                                                        sz_type='sx')
                # obtan scale of x,y and z
                # - end
                ##########################################################

                # calculate world xyz of all keypoints
                # There is the keypoint which the nearest keypoint from camera, we make its z-value as 0.
                # cf) Originally, wrist point is 0 on mediapipe algorithm.
                '''
                TODO
                Apply scale of x,y and z  
                and offset from top distance !!! 2024.05.13
                '''
                align_z_keypoint_arr = keypoint_arr
                align_z_keypoint_arr[:, 2] = keypoint_arr[:, 2] - min_z
                align_z_keypoint_world_arr = []
                for idx, pixel_xyz in enumerate(align_z_keypoint_arr):
                    up_x = int(pixel_xyz[0] * w)
                    up_y = int(pixel_xyz[1] * h)
                    x = min(int(pixel_xyz[0] * w), w - 1)  # prevent the index overflow
                    y = min(int(pixel_xyz[1] * h), h - 1)  # prevent the index overflow
                    if x <= 0 or x >= w or y <= 0 or y >= h:
                        ''' out of index '''
                        return
                    # z = self.depth_image_filtered[y, x]
                    z = pixel_xyz[2] * s_z + offset_min_z

                    print(f'scale: {s_z} : pixel_xyz={pixel_xyz[2]} / z={z}')
                    xyz_world = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x, y], z)
                    align_z_keypoint_world_arr.append(xyz_world)

                align_z_keypoint_world_arr = np.asarray(align_z_keypoint_world_arr, dtype=np.float64)
                return align_z_keypoint_world_arr

        except Exception as e:
            print(f'[mediapipe_hands.py] Exception error: {e}')
        finally:
            pass

    def get_top_n_values(self, arr, n=3):
        arr = np.array(arr)
        max_indices = np.argpartition(arr, -n)[-n:]  # 가장 큰 값의 인덱스를 찾습니다.
        max_values = np.partition(arr, -n)[-n:]  # 배열에서 가장 큰 값 3개를 찾습니다.
        return max_indices, max_values

    def get_bottom_n_values(self, arr, n=3):
        arr = np.array(arr)
        min_indices = np.argpartition(arr, n)[:n]  # 가장 작은 값의 인덱스를 찾습니다.
        min_values = np.partition(arr, n)[:n]  # 배열에서 가장 작은 값 3개를 찾습니다.
        return min_indices, min_values

    def get_scales_xyz_coordinate_to_world(self, selected_keypoint_arr, selected_keypoint_world_arr, sz_type='sx'):
        """Obtain scales of x,y and z
        scale = abs(real_distance / point_distance)
        It is the average of values that have several points and are selected as a combination of two points.
        (nC2)

        Args:
            selected_keypoint_arr (_type_): _description_
            selected_keypoint_world_arr (_type_): _description_

        Returns:
            _type_: _description_
        """
        n_arr = np.array(selected_keypoint_arr)
        w_arr = np.array(selected_keypoint_world_arr)

        scale_xyz = []
        # n = len(selected_keypoint_arr)
        for idx, _ in enumerate(selected_keypoint_arr):
            for i in range(idx + 1, len(selected_keypoint_arr)):
                scales = abs((w_arr[idx] - w_arr[i]) / (n_arr[idx] - n_arr[i]))
                scale_xyz.append(scales)
        scale_xyz = np.asarray(scale_xyz)  # convert to numpay array
        scale_x = np.mean(scale_xyz[:, 0])
        scale_y = np.mean(scale_xyz[:, 1])
        if (sz_type == 'sx'):
            scale_z = scale_x
        else:
            scale_z = np.mean(scale_xyz[:, 2])

        return scale_x, scale_y, scale_z

    # def get_hand_keypoint_normalized_xyz(self, hand_index=0, keypoint='index_finger_tip'):
    #     try:
    #         keypoint_index = self.keypoints['keypoints'][keypoint]
    #         landmarks = self.hand_results[hand_index]['landmarks']
    #         x = landmarks[keypoint_index,0]
    #         y = landmarks[keypoint_index,1]
    #         z = landmarks[keypoint_index,2]

    #         return x,y,z
    #     except Exception as e:
    #         print(f'[mediapipe_hands.py] Exception error: {e}')
    #     finally:
    #         pass

    def rs_init(self):
        # Realsense 카메라 객체 생성
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # 카메라 시작
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        # align = rs.align(rs.stream.depth)

        self.color_intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.depth_intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        # color_intrinsics = align.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        # depth_intrinsics = align.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        print(f'intrinsics(color): {self.color_intrinsics}')
        print(f'intrinsics(depth): {self.depth_intrinsics}')

        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        if self.depth_sensor.supports(rs.option.depth_units):
            self.depth_sensor.set_option(rs.option.depth_units, 0.001)
            print(f'depth sensor: {self.depth_sensor} - change depth_units to : 0.001')
            self.pipeline.stop()
            print(f'Reopen the camera...')
            time.sleep(1)

            self.profile = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
        else:
            print('depth sensor doesn''t support changing depth_units.')

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        # align_frames = align.process(frames)
        align_frames = frames
        color_frame = align_frames.get_color_frame()
        depth_frame = align_frames.get_depth_frame()
        depth_frame = rs.hole_filling_filter().process(depth_frame)

        if not depth_frame or not color_frame:
            return depth_frame, color_frame

        self.color_image = np.asanyarray(color_frame.get_data())
        self.depth_image = np.asanyarray(depth_frame.get_data())

        # return
        return self.color_image, self.depth_image


def main(args=None):
    try:
        mphand = MediaPipeHandLandmarkDetector()
        mphand.rs_init()

        while True:
            try:
                image_color, image_depth = mphand.get_frames()

                # if not image_color or not image_depth:
                #     continue

                mphand.hand_detection(image_color, image_depth, mphand.depth_intrinsics)

                # cv2.putText(image, f"FPS: {int(fps)}", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

                # pixel_x = int(g_hand_data[8, 0] * image_width)
                # pixel_y = int(g_hand_data[8, 1] * image_height)
                # # print(pixel_x, pixel_y)
                # if pixel_x > 0 and pixel_x < image_width and pixel_y > 0 and pixel_y < image_height:
                #
                #     # Applying filter
                #     finger_depth_curr = depth_image[pixel_y, pixel_x]
                #     filter_depth = filter_sensitivity * finger_depth_curr + (1 - filter_sensitivity) * finger_depth_prev
                #     # camera_to_hand_vector = rs.rs2_deproject_pixel_to_point(color_intrinsics, [pixel_x, pixel_y], filter_depth)
                #     camera_to_hand_vector = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [pixel_x, pixel_y],
                #                                                             filter_depth)
                #     self.camera_to_hand_vector['x'] = camera_to_hand_vector[0]
                #     self.camera_to_hand_vector['y'] = camera_to_hand_vector[1]
                #     self.camera_to_hand_vector['z'] = camera_to_hand_vector[2]
                #
                #     if hand_label == "Right":
                #         self.publishall()
                #     else:
                #         msg = Vector3()
                #         msg.x = 0.0
                #         msg.y = 0.0
                #         msg.z = 300.0
                #         self.handpose_publisher.publish(msg)

                image_width = 640
                image_height = 480

                index_finger_tip_image = mphand.hand_landmarks['landmarks']
                xyz_idx = index_finger_tip_image[8]
                index_finger_tip = mphand.hand_landmarks['world_landmarks']
                x = min(int(xyz_idx[0] * image_width), image_width - 1)  # prevent the index overflow
                y = min(int(xyz_idx[1] * image_height), image_height - 1)  # prevent the index overflow
                # if x <= 0 or x >= w or y <= 0 or y >= h:
                #     ''' out of index '''
                #     return
                z = mphand.depth_image_filtered[y, x]
                z2 = index_finger_tip[8][2]

                finger_depth = z
                cv2.line(mphand.color_image, (int(image_width / 2), int(image_height / 2)),
                         (int(image_width / 2), int(image_height / 2)), (255, 0, 0), 5)
                cv2.putText(mphand.color_image, f"(basic: {finger_depth:.1f} mm",
                            (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                cv2.putText(mphand.color_image, f"(scale: {z2:.1f}) mm",
                            (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                cv2.putText(mphand.color_image,
                            f"{x:.1f}, {y:.1f}, {z:.1f} mm",
                            (x, y + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                cv2.line(mphand.depth_image_filtered, (x, y), (x, y), (0, 255, 0), 5)
                cv2.putText(mphand.depth_image_filtered, f"{finger_depth:.1f} mm",
                            (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
                cv2.putText(mphand.depth_image_filtered,
                            f"{x:.1f}, {y:.1f}, {z:.1f} mm",
                            (x, y + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

                cv2.imshow('MediaPipe Hands', mphand.color_image)
                cv2.imshow('depth image', mphand.depth_image_filtered)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
            except KeyboardInterrupt:
                print(f'Keyboard Interrupt (SIGINT)')

    finally:
        pass


if __name__ == '__main__':
    main()


