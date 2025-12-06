#!/usr/bin/env python3

import rclpy, os, cv2, datetime
import numpy as np
from cv_bridge import CvBridge
import open3d as o3d
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer
import threading
from collections import deque

from ros2_camera_lidar_fusion.read_yaml import extract_configuration

class SaveData(Node):
    def __init__(self):
        super().__init__('save_data_node')
        self.get_logger().info('Save data node has been started')

        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        self.max_file_saved = config_file['general']['max_file_saved']
        self.storage_path = config_file['general']['data_folder']
        self.image_topic = config_file['camera']['image_topic']
        self.lidar_topic = config_file['lidar']['lidar_topic']
        self.keyboard_listener_enabled = config_file['general']['keyboard_listener']
        self.slop = config_file['general']['slop']
        self.scans_per_file = 100  # Количество сканов в одном файле

        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
        self.get_logger().warn(f'Data will be saved at {self.storage_path}')

        # Очередь для хранения сканов лидара
        self.lidar_queue = deque(maxlen=self.scans_per_file)
        self.image_queue = deque(maxlen=self.scans_per_file)
        
        self.image_sub = Subscriber(
            self,
            Image,
            self.image_topic
        )
        self.pointcloud_sub = Subscriber(
            self,
            PointCloud2,
            self.lidar_topic
        )

        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.pointcloud_sub],
            queue_size=30,
            slop=self.slop
        )
        self.ts.registerCallback(self.synchronize_data)

        self.save_data_flag = not self.keyboard_listener_enabled
        if self.keyboard_listener_enabled:
            self.start_keyboard_listener()

    def start_keyboard_listener(self):
        """Starts a thread to listen for keyboard events."""
        def listen_for_space():
            while True:
                key = input("Press 'Enter' to save data (keyboard listener enabled): ")
                if key.strip() == '':
                    self.save_data_flag = True
                    self.get_logger().info('Space key pressed, ready to save data')
        thread = threading.Thread(target=listen_for_space, daemon=True)
        thread.start()

    def synchronize_data(self, image_msg, pointcloud_msg):
        """Handles synchronized messages and saves data if the flag is set."""
        if self.save_data_flag:
            # Сохраняем данные в очереди
            self.lidar_queue.append(pointcloud_msg)
            self.image_queue.append((image_msg, datetime.datetime.now()))
            
            # Если накопилось нужное количество сканов
            if len(self.lidar_queue) >= self.scans_per_file:
                file_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                self.get_logger().info(f'Saving {self.scans_per_file} scans at {file_name}')
                total_files = len(os.listdir(self.storage_path))
                if total_files < self.max_file_saved:
                    self.save_batch_data(file_name)
                    if self.keyboard_listener_enabled:
                        self.save_data_flag = False
                # Очищаем очереди после сохранения
                self.lidar_queue.clear()
                self.image_queue.clear()

    def pointcloud2_to_open3d(self, pointcloud_msg):
        """Converts a PointCloud2 message to an Open3D point cloud."""
        points = []
        for p in point_cloud2.read_points(pointcloud_msg, skip_nans=True):
            points.append([p[0], p[1], p[2]])
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float32))
        return pointcloud

    def save_batch_data(self, file_name):
        """Сохраняет батч сканов лидара и одно изображение"""
        # Объединяем все точки из очереди сканов
        all_points = []
        
        for pointcloud_msg in self.lidar_queue:
            for p in point_cloud2.read_points(pointcloud_msg, skip_nans=True):
                all_points.append([p[0], p[1], p[2]])
        
        # Создаем и сохраняем объединенное облако точек
        if all_points:
            combined_pointcloud = o3d.geometry.PointCloud()
            combined_pointcloud.points = o3d.utility.Vector3dVector(
                np.array(all_points, dtype=np.float32)
            )
            o3d.io.write_point_cloud(f'{self.storage_path}/{file_name}_combined.pcd', 
                                    combined_pointcloud)
            self.get_logger().info(f'Saved combined point cloud with {len(all_points)} points')
        
        # Сохраняем последнее изображение (или первое)
        if self.image_queue:
            bridge = CvBridge()
            image_msg, _ = self.image_queue[-1]  # Берем последнее изображение
            image = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            cv2.imwrite(f'{self.storage_path}/{file_name}.png', image)
            
        self.get_logger().info(f'Batch data saved at {self.storage_path}/{file_name}')


def main(args=None):
    rclpy.init(args=args)
    node = SaveData()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()