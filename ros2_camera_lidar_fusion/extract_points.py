#!/usr/bin/env python3

import os
import cv2
import open3d as o3d
import numpy as np
from rclpy.node import Node
import rclpy

from ros2_camera_lidar_fusion.read_yaml import extract_configuration

class ImageCloudCorrespondenceNode(Node):
    def __init__(self):
        super().__init__('image_cloud_correspondence_node')

        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        self.data_dir = config_file['general']['data_folder']
        self.file = config_file['general']['correspondence_file']

        if not os.path.exists(self.data_dir):
            self.get_logger().warn(f"Data directory '{self.data_dir}' does not exist.")
            os.makedirs(self.data_dir)

        self.get_logger().info(f"Looking for .png and .pcd file pairs in '{self.data_dir}'")
        self.process_file_pairs()

    def get_file_pairs(self, directory):
        files = os.listdir(directory)
        pairs_dict = {}
        for f in files:
            full_path = os.path.join(directory, f)
            if not os.path.isfile(full_path):
                continue
            name, ext = os.path.splitext(f)

            if ext.lower() in [".png", ".jpg", ".jpeg", ".pcd"]:
                if name not in pairs_dict:
                    pairs_dict[name] = {}
                if ext.lower() == ".png":
                    pairs_dict[name]['png'] = full_path
                elif ext.lower() == ".pcd":
                    pairs_dict[name]['pcd'] = full_path

        file_pairs = []
        for prefix, d in pairs_dict.items():
            if 'png' in d and 'pcd' in d:
                file_pairs.append((prefix, d['png'], d['pcd']))

        file_pairs.sort()
        return file_pairs

    def pick_image_points(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            self.get_logger().error(f"Error loading image: {image_path}")
            return []

        points_2d = []
        window_name = "Select points on the image (press 'q' or ESC to finish)"

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points_2d.append((x, y))
                self.get_logger().info(f"Image: click at ({x}, {y})")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            display_img = img.copy()
            for pt in points_2d:
                cv2.circle(display_img, pt, 5, (0, 0, 255), -1)

            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(10)
            if key == 27 or key == ord('q'):
                break

        cv2.destroyWindow(window_name)
        return points_2d

    def pick_cloud_points(self, pcd_path):
        pcd = o3d.io.read_point_cloud(pcd_path)
        if pcd.is_empty():
            self.get_logger().error(f"Empty or invalid point cloud: {pcd_path}")
            return []
        
        # Координаты стола из твоих точек
        table_points = np.array([
            [0.372859, 0.517000, -0.655153],  # Точка 1
            [0.801100, 0.503000, -0.707209],  # Точка 2  
            [0.782560, -0.666000, -0.702933], # Точка 3
            [0.419048, -0.587999, -0.617423]  # Точка 4
        ])
        
        # Вычисляем границы стола с запасом
        min_bound = table_points.min(axis=0) - np.array([0.1, 0.1, 0.05])  # Запас 10 см по бокам, 5 см по высоте
        max_bound = table_points.max(axis=0) + np.array([0.1, 0.1, 0.05])  # Запас 10 см по бокам, 5 см по высоте
        
        self.get_logger().info(f"\nГраницы стола:")
        self.get_logger().info(f"min: ({min_bound[0]:.3f}, {min_bound[1]:.3f}, {min_bound[2]:.3f})")
        self.get_logger().info(f"max: ({max_bound[0]:.3f}, {max_bound[1]:.3f}, {max_bound[2]:.3f})")
        
        # Обрезаем облако вокруг стола
        cropped_pcd = self.crop_point_cloud(pcd, min_bound, max_bound)
        
        if len(cropped_pcd.points) == 0:
            self.get_logger().warn("Не удалось обрезать облако. Возможно, координаты неверны.")
            self.get_logger().info("Продолжаю с полным облаком...")
            cropped_pcd = pcd
        else:
            original_count = len(pcd.points)
            cropped_count = len(cropped_pcd.points)
            reduction = (1 - cropped_count/original_count) * 100
            self.get_logger().info(f"\nОблако обрезано:")
            self.get_logger().info(f"Было: {original_count} точек")
            self.get_logger().info(f"Стало: {cropped_count} точек")
            self.get_logger().info(f"Удалено: {reduction:.1f}% точек")
        
        # Визуализируем и выбираем точки
        return self.visualize_and_pick(cropped_pcd, table_points)

    def crop_point_cloud(self, pcd, min_bound, max_bound):
        """Обрезает облако точек по заданным границам"""
        points = np.asarray(pcd.points)
        
        if len(points) == 0:
            return o3d.geometry.PointCloud()
        
        # Маска точек внутри границ
        mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
        
        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(points[mask])
        
        if pcd.has_colors() and len(pcd.colors) > 0:
            colors = np.asarray(pcd.colors)
            if len(colors) == len(points):
                cropped_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        
        return cropped_pcd

    def visualize_and_pick(self, pcd, table_corners=None):
        """Визуализация и выбор точек из облака"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("ИНСТРУКЦИЯ ДЛЯ ТОЧНОГО ВЫБОРА:")
        self.get_logger().info("1. Вы видите ОБРЕЗАННОЕ облако - только стол")
        self.get_logger().info("2. Приблизьтесь колесиком мыши к кубику")
        self.get_logger().info("3. Кубик должен занимать большую часть экрана")
        self.get_logger().info("4. Shift+ЛКМ - выбрать точку на кубике")
        self.get_logger().info("5. Q - завершить выбор")
        self.get_logger().info("="*60 + "\n")
        
        # Если есть углы стола, добавим их как маркеры
        if table_corners is not None:
            # Создаем маркеры для углов стола (маленькие сферы)
            corner_markers = []
            for i, corner in enumerate(table_corners):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # 1 см радиус
                sphere.translate(corner)
                sphere.paint_uniform_color([1, 0, 0])  # Красный
                corner_markers.append(sphere)
        
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(
            window_name="Выбор точек на КУБИКАХ (приблизьтесь!)", 
            width=1600, 
            height=900
        )
        vis.add_geometry(pcd)
        
        # Добавляем маркеры углов стола
        if table_corners is not None:
            for marker in corner_markers:
                vis.add_geometry(marker)
        
        # Настройки рендеринга
        opt = vis.get_render_option()
        opt.point_size = 3.5  # Нормальный размер для обрезанного облака
        opt.background_color = np.asarray([0.0, 0.0, 0.0])
        
        # Делаем точки яркими для лучшей видимости
        if not pcd.has_colors() or len(pcd.colors) == 0:
            colors = np.ones((len(pcd.points), 3))  # Белый цвет
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Настраиваем камеру чтобы автоматически приблизиться к столу
        if len(pcd.points) > 0:
            ctr = vis.get_view_control()
            
            # Получаем ограничивающую рамку
            bbox = pcd.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            
            # Устанавливаем камеру - ПРАВИЛЬНАЯ ОРИЕНТАЦИЯ
            ctr.set_lookat(center)           # Смотрим на центр стола
            ctr.set_up([0, -1, 0])       # Z вверх (вертикально)
            ctr.set_front([0, 0, -1])  # Смотрим сбоку
            ctr.set_zoom(1.0)                # Ближе, чем обычно
            
            self.get_logger().info(f"Камера нацелена на центр стола: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        vis.run()
        vis.destroy_window()
        
        # Получаем выбранные точки
        picked_indices = vis.get_picked_points()
        np_points = np.asarray(pcd.points)
        
        # Фильтруем - убираем маркеры углов стола (если они были выбраны)
        # Маркеры имеют индексы после точек облака
        valid_pcd_indices = [idx for idx in picked_indices if idx < len(np_points)]
        
        picked_xyz = []
        for idx in valid_pcd_indices:
            if idx < len(np_points):
                xyz = np_points[idx]
                picked_xyz.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
                self.get_logger().info(f"Выбрана точка {len(picked_xyz)}: ({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f})")
        
        self.get_logger().info(f"\nИтого выбрано {len(picked_xyz)} точек на кубиках")
        
        return picked_xyz

    def process_file_pairs(self):
        file_pairs = self.get_file_pairs(self.data_dir)
        if not file_pairs:
            self.get_logger().error(f"No .png / .pcd pairs found in '{self.data_dir}'")
            return

        self.get_logger().info("Found the following pairs:")
        for prefix, png_path, pcd_path in file_pairs:
            self.get_logger().info(f"  {prefix} -> {png_path}, {pcd_path}")

        for prefix, png_path, pcd_path in file_pairs:
            self.get_logger().info("\n========================================")
            self.get_logger().info(f"Processing pair: {prefix}")
            self.get_logger().info(f"Image: {png_path}")
            self.get_logger().info(f"Point Cloud: {pcd_path}")
            self.get_logger().info("========================================\n")

            self.get_logger().info(f"\nОткрывается изображение: {os.path.basename(png_path)}")
            image_points = self.pick_image_points(png_path)
            self.get_logger().info(f"\nSelected {len(image_points)} points in the image.\n")

            self.get_logger().info(f"\nОткрывается облако точек: {os.path.basename(pcd_path)}")
            cloud_points = self.pick_cloud_points(pcd_path)
            self.get_logger().info(f"\nSelected {len(cloud_points)} points in the cloud.\n")

            out_txt = os.path.join(self.data_dir, self.file)
            with open(out_txt, 'w') as f:
                f.write("# u, v, x, y, z\n")
                min_len = min(len(image_points), len(cloud_points))
                for i in range(min_len):
                    (u, v) = image_points[i]
                    (x, y, z) = cloud_points[i]
                    f.write(f"{u},{v},{x},{y},{z}\n")

            self.get_logger().info(f"Saved {min_len} correspondences in: {out_txt}")
            self.get_logger().info("========================================")

        self.get_logger().info("\nProcessing complete! Correspondences saved for all pairs.")


def main(args=None):
    rclpy.init(args=args)
    node = ImageCloudCorrespondenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
