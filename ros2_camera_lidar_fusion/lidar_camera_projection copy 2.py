#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import yaml
from collections import deque
import threading

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge

from ros2_camera_lidar_fusion.read_yaml import extract_configuration


def load_extrinsic_matrix(yaml_path: str) -> np.ndarray:
    """Загрузка матрицы экстринсиков из YAML файла"""
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"No extrinsic file found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    if 'extrinsic_matrix' not in data:
        raise KeyError(f"YAML {yaml_path} has no 'extrinsic_matrix' key.")

    matrix_list = data['extrinsic_matrix']
    T = np.array(matrix_list, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("Extrinsic matrix is not 4x4.")
    return T


def load_camera_calibration(yaml_path: str) -> (np.ndarray, np.ndarray):
    """Загрузка калибровки камеры из YAML файла"""
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"No camera calibration file: {yaml_path}")

    with open(yaml_path, 'r') as f:
        calib_data = yaml.safe_load(f)

    cam_mat_data = calib_data['camera_matrix']['data']
    camera_matrix = np.array(cam_mat_data, dtype=np.float64)

    dist_data = calib_data['distortion_coefficients']['data']
    dist_coeffs = np.array(dist_data, dtype=np.float64).reshape((1, -1))

    return camera_matrix, dist_coeffs


def pointcloud2_to_xyz_array_fast(cloud_msg: PointCloud2, skip_rate: int = 1) -> np.ndarray:
    """Быстрое преобразование PointCloud2 в массив XYZ"""
    if cloud_msg.height == 0 or cloud_msg.width == 0:
        return np.zeros((0, 3), dtype=np.float32)

    field_names = [f.name for f in cloud_msg.fields]
    if not all(k in field_names for k in ('x', 'y', 'z')):
        return np.zeros((0, 3), dtype=np.float32)

    dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('_', 'V{}'.format(cloud_msg.point_step - 12))
    ])

    raw_data = np.frombuffer(cloud_msg.data, dtype=dtype)
    points = np.zeros((raw_data.shape[0], 3), dtype=np.float32)
    points[:, 0] = raw_data['x']
    points[:, 1] = raw_data['y']
    points[:, 2] = raw_data['z']

    if skip_rate > 1:
        points = points[::skip_rate]

    return points


class ScanBuffer:
    """Буфер для накопления лидарных сканов"""
    def __init__(self, max_scans: int = 50):
        self.max_scans = max_scans
        self.buffer = deque(maxlen=max_scans)
        self.lock = threading.Lock()
        
    def add_scan(self, points: np.ndarray):
        """Добавить новый скан в буфер"""
        with self.lock:
            self.buffer.append(points)
            
    def get_all_points(self) -> np.ndarray:
        """Получить все точки из буфера"""
        with self.lock:
            if not self.buffer:
                return np.zeros((0, 3), dtype=np.float32)
            
            all_points = []
            for scan in self.buffer:
                if len(scan) > 0:
                    all_points.append(scan)
            
            if not all_points:
                return np.zeros((0, 3), dtype=np.float32)
                
            return np.vstack(all_points)
    
    def clear(self):
        """Очистить буфер"""
        with self.lock:
            self.buffer.clear()
            
    def size(self) -> int:
        """Текущее количество сканов в буфере"""
        with self.lock:
            return len(self.buffer)
            
    def is_full(self) -> bool:
        """Буфер заполнен до максимального размера?"""
        with self.lock:
            return len(self.buffer) >= self.max_scans


class LidarCameraProjectionNode(Node):
    def __init__(self):
        super().__init__('lidar_camera_projection_node')
        
        # Загрузка конфигурации
        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        config_folder = config_file['general']['config_folder']
        extrinsic_yaml = config_file['general']['camera_extrinsic_calibration']
        extrinsic_yaml = os.path.join(config_folder, extrinsic_yaml)
        self.T_lidar_to_cam = load_extrinsic_matrix(extrinsic_yaml)

        camera_yaml = config_file['general']['camera_intrinsic_calibration']
        camera_yaml = os.path.join(config_folder, camera_yaml)
        self.camera_matrix, self.dist_coeffs = load_camera_calibration(camera_yaml)

        # Проверка калибровочных данных
        self.validate_calibration_data()
        
        # Параметры буферизации
        self.max_scans = self.declare_parameter('max_scans', 50).value
        self.min_scans_to_project = self.declare_parameter('min_scans_to_project', 10).value
        self.skip_rate = self.declare_parameter('skip_rate', 1).value
        self.debug_mode = self.declare_parameter('debug_mode', True).value
        
        # Топики
        lidar_topic = config_file['lidar']['lidar_topic']
        image_topic = config_file['camera']['image_topic']
        projected_topic = config_file['camera']['projected_topic']
        
        self.get_logger().info(f"Buffer configuration: max_scans={self.max_scans}, min_scans={self.min_scans_to_project}")
        self.get_logger().info(f"Subscribing to lidar topic: {lidar_topic}")
        self.get_logger().info(f"Subscribing to image topic: {image_topic}")
        self.get_logger().info(f"Publishing to topic: {projected_topic}")

        # Инициализация буфера
        self.scan_buffer = ScanBuffer(max_scans=self.max_scans)
        self.processing_lock = threading.Lock()
        
        # Подписка на лидар
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            lidar_topic,
            self.lidar_callback,
            10
        )
        
        # Подписка на изображение
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        # Публикатор для результата
        self.pub_image = self.create_publisher(Image, projected_topic, 1)
        self.bridge = CvBridge()
        
        # Статистика
        self.scan_count = 0
        
        self.get_logger().info(f"✅ Node initialized. Buffer size: {self.max_scans} scans")

    def validate_calibration_data(self):
        """Проверка калибровочных данных"""
        self.get_logger().info("🔍 Проверка калибровочных данных...")
        
        # Проверка матрицы экстринсиков
        if self.T_lidar_to_cam.shape != (4, 4):
            self.get_logger().error(f"Матрица экстринсиков имеет неправильную форму: {self.T_lidar_to_cam.shape}")
        
        # Проверка матрицы камеры
        if self.camera_matrix.shape != (3, 3):
            self.get_logger().error(f"Матрица камеры имеет неправильную форму: {self.camera_matrix.shape}")
        
        # Проверка типа данных
        if self.camera_matrix.dtype != np.float64:
            self.get_logger().warning(f"Матрица камеры имеет тип {self.camera_matrix.dtype}. Конвертирую в float64")
            self.camera_matrix = self.camera_matrix.astype(np.float64)
        
        if self.dist_coeffs.dtype != np.float64:
            self.get_logger().warning(f"Коэффициенты дисторсии имеют тип {self.dist_coeffs.dtype}. Конвертирую в float64")
            self.dist_coeffs = self.dist_coeffs.astype(np.float64)
        
        self.get_logger().info("✅ Калибровочные данные проверены")

    def lidar_callback(self, lidar_msg: PointCloud2):
        """Обработка нового лидарного скана"""
        self.scan_count += 1
        
        # Извлекаем точки из облака
        xyz_points = pointcloud2_to_xyz_array_fast(lidar_msg, skip_rate=self.skip_rate)
        
        if xyz_points.shape[0] == 0:
            self.get_logger().debug("Empty point cloud received")
            return
            
        # Добавляем скан в буфер
        self.scan_buffer.add_scan(xyz_points)
        
        # Логируем статистику
        if self.scan_count % 100 == 0:
            buffer_size = self.scan_buffer.size()
            total_points = self.scan_buffer.get_all_points().shape[0]
            self.get_logger().info(
                f"📊 Lidar stats: scans={self.scan_count}, "
                f"buffer={buffer_size}/{self.max_scans} scans, "
                f"total points={total_points}"
            )

    def image_callback(self, image_msg: Image):
        """Обработка нового кадра изображения"""
        start_time = self.get_clock().now()
        
        with self.processing_lock:
            # Получаем текущий кадр
            try:
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().error(f"Failed to convert image: {e}")
                return
            
            # Получаем все накопленные точки из буфера
            all_points = self.scan_buffer.get_all_points()
            buffer_size = self.scan_buffer.size()
            
            # Добавляем тестовые точки в режиме отладки
            if self.debug_mode and buffer_size > 0:
                self.get_logger().info("🔍 Добавляю тестовые точки для отладки")
                all_points = self.add_test_points(all_points)
            
            if all_points.shape[0] == 0 or buffer_size < self.min_scans_to_project:
                self.get_logger().debug(
                    f"Not enough data: {buffer_size} scans, {all_points.shape[0]} points"
                )
                # Публикуем исходное изображение без проекции
                out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
                out_msg.header = image_msg.header
                self.pub_image.publish(out_msg)
                return
            
            # Проецируем все накопленные точки
            projected_image = self.project_points_to_image(cv_image, all_points)
            
            # Публикуем результат
            out_msg = self.bridge.cv2_to_imgmsg(projected_image, encoding='bgr8')
            out_msg.header = image_msg.header
            self.pub_image.publish(out_msg)
            
            # Логируем производительность
            processing_time = (self.get_clock().now() - start_time).nanoseconds / 1e6
            self.get_logger().debug(
                f"🖼️ Projected {all_points.shape[0]} points from {buffer_size} scans "
                f"in {processing_time:.1f} ms"
            )

    def project_points_to_image(self, cv_image: np.ndarray, xyz_lidar: np.ndarray) -> np.ndarray:
        """Проецирует точки лидара на изображение"""
        n_points = xyz_lidar.shape[0]
        
        self.get_logger().info(f"📊 Начало проекции: {n_points} точек из лидара")
        
        # Проверка входных данных
        if n_points == 0:
            self.get_logger().warning("Нет точек для проекции!")
            return cv_image.copy()
        
        # Тщательная проверка и очистка данных
        self.get_logger().debug("🔍 Проверка данных...")
        
        # 1. Проверка на NaN и Inf
        nan_mask = np.any(np.isnan(xyz_lidar), axis=1)
        inf_mask = np.any(np.isinf(xyz_lidar), axis=1)
        invalid_mask = nan_mask | inf_mask
        
        if np.any(invalid_mask):
            self.get_logger().warning(f"Найдено {np.sum(invalid_mask)} невалидных точек (NaN/Inf)")
            xyz_lidar = xyz_lidar[~invalid_mask]
            n_points = xyz_lidar.shape[0]
            self.get_logger().info(f"После очистки NaN/Inf: {n_points} точек")
        
        if n_points == 0:
            self.get_logger().warning("Нет валидных точек после очистки!")
            return cv_image.copy()
        
        # 2. Проверка типа данных (должен быть float32 или float64)
        if xyz_lidar.dtype not in [np.float32, np.float64]:
            self.get_logger().warning(f"Неправильный тип данных: {xyz_lidar.dtype}. Конвертирую в float64")
            xyz_lidar = xyz_lidar.astype(np.float64)
        
        # 3. Преобразование в однородные координаты
        xyz_lidar_f64 = xyz_lidar.astype(np.float64)
        ones = np.ones((n_points, 1), dtype=np.float64)
        xyz_lidar_h = np.hstack((xyz_lidar_f64, ones))
        
        # 4. Преобразование в систему координат камеры
        xyz_cam_h = xyz_lidar_h @ self.T_lidar_to_cam.T
        xyz_cam = xyz_cam_h[:, :3]
        
        # 5. Проверяем результат преобразования
        self.get_logger().info("📐 Статистика после преобразования:")
        self.get_logger().info(f"  X: [{xyz_cam[:, 0].min():.2f}, {xyz_cam[:, 0].max():.2f}]")
        self.get_logger().info(f"  Y: [{xyz_cam[:, 1].min():.2f}, {xyz_cam[:, 1].max():.2f}]")
        self.get_logger().info(f"  Z: [{xyz_cam[:, 2].min():.2f}, {xyz_cam[:, 2].max():.2f}]")
        
        # 6. Считаем точки с разными знаками Z
        z_positive = np.sum(xyz_cam[:, 2] > 0)
        z_negative = np.sum(xyz_cam[:, 2] < 0)
        z_zero = np.sum(xyz_cam[:, 2] == 0)
        
        self.get_logger().info(f"📊 Распределение по Z:")
        self.get_logger().info(f"  Z > 0 (перед камерой): {z_positive} точек")
        self.get_logger().info(f"  Z < 0 (позади камеры): {z_negative} точек")
        self.get_logger().info(f"  Z = 0 (на плоскости): {z_zero} точек")
        
        # 7. Используем ВСЕ точки (без фильтрации по z > 0)
        xyz_cam_front = xyz_cam
        
        if xyz_cam_front.shape[0] == 0:
            self.get_logger().info("Нет точек для проекции.")
            return cv_image.copy()
        
        # 8. Проекция на изображение
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        
        # Гарантируем правильный формат для OpenCV
        points_for_projection = xyz_cam_front.astype(np.float64)
        
        try:
            # Проекция
            self.get_logger().info("🔄 Проецирование точек...")
            image_points, _ = cv2.projectPoints(
                points_for_projection,
                rvec, tvec,
                self.camera_matrix,
                self.dist_coeffs
            )
            
            # Преобразуем результат
            image_points = image_points.reshape(-1, 2)
            self.get_logger().info(f"✅ Успешно спроецировано: {image_points.shape[0]} точек")
            
        except cv2.error as e:
            self.get_logger().error(f"❌ Ошибка OpenCV при проекции: {e}")
            
            # Детальная диагностика
            self.get_logger().error(f"Данные shape: {points_for_projection.shape}")
            self.get_logger().error(f"Данные dtype: {points_for_projection.dtype}")
            self.get_logger().error(f"Матрица камеры dtype: {self.camera_matrix.dtype}")
            self.get_logger().error(f"Коэффициенты дисторсии dtype: {self.dist_coeffs.dtype}")
            
            # Пробуем альтернативный метод проекции
            self.get_logger().info("🔄 Пробую альтернативный метод проекции...")
            return self.project_points_manual(cv_image, xyz_cam_front)
        
        # 9. Отрисовка точек на изображении
        result_image = cv_image.copy()
        h, w = result_image.shape[:2]
        points_drawn = 0
        points_outside = 0
        
        # Логирование статистики проекции
        u_coords = image_points[:, 0]
        v_coords = image_points[:, 1]
        
        self.get_logger().info(f"📏 Диапазон проекции:")
        self.get_logger().info(f"  U: [{u_coords.min():.1f}, {u_coords.max():.1f}]")
        self.get_logger().info(f"  V: [{v_coords.min():.1f}, {v_coords.max():.1f}]")
        self.get_logger().info(f"  Изображение: {w}x{h}")
        
        # Цветовая схема в зависимости от знака Z
        z_vals = xyz_cam_front[:, 2]
        
        for i, (u, v) in enumerate(image_points):
            u_int = int(round(u))
            v_int = int(round(v))
            
            if 0 <= u_int < w and 0 <= v_int < h:
                # Цвет зависит от знака Z
                z_val = z_vals[i]
                
                if z_val > 0:
                    color = (0, 255, 0)  # зеленый - перед камерой
                elif z_val < 0:
                    color = (0, 0, 255)  # красный - позади камеры
                else:
                    color = (255, 255, 0)  # желтый - на плоскости камеры
                
                # Размер точки в зависимости от абсолютного значения Z
                radius = max(1, int(3 * (1.0 - min(abs(z_val) / 30.0, 1.0))))
                
                cv2.circle(result_image, (u_int, v_int), radius, color, -1)
                points_drawn += 1
            else:
                points_outside += 1
        
        self.get_logger().info(f"🎯 Результат отрисовки:")
        self.get_logger().info(f"  Нарисовано на изображении: {points_drawn}")
        self.get_logger().info(f"  Вне границ кадра: {points_outside}")
        self.get_logger().info(f"  Всего спроецировано: {len(image_points)}")
        
        # Добавляем информационный текст
        cv2.putText(result_image, 
                   f"Scans: {self.scan_buffer.size()}/{self.max_scans}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_image, 
                   f"Points: {points_drawn}/{len(image_points)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_image,
                   f"Z: [{xyz_cam[:, 2].min():.1f}, {xyz_cam[:, 2].max():.1f}]",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Показываем границы кадра
        cv2.rectangle(result_image, (0, 0), (w-1, h-1), (255, 255, 255), 1)
        
        return result_image

    def project_points_manual(self, cv_image: np.ndarray, xyz_cam: np.ndarray) -> np.ndarray:
        """Альтернативный метод проекции точек (ручная реализация)"""
        self.get_logger().info("🔄 Использую ручную проекцию...")
        
        result_image = cv_image.copy()
        h, w = result_image.shape[:2]
        
        # Упрощенная проекция (без учета дисторсии)
        valid_mask = xyz_cam[:, 2] != 0  # Избегаем деления на ноль
        if np.sum(valid_mask) == 0:
            return result_image
        
        xyz_valid = xyz_cam[valid_mask]
        
        # Нормализованные координаты
        x_norm = xyz_valid[:, 0] / xyz_valid[:, 2]
        y_norm = xyz_valid[:, 1] / xyz_valid[:, 2]
        
        # Проекция с использованием матрицы камеры
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        u_coords = fx * x_norm + cx
        v_coords = fy * y_norm + cy
        
        points_drawn = 0
        z_vals = xyz_valid[:, 2]
        
        for i in range(len(u_coords)):
            u_int = int(round(u_coords[i]))
            v_int = int(round(v_coords[i]))
            
            if 0 <= u_int < w and 0 <= v_int < h:
                # Цвет зависит от знака Z
                z_val = z_vals[i]
                
                if z_val > 0:
                    color = (0, 200, 0)  # темно-зеленый
                elif z_val < 0:
                    color = (0, 0, 200)  # темно-красный
                else:
                    color = (200, 200, 0)  # темно-желтый
                
                cv2.circle(result_image, (u_int, v_int), 2, color, -1)
                points_drawn += 1
        
        self.get_logger().info(f"🎯 Ручная проекция: нарисовано {points_drawn} точек")
        return result_image

    def add_test_points(self, xyz_lidar: np.ndarray) -> np.ndarray:
        """Добавляет тестовые точки для проверки проекции"""
        test_points = []
        
        # Тестовые точки в системе лидара
        for x in [1, 5, 10, 20]:  # метры вперед
            test_points.append([x, 0, 0])      # прямо вперед
            test_points.append([x, 1, 0])      # вперед и влево
            test_points.append([x, -1, 0])     # вперед и вправо
            test_points.append([x, 0, 1])      # вперед и вверх
            test_points.append([x, 0, -1])     # вперед и вниз
        
        test_array = np.array(test_points, dtype=np.float32)
        
        if xyz_lidar.shape[0] == 0:
            return test_array
        
        # Добавляем тестовые точки к реальным данным
        return np.vstack([xyz_lidar, test_array])

    def reset_buffer(self):
        """Сбросить буфер"""
        self.scan_buffer.clear()
        self.get_logger().info("Buffer cleared")


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraProjectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()