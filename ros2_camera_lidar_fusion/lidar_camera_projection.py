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


class PointSelectionManager:
    """Менеджер для выбора точек кликом мыши"""
    def __init__(self, window_name="Lidar Projection"):
        self.window_name = window_name
        self.selected_point = None
        self.last_click = None
        self.projection_map = {}  # (u, v) -> list of (lidar_coords, camera_coords)
        self.click_callback = None
        
        # Создаем окно и устанавливаем обработчик мыши
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Обработчик событий мыши"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.last_click = (x, y)
            self.selected_point = self.find_nearest_projected_point(x, y)
            
            if self.selected_point:
                lidar_coords, _ = self.selected_point
                print(f"\n{'='*50}")
                print(f"CLICK: Image coordinates: ({x}, {y})")
                print(f"NEAREST: Projected point: {self.selected_point[0]}")
                print(f"LIDAR COORDS: x={lidar_coords[0]:.3f}m, y={lidar_coords[1]:.3f}m, z={lidar_coords[2]:.3f}m")
                print(f"{'='*50}")
                
                if self.click_callback:
                    self.click_callback(lidar_coords, (x, y))
    
    def find_nearest_projected_point(self, x, y, max_distance=15):
        """Найти ближайшую спроецированную точку к клику"""
        if not self.projection_map:
            return None
        
        nearest_point = None
        min_dist = float('inf')
        
        # Простой линейный поиск (достаточно быстрый для тысяч точек)
        for (u, v), points_list in self.projection_map.items():
            dist = (u - x)**2 + (v - y)**2  # квадрат расстояния
            
            if dist < min_dist and dist <= max_distance**2:
                min_dist = dist
                # Берем первую точку из списка (обычно это ближайшая к камере)
                nearest_point = points_list[0]
        
        return nearest_point
    
    def update_projection_map(self, image_points, lidar_points, camera_points):
        """Обновить карту проекции"""
        self.projection_map.clear()
        
        # Группируем точки по пиксельным координатам
        for img_pt, lidar_pt, cam_pt in zip(image_points, lidar_points, camera_points):
            u, v = int(round(img_pt[0])), int(round(img_pt[1]))
            key = (u, v)
            
            if key not in self.projection_map:
                self.projection_map[key] = []
            
            # Сохраняем все точки, проецирующиеся в этот пиксель
            self.projection_map[key].append((lidar_pt, cam_pt))
        
        # Сортируем точки по расстоянию от камеры (ближайшие первые)
        for key in self.projection_map:
            self.projection_map[key].sort(key=lambda x: x[1][2])  # сортировка по Z в системе камеры
    
    def draw_selection(self, image):
        """Нарисовать выделение на изображении"""
        if self.selected_point and self.last_click:
            lidar_coords, _ = self.selected_point
            
            # Найти точку в projection_map, соответствующую выбранным координатам
            for (u, v), points_list in self.projection_map.items():
                for pt_lidar, pt_cam in points_list:
                    if np.allclose(pt_lidar, lidar_coords, atol=0.001):
                        # Рисуем выделение
                        cv2.circle(image, (u, v), 8, (0, 255, 255), 2)  # желтый кружок
                        cv2.circle(image, (u, v), 3, (255, 255, 255), -1)  # белая точка
                        
                        # Линия от клика к точке
                        cv2.line(image, self.last_click, (u, v), (255, 200, 0), 1, cv2.LINE_AA)
                        
                        # Текст с координатами
                        text = f"({lidar_coords[0]:.1f}, {lidar_coords[1]:.1f}, {lidar_coords[2]:.1f})"
                        cv2.putText(image, text, (u + 10, v - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        break
        return image


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
        self.show_window = self.declare_parameter('show_window', True).value
        
        # Топики
        lidar_topic = config_file['lidar']['lidar_topic']
        image_topic = config_file['camera']['image_topic']
        projected_topic = config_file['camera']['projected_topic']
        
        # Инициализация менеджера выбора точек
        if self.show_window:
            self.selection_manager = PointSelectionManager()
            self.selection_manager.click_callback = self.on_point_selected
            self.get_logger().info("Интерактивное окно включено. Кликните по изображению для выбора точки.")
        else:
            self.selection_manager = None
        
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
        
        # Для хранения данных проекции
        self.current_projection_data = None
        
        self.get_logger().info(f"✅ Node initialized. Interactive mode: {self.show_window}")

    def validate_calibration_data(self):
        """Проверка калибровочных данных"""
        self.get_logger().info("🔍 Проверка калибровочных данных...")
        
        # Проверка матрицы экстринсиков
        if self.T_lidar_to_cam.shape != (4, 4):
            self.get_logger().error(f"Матрица экстринсиков имеет неправильную форму: {self.T_lidar_to_cam.shape}")
        
        # Проверка матрицы камеры
        if self.camera_matrix.shape != (3, 3):
            self.get_logger().error(f"Матрица камеры имеет неправильную форму: {self.camera_matrix.shape}")
        
        self.get_logger().info("✅ Калибровочные данные проверены")

    def on_point_selected(self, lidar_coords, image_coords):
        """Callback при выборе точки"""
        self.get_logger().info(f"Выбрана точка: лидар=({lidar_coords[0]:.2f}, {lidar_coords[1]:.2f}, {lidar_coords[2]:.2f}), "
                              f"изображение=({image_coords[0]}, {image_coords[1]})")

    def lidar_callback(self, lidar_msg: PointCloud2):
        """Обработка нового лидарного скана"""
        # Извлекаем точки из облака
        xyz_points = pointcloud2_to_xyz_array_fast(lidar_msg, skip_rate=self.skip_rate)
        
        if xyz_points.shape[0] == 0:
            return
            
        # Добавляем скан в буфер
        self.scan_buffer.add_scan(xyz_points)

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
            
            if all_points.shape[0] == 0 or buffer_size < self.min_scans_to_project:
                # Публикуем исходное изображение
                out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
                out_msg.header = image_msg.header
                self.pub_image.publish(out_msg)
                return
            
            # Проецируем все накопленные точки
            projected_image, projection_data = self.project_points_to_image(cv_image, all_points)
            
            # Сохраняем данные проекции для выбора точек
            self.current_projection_data = projection_data
            
            # Обновляем карту проекции для менеджера выбора
            if self.selection_manager and projection_data:
                self.selection_manager.update_projection_map(
                    projection_data['image_points'],
                    projection_data['lidar_points'],
                    projection_data['camera_points']
                )
            
            # Добавляем выделение, если есть выбранная точка
            if self.selection_manager:
                projected_image = self.selection_manager.draw_selection(projected_image)
            
            # Публикуем результат
            out_msg = self.bridge.cv2_to_imgmsg(projected_image, encoding='bgr8')
            out_msg.header = image_msg.header
            self.pub_image.publish(out_msg)
            
            # Показываем окно (блокирующий вызов, но короткий)
            if self.selection_manager:
                cv2.imshow(self.selection_manager.window_name, projected_image)
                cv2.waitKey(1)  # короткая задержка для обработки событий
            
            # Логируем производительность
            processing_time = (self.get_clock().now() - start_time).nanoseconds / 1e6
            if buffer_size % 10 == 0:
                self.get_logger().info(
                    f"Проецировано {all_points.shape[0]} точек за {processing_time:.1f} мс"
                )

    def project_points_to_image(self, cv_image: np.ndarray, xyz_lidar: np.ndarray):
        """Проецирует точки лидара на изображение"""
        n_points = xyz_lidar.shape[0]
        
        if n_points == 0:
            return cv_image.copy(), None
        
        # Конвертируем в float64 для точности
        xyz_lidar_f64 = xyz_lidar.astype(np.float64)
        
        # Преобразуем в систему координат камеры
        ones = np.ones((n_points, 1), dtype=np.float64)
        xyz_lidar_h = np.hstack((xyz_lidar_f64, ones))
        xyz_cam_h = xyz_lidar_h @ self.T_lidar_to_cam.T
        xyz_cam = xyz_cam_h[:, :3]
        
        # Фильтруем точки перед камерой (Z > 0)
        # front_mask = xyz_cam[:, 2] > 0
        # if not np.any(front_mask):
        #     return cv_image.copy(), None
        
        
        xyz_cam_front = xyz_cam
        xyz_lidar_front = xyz_lidar_f64
        
        # Проецируем точки
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        
        image_points, _ = cv2.projectPoints(
            xyz_cam_front.reshape(-1, 1, 3),
            rvec, tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        image_points = image_points.reshape(-1, 2)
        
        # Подготавливаем данные для менеджера выбора
        projection_data = {
            'image_points': image_points,
            'lidar_points': xyz_lidar_front,
            'camera_points': xyz_cam_front
        }
        
        # Отрисовываем точки
        result_image = cv_image.copy()
        h, w = result_image.shape[:2]
        
        # Рисуем только точки в пределах кадра
        for i, (u, v) in enumerate(image_points):
            u_int = int(round(u))
            v_int = int(round(v))
            
            if 0 <= u_int < w and 0 <= v_int < h:
                # Цвет в зависимости от расстояния
                distance = xyz_cam_front[i, 2]
                intensity = int(255 * (1.0 - min(distance / 30.0, 1.0)))
                color = (0, intensity, 255 - intensity)  # от синего к красному
                
                cv2.circle(result_image, (u_int, v_int), 1, color, -1)
        
        # Информация о количестве точек
        cv2.putText(result_image, 
                   f"Points: {len(image_points)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(result_image,
                   f"Buffer: {self.scan_buffer.size()}/{self.max_scans}",
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return result_image, projection_data

    def destroy_node(self):
        """Корректное завершение работы"""
        if self.selection_manager:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraProjectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Завершение работы...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()