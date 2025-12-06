import rosbag2_py
import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import numpy as np
from collections import defaultdict

def calculate_time_offset(bag_path):
    """Расчет смещения между лидаром и камерой с фильтрацией метаданных"""
    
    rclpy.init()
    
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id='sqlite3'
    )
    
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    reader = rosbag2_py.SequentialReader()
    
    try:
        reader.open(storage_options, converter_options)
        
        # Получаем информацию о топиках
        topic_types = reader.get_all_topics_and_types()
        topic_type_map = {topic.name: topic.type for topic in topic_types}
        
        print("Найденные топики:")
        for topic_name, topic_type in topic_type_map.items():
            print(f"  - {topic_name} ({topic_type})")
        
        # Автоматически определяем топики для анализа
        # Ищем топики лидара (PointCloud2) и камеры (Image, но не metadata)
        lidar_topics = []
        image_topics = []
        
        for topic_name, topic_type in topic_type_map.items():
            if 'PointCloud2' in topic_type:
                lidar_topics.append(topic_name)
            elif 'Image' in topic_type:
                # Фильтруем метаданные - оставляем только топики с изображениями
                if 'metadata' not in topic_name.lower():
                    image_topics.append(topic_name)
        
        print(f"\nВыбранные для анализа:")
        print(f"  Лидар топики: {lidar_topics}")
        print(f"  Изображения топики: {image_topics}")
        
        if not lidar_topics or not image_topics:
            print("Ошибка: не найдены подходящие топики")
            return None
        
        # Собираем временные метки
        lidar_times = []
        image_times = []
        
        msg_count = 0
        lidar_count = 0
        image_count = 0
        error_count = 0
        
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            msg_count += 1
            
            # Определяем тип сообщения
            msg_type = topic_type_map.get(topic)
            if not msg_type:
                continue
            
            try:
                # Для лидара
                if topic in lidar_topics:
                    msg = deserialize_message(data, PointCloud2)
                    time_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    lidar_times.append(time_sec)
                    lidar_count += 1
                
                # Для изображений (но не метаданных)
                elif topic in image_topics:
                    msg = deserialize_message(data, Image)
                    time_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    image_times.append(time_sec)
                    image_count += 1
                    
            except Exception as e:
                error_count += 1
                if error_count <= 5:  # Выводим только первые 5 ошибок
                    print(f"Предупреждение: ошибка для топика {topic}: {str(e)[:100]}")
                continue
        
        print(f"\nСтатистика обработки:")
        print(f"  Всего сообщений: {msg_count}")
        print(f"  Успешно обработано лидар: {lidar_count}")
        print(f"  Успешно обработано изображений: {image_count}")
        print(f"  Ошибок десериализации: {error_count}")
        
        if not lidar_times or not image_times:
            print("Ошибка: не найдены данные для анализа")
            return None
        
        # Сортируем временные метки
        lidar_times.sort()
        image_times.sort()
        
        print(f"\nВременные диапазоны:")
        print(f"  Лидар: {lidar_times[0]:.3f} - {lidar_times[-1]:.3f} сек")
        print(f"  Изображения: {image_times[0]:.3f} - {image_times[-1]:.3f} сек")
        print(f"  Длительность лидара: {lidar_times[-1] - lidar_times[0]:.2f} сек")
        print(f"  Длительность камеры: {image_times[-1] - image_times[0]:.2f} сек")
        
        # Вычисляем смещение несколькими методами
        print(f"\nРасчет смещения:")
        
        # 1. Разница первых сообщений
        offset_first = lidar_times[0] - image_times[0]
        print(f"  По первым сообщениям: {offset_first:.6f} сек")
        
        # 2. Разница последних сообщений
        offset_last = lidar_times[-1] - image_times[-1]
        print(f"  По последним сообщениям: {offset_last:.6f} сек")
        
        # 3. Разница средних
        offset_mean = np.mean(lidar_times) - np.mean(image_times)
        print(f"  По средним: {offset_mean:.6f} сек")
        
        # 4. Создаем гистограммы и находим смещение по максимуму корреляции
        # Это наиболее надежный метод для данных с разной частотой
        time_min = min(min(lidar_times), min(image_times))
        time_max = max(max(lidar_times), max(image_times))
        
        # Нормализуем временные метки
        lidar_norm = [t - time_min for t in lidar_times]
        image_norm = [t - time_min for t in image_times]
        
        # Создаем бины (10000 бинов на весь диапазон)
        n_bins = 10000
        bins = np.linspace(0, time_max - time_min, n_bins)
        
        # Гистограммы
        lidar_hist, _ = np.histogram(lidar_norm, bins=bins)
        image_hist, _ = np.histogram(image_norm, bins=bins)
        
        # Кросс-корреляция
        correlation = np.correlate(lidar_hist, image_hist, mode='full')
        lags = np.arange(-len(image_hist) + 1, len(lidar_hist))
        
        # Находим максимальную корреляцию
        max_corr_idx = np.argmax(correlation)
        optimal_lag = lags[max_corr_idx]
        
        # Конвертируем лаг в секунды
        bin_width = bins[1] - bins[0]
        offset_correlation = optimal_lag * bin_width
        
        print(f"  По корреляции: {offset_correlation:.6f} сек")
        print(f"  (максимальная корреляция: {correlation[max_corr_idx]:.2f})")
        
        # Собираем все оценки
        offsets = [offset_first, offset_last, offset_mean, offset_correlation]
        
        # Фильтруем выбросы (разница > 20 секунд)
        offsets_filtered = [o for o in offsets if abs(o) < 20]
        
        if not offsets_filtered:
            offsets_filtered = offsets
        
        # Берем медиану
        final_offset = np.median(offsets_filtered)
        
        print(f"\n" + "="*60)
        print(f"РЕЗУЛЬТАТ АНАЛИЗА:")
        print(f"  Рекомендуемое смещение: {final_offset:.6f} секунд")
        
        if final_offset > 0:
            print(f"  Лидар опережает камеру на {final_offset:.3f} сек")
            print(f"  (вычитайте {final_offset:.3f} из времени лидара)")
        else:
            print(f"  Камера опережает лидар на {abs(final_offset):.3f} сек")
            print(f"  (прибавляйте {abs(final_offset):.3f} к времени лидара)")
        
        print(f"\nПроверка:")
        print(f"  Лидар (первое): {lidar_times[0]:.6f}")
        print(f"  Камера (первое): {image_times[0]:.6f}")
        print(f"  Сырая разница: {lidar_times[0] - image_times[0]:.6f}")
        print(f"  После коррекции: {(lidar_times[0] - final_offset) - image_times[0]:.6f}")
        
        # Проверяем стабильность смещения
        print(f"\nСтабильность смещения (первые 10 пар):")
        for i in range(min(10, len(lidar_times), len(image_times))):
            diff = lidar_times[i] - image_times[i]
            print(f"  Сообщение {i}: {diff:.6f} сек")
        
        print("="*60)
        
        return final_offset
        
    except Exception as e:
        print(f"Ошибка при обработке: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        rclpy.shutdown()

# ВЕРСИЯ 2: Только анализ временных меток без десериализации сообщений
def analyze_bag_fast(bag_path):
    """Быстрый анализ через чтение заголовков"""
    
    import struct
    
    rclpy.init()
    
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('cdr', 'cdr')
    
    reader = rosbag2_py.SequentialReader()
    
    try:
        reader.open(storage_options, converter_options)
        
        # Получаем информацию о топиках
        topic_types = reader.get_all_topics_and_types()
        topic_type_map = {topic.name: topic.type for topic in topic_types}
        
        print("Найденные топики:")
        for topic_name in topic_type_map:
            print(f"  - {topic_name}")
        
        # Автоматический выбор топиков
        lidar_topics = [t for t in topic_type_map if 'point' in t.lower() or 'cloud' in t.lower() or 'ouster' in t.lower()]
        image_topics = [t for t in topic_type_map if ('image' in t.lower() or 'camera' in t.lower()) 
                       and 'metadata' not in t.lower() and 'info' not in t.lower()]
        
        # Если не нашли автоматически, используем все подходящие по типу
        if not lidar_topics:
            lidar_topics = [t for t, tp in topic_type_map.items() if 'PointCloud2' in tp]
        if not image_topics:
            image_topics = [t for t, tp in topic_type_map.items() if 'Image' in tp and 'metadata' not in t]
        
        print(f"\nАнализируем:")
        print(f"  Лидар: {lidar_topics}")
        print(f"  Изображения: {image_topics}")
        
        # Собираем временные метки
        lidar_times = []
        image_times = []
        
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            
            # Для ускорения читаем только начало сообщения, где находится header
            if len(data) < 24:  # Минимальный размер для чтения timestamp
                continue
            
            try:
                # Пытаемся прочитать timestamp из бинарных данных
                # В ROS2 сообщениях stamp обычно в начале
                # Читаем sec (4 байта) и nanosec (4 байта)
                sec = struct.unpack_from('<I', data, 8)[0]  # Смещение может отличаться
                nsec = struct.unpack_from('<I', data, 12)[0]
                time_sec = sec + nsec * 1e-9
                
                if topic in lidar_topics:
                    lidar_times.append(time_sec)
                elif topic in image_topics:
                    image_times.append(time_sec)
                    
            except:
                # Если не удалось прочитать бинарно, пропускаем
                continue
        
        if not lidar_times or not image_times:
            print("Недостаточно данных")
            return None
        
        # Вычисляем смещение
        offset = np.median(lidar_times) - np.median(image_times)
        
        print(f"\nБыстрый анализ:")
        print(f"  Лидар сообщений: {len(lidar_times)}")
        print(f"  Камера сообщений: {len(image_times)}")
        print(f"  Смещение: {offset:.6f} сек")
        
        return offset
        
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    bag_path = "/home/moody/Alise_work/rosbag2_2025_12_06-14_52_10"
    
    print("="*60)
    print("АНАЛИЗ ВРЕМЕННОГО СМЕЩЕНИЯ ЛИДАР-КАМЕРА")
    print("="*60)
    
    # Запускаем подробный анализ
    offset = calculate_time_offset(bag_path)
    
    if offset is not None:
        print(f"\n✅ Используйте это значение в вашем ROS2 узле:")
        print(f"self.time_offset = {offset:.6f}  # секунд")
        
        print(f"\nПример использования в колбэке:")
        print(f"def sync_callback(self, image_msg, lidar_msg):")
        print(f"    # Корректируем время лидара")
        print(f"    lidar_time = lidar_msg.header.stamp")
        print(f"    corrected_sec = lidar_time.sec")
        print(f"    corrected_nsec = lidar_time.nanosec - int({offset:.6f} * 1e9)")
        print(f"    # Или просто используем время камеры как эталон")
        print(f"    lidar_msg.header.stamp = image_msg.header.stamp")
    else:
        print("\n❌ Не удалось вычислить смещение")