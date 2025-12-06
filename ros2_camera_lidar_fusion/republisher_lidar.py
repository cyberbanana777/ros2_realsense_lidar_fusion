import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from builtin_interfaces.msg import Time

class TimeShiftRepublisher(Node):
    def __init__(self):
        super().__init__('time_shift_republisher')
        
        # Параметры
        self.declare_parameter('input_topic', 'sensors/livox/point_cloud2')
        self.declare_parameter('output_topic', 'sensors/livox/point_cloud2_sync')
        self.declare_parameter('time_shift_seconds', 10)
        
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.time_shift_seconds = self.get_parameter('time_shift_seconds').value
        
        # Подписчик и издатель
        self.subscription = self.create_subscription(
            PointCloud2,
            input_topic,
            self.callback,
            10)
        
        self.publisher = self.create_publisher(
            PointCloud2,
            output_topic,
            10)
        
        self.get_logger().info(
            f'Republisher initialized: {input_topic} -> {output_topic} '
            f'(time shift: -{self.time_shift_seconds} seconds)')
    
    def callback(self, msg):
        """
        Обработка входящего сообщения PointCloud2
        """
        # Создаем копию сообщения
        output_msg = msg
        
        # Вычитаем 10 секунд из временной метки заголовка
        new_time_sec = msg.header.stamp.sec - self.time_shift_seconds
        
        # Проверяем, чтобы время не стало отрицательным
        if new_time_sec < 0:
            self.get_logger().warn(
                f'Resulting timestamp would be negative: {new_time_sec} sec')
            new_time_sec = 0
        
        # Обновляем временную метку
        output_msg.header.stamp.sec = new_time_sec
        
        # Публикуем измененное сообщение
        self.publisher.publish(output_msg)
        
        # Логируем для отладки (можно закомментировать для производительности)
        self.get_logger().debug(
            f'Republished: {msg.header.stamp.sec}.{msg.header.stamp.nanosec} -> '
            f'{output_msg.header.stamp.sec}.{output_msg.header.stamp.nanosec}',
            throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    
    node = TimeShiftRepublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()