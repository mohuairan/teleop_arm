#!/usr/bin/env python3
"""
ROS 2 14关节角度发布节点 (Publisher)

此脚本周期性地发布预定义的 14 个关节角度序列到 /planning/joint_angles 话题，
供 JointAngleSubscriber 节点订阅和执行。
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState # 用于发布角度数据

# =========================================================================
# 配置参数
# =========================================================================
PLANNING_TOPIC = "/planning/joint_angles" 
PUBLISH_FREQUENCY_HZ = 0.2 
PI = 3.1415926

# 14 个关节的名称 (左臂 7 个，右臂 7 个)
ALL_JOINT_NAMES = [
    "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6", "left_j7",
    "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_j7",
]

# 预定义的 14 关节目标位置序列 (弧度)
# [L1, L2, L3, L4, L5, L6, L7, R1, R2, R3, R4, R5, R6, R7]
POSES_SEQUENCE = [
    # 1. 初始/零位
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    
    # 2. 左臂前伸，右臂后缩 (协调动作 1)
    #[0.5, -0.8, 0.0, 1.5, 0.0, 0.0, 0.0, -0.5, 0.8, 0.0, -1.5, 0.0, 0.0, 0.0],

    #自定义初始角度
    [-2.0, 0.85, -0.7, -1.42, 0.64, 0.64, 0.0, 2.0-PI, 0.85, 0.7-PI, 1.42, -0.64, 0.64, 0.0],

    # 3. 双臂同时向上举
    #[0.0, -1.2, 0.5, 1.5, -0.3, 0.0, 0.0, 0.0, -1.2, 0.5, 1.5, -0.3, 0.0, 0.0],
    
    # 4. 回到初始位置
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]

# =========================================================================

class AnglePublisherNode(Node):
    """周期性发布 14 关节角度的节点"""

    def __init__(self):
        super().__init__("angle_publisher_node")
        self.get_logger().info("初始化 14 关节角度发布节点...")
        
        # 创建发布器
        self.publisher = self.create_publisher(JointState, PLANNING_TOPIC, 10)
        self.get_logger().info(f"发布话题: {PLANNING_TOPIC}")

        # 计时器设置
        timer_period = 1.0 / PUBLISH_FREQUENCY_HZ
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.pose_index = 0
        self.total_poses = len(POSES_SEQUENCE)

    def timer_callback(self):
        """计时器回调函数，用于发布下一个目标位置"""
        
        # 1. 获取当前目标位置
        target_positions = POSES_SEQUENCE[self.pose_index]
        
        # 2. 构造 JointState 消息
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ALL_JOINT_NAMES # 理论上可以省略，但包含名称是好习惯
        msg.position = target_positions
        
        # 3. 发布消息
        self.publisher.publish(msg)
        self.get_logger().info(f'发布目标 {self.pose_index + 1}/{self.total_poses}: {[f"{v:.2f}" for v in target_positions]}')
        
        # 4. 更新索引，循环回到第一个位置
        self.pose_index = (self.pose_index + 1) % self.total_poses


def main(args=None):
    rclpy.init(args=args)
    publisher = AnglePublisherNode()
    
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        publisher.get_logger().info("发布程序被用户中断。")
    finally:
        publisher.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()