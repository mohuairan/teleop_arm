#!/usr/bin/env python3
"""
双臂关节角度指令接收器 (Subscriber/Publisher)

此脚本订阅外部发布的 14 个关节角度，将它们拆分为左右臂，
并发布到相应的 JointTrajectoryController Topic 接口，驱动双臂运动。

假设:
1. 订阅话题: /planning/joint_angles (需要您外部节点发布此话题)
2. 消息类型: sensor_msgs/msg/JointState (或其他包含 14 个浮点数的数组消息)
3. 左右臂关节名称已在代码中定义 (left_j1...left_j7, right_j1...right_j7)
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# 假设您的外部发布者使用 JointState 消息发布 14 个角度
from sensor_msgs.msg import JointState 

# =========================================================================
# 配置参数
# =========================================================================
PLANNING_TOPIC = "/planning/joint_angles" 
LEFT_CONTROLLER_TOPIC = "/left_arm_controller/joint_trajectory" 
RIGHT_CONTROLLER_TOPIC = "/right_arm_controller/joint_trajectory" 
NUM_JOINTS_PER_ARM = 7
TIME_TO_REACH_SEC = 3.0 # 目标运动执行时间

class JointAngleSubscriber(Node):
    """接收 14 关节角度并分发给左右臂控制器的节点"""

    def __init__(self):
        super().__init__("joint_angle_subscriber")
        self.get_logger().info("初始化 14 关节角度指令接收器...")
        
        # 关节名称 (从原模板中获取)
        self.left_joint_names = [
            "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6", "left_j7",
        ]
        self.right_joint_names = [
            "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_j7",
        ]
        
        # 1. 创建左右臂控制器的话题发布器
        self.left_publisher = self.create_publisher(
            JointTrajectory, LEFT_CONTROLLER_TOPIC, 10
        )
        self.right_publisher = self.create_publisher(
            JointTrajectory, RIGHT_CONTROLLER_TOPIC, 10
        )
        self.get_logger().info(f"左臂发布器: {LEFT_CONTROLLER_TOPIC}")
        self.get_logger().info(f"右臂发布器: {RIGHT_CONTROLLER_TOPIC}")


        # 2. 创建接收 14 目标角度的订阅器
        # ⚠️ 注意: 确保 /planning/joint_angles 话题的消息类型与此处的 JointState 匹配
        self.subscription = self.create_subscription(
            JointState,
            PLANNING_TOPIC,
            self.angle_callback,
            10
        )
        self.get_logger().info(f"订阅规划话题: {PLANNING_TOPIC}")


    def angle_callback(self, msg: JointState):
        """订阅到新的 14 个关节角度时的回调函数"""
        
        target_positions = msg.position
        
        # 1. 验证数据完整性
        if len(target_positions) != 2 * NUM_JOINTS_PER_ARM:
            self.get_logger().warn(
                f"收到的关节数不匹配！预期 14 个，收到 {len(target_positions)} 个。"
            )
            return
        
        # 2. 拆分左右臂角度
        # 假设前 7 个角度是左臂，后 7 个角度是右臂
        left_positions = list(target_positions[:NUM_JOINTS_PER_ARM])
        right_positions = list(target_positions[NUM_JOINTS_PER_ARM:])

        self.get_logger().info(f'收到新目标: 左臂 {[f"{v:.2f}" for v in left_positions]}, 右臂 {[f"{v:.2f}" for v in right_positions]}')
        
        # 3. 分别发布左右臂轨迹
        self._publish_arm_trajectory("left", self.left_publisher, self.left_joint_names, left_positions)
        self._publish_arm_trajectory("right", self.right_publisher, self.right_joint_names, right_positions)


    def _publish_arm_trajectory(self, arm_name, publisher, joint_names, positions):
        """创建并发布单个手臂的 JointTrajectory 消息"""
        
        trajectory_msg = JointTrajectory()
        trajectory_msg.header.stamp = self.get_clock().now().to_msg()
        trajectory_msg.joint_names = joint_names
        
        # 构造单个目标点
        point = JointTrajectoryPoint()
        point.positions = positions
        
        # 设定到达目标的时间
        point.time_from_start = Duration(seconds=TIME_TO_REACH_SEC).to_msg()
        
        # 由于我们只发布单个目标点，不需要速度/加速度，控制器会计算
        
        trajectory_msg.points.append(point)
        
        # 发布消息
        publisher.publish(trajectory_msg)
        self.get_logger().debug(f'已发布 {arm_name} 臂目标。')


def main(args=None):
    rclpy.init(args=args)
    commander = JointAngleSubscriber()
    
    try:
        rclpy.spin(commander)
    except KeyboardInterrupt:
        commander.get_logger().info("程序被用户中断。")
    except Exception as e:
        commander.get_logger().error(f"发生错误: {e}")
    finally:
        commander.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()