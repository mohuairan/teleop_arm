#!/usr/bin/env python3
"""
双臂关节角度指令接收器 (Subscriber/Publisher)

此脚本订阅外部发布的高频 14 个关节角度，将它们拆分为左右臂，
并发布到相应的 JointTrajectoryController Topic 接口，驱动双臂运动。
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState 
from rclpy.action import ActionClient
import argparse # 新增：用于命令行参数解析
import sys
import math

# =========================================================================
# 配置参数
# =========================================================================
PLANNING_TOPIC = "/planning/joint_angles" 
LEFT_CONTROLLER_ACTION = "/left_arm_controller/follow_joint_trajectory"
RIGHT_CONTROLLER_ACTION = "/right_arm_controller/follow_joint_trajectory"
LEFT_CONTROLLER_TOPIC = "/left_arm_controller/joint_trajectory" 
RIGHT_CONTROLLER_TOPIC = "/right_arm_controller/joint_trajectory" 
NUM_JOINTS_PER_ARM = 7
PI = math.pi

# 初始姿态 (Start Pose) 
# [-2, 0.85, -0.7, 1.42, 0.64, 0.64, 0, 2, 0.85, 0.7, 1.42, -0.64, 0.64, 0]
START_POSE = [
    -2.0, 0.85, 0.7, 1.42, 0.64, 0.64, 0.0, 
    2.0-PI, 0.85, 0.7-PI, 1.42, -0.64, 0.64, 0.0
]
START_POSE_DURATION_SEC = 8.0 # 初始姿态的运动时间，确保缓慢安全

# ... (其他定义) ...
START_POSE_DURATION_SEC = 8.0 # 初始姿态的运动时间，确保缓慢安全 (每个阶段 3秒)

# 中间姿态 P_mid: 仅移动前 2 个关节到 START_POSE 的目标值，后 5 个关节保持 0
# [L1, L2, L3, L4, L5, L6, L7, R1, R2, R3, R4, R5, R6, R7]
MID_POSE = [
    -2.0, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 
    2.0-PI, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0
]
# ... (其他定义) ...

# 基础执行时间 (Base Time to Reach): 用于高频透传的最小时间间隔 (秒)
# 对应于 ~50Hz 的控制周期，适合 80Hz 的输入。
BASE_TIME_TO_REACH_SEC = 0.02 

class JointAngleSubscriber(Node):
    """接收 14 关节角度并分发给左右臂控制器的节点"""

    def __init__(self, speed_scaling=1.0): # 接收速度缩放因子
        super().__init__("joint_angle_subscriber")
        self.get_logger().info("初始化 14 关节角度指令接收器...")
        
        # 速度缩放因子 (1.0 为默认速度)
        self.speed_scaling = max(0.01, min(1.0, speed_scaling)) 
        
        # 实际目标执行时间
        # 速度越慢 (speed_scaling 越小), 执行时间越长
        self.actual_time_to_reach = BASE_TIME_TO_REACH_SEC / self.speed_scaling
        
        self.get_logger().info(f"速度缩放因子: {self.speed_scaling * 100:.0f}%")
        self.get_logger().info(f"每个目标执行时间: {self.actual_time_to_reach:.4f} 秒")
        
        # 关节名称 (保持不变)
        self.left_joint_names = [
            "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6", "left_j7",
        ]
        self.right_joint_names = [
            "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_j7",
        ]
        
        # 1. Action 客户端 (用于初始移动)
        self.left_action_client = ActionClient(self, FollowJointTrajectory, LEFT_CONTROLLER_ACTION)
        self.right_action_client = ActionClient(self, FollowJointTrajectory, RIGHT_CONTROLLER_ACTION)
       
        # 2. 创建左右臂控制器的话题发布器
        self.left_publisher = None
        self.right_publisher = None

        # 3. 创建接收 14 目标角度的订阅器
        self.subscription = None

        # 4. 执行初始姿态移动
        self._initialize_arm_pose()
        
    def _initialize_arm_pose(self):
        """发送初始 Action 目标，分两段执行，完成后启动订阅器"""
        self.get_logger().info("等待左右臂控制器 Action Server 启动...")
        
        # 阻塞等待 Action Server 启动 (保持不变)
        if not self.left_action_client.wait_for_server(timeout_sec=10.0) or \
           not self.right_action_client.wait_for_server(timeout_sec=10.0):
             self.get_logger().error("Action Server 未启动！无法执行初始移动。")
             return
             
        self.get_logger().info("Action Server 均已就绪。")

        # --- 阶段 1: 移动到中间姿态 (仅 J1, J2 运动) ---
        self.get_logger().info("--- 阶段 1: 移动 J1, J2 到中间姿态 (保持 J3-J7 安全) ---")

        mid_pose_left = self._get_action_goal(
            self.left_joint_names, 
            MID_POSE[:NUM_JOINTS_PER_ARM], 
            START_POSE_DURATION_SEC
        )
         self.get_logger().info("左臂中间状态完成")

        mid_pose_right = self._get_action_goal(
            self.right_joint_names, 
            MID_POSE[NUM_JOINTS_PER_ARM:], 
            START_POSE_DURATION_SEC
        )
         self.get_logger().info("右臂中间状态完成")
        # 发送目标并等待结果
        if not self._wait_for_action_completion(mid_pose_left, mid_pose_right):
            self.get_logger().error("阶段 1 移动失败。")
            return
            
        # --- 阶段 2: 移动到最终起始姿态 (仅 J3-J7 运动) ---
        self.get_logger().info("--- 阶段 2: 移动 J3-J7 到最终起始姿态 ---")
        
        final_pose_left = self._get_action_goal(
            self.left_joint_names, 
            START_POSE[:NUM_JOINTS_PER_ARM], 
            START_POSE_DURATION_SEC
        )
        final_pose_right = self._get_action_goal(
            self.right_joint_names, 
            START_POSE[NUM_JOINTS_PER_ARM:], 
            START_POSE_DURATION_SEC
        )

        # 发送目标并等待结果
        if not self._wait_for_action_completion(final_pose_left, final_pose_right):
            self.get_logger().error("阶段 2 移动失败。")
            return

        self.get_logger().info("✅ 两阶段初始姿态移动完成。正在启动高频订阅...")
        self._start_high_frequency_mode()


    def _wait_for_action_completion(self, goal_left, goal_right):
        """辅助函数：发送左右臂 Action 目标并阻塞等待完成"""
        
        future_left = self.left_action_client.send_goal(goal_left)
        future_right = self.right_action_client.send_goal(goal_right)
        
        rclpy.spin_until_future_complete(self, future_left)
        rclpy.spin_until_future_complete(self, future_right)
        
        goal_handle_left = future_left.result()
        goal_handle_right = future_right.result()

        if not goal_handle_left or not goal_handle_right or \
           not goal_handle_left.accepted or not goal_handle_right.accepted:
            self.get_logger().error("目标被控制器拒绝。")
            return False

        self.get_logger().info("目标已被接受，等待执行...") 
        # 阻塞等待结果
        rclpy.spin_until_future_complete(self, goal_handle_left.get_result_async())
        rclpy.spin_until_future_complete(self, goal_handle_right.get_result_async())
        
        result_left = goal_handle_left.get_result_async().result()
        result_right = goal_handle_right.get_result_async().result()

        if result_left and result_right:
            if result_left.result.error_code == 0 and result_right.result.error_code == 0:
                self.get_logger().info("阶段 Action 成功。")
                return True
        
        self.get_logger().error("阶段 Action 失败或超时。")
        return False

    def _get_action_goal(self, joint_names, positions, duration):
        """辅助函数：创建 FollowJointTrajectory Action 目标"""
        goal_msg = FollowJointTrajectory.Goal()
        trajectory = JointTrajectory()
        
        trajectory.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(seconds=duration).to_msg()
        trajectory.points.append(point)
        
        goal_msg.trajectory = trajectory
        return goal_msg

    def _start_high_frequency_mode(self):
        """初始化 Topic 发布器和订阅器，开始处理流数据"""
        
        # 1. 创建左右臂控制器的话题发布器 (用于高频流)
        self.left_publisher = self.create_publisher(
            JointTrajectory, LEFT_CONTROLLER_TOPIC, 10
        )
        self.right_publisher = self.create_publisher(
            JointTrajectory, RIGHT_CONTROLLER_TOPIC, 10
        )
        self.get_logger().info(f"左臂流发布器: {LEFT_CONTROLLER_TOPIC}")
        self.get_logger().info(f"右臂流发布器: {RIGHT_CONTROLLER_TOPIC}")


        # 2. 创建接收 14 目标角度的订阅器 (用于高频流)
        self.subscription = self.create_subscription(
            JointState,
            PLANNING_TOPIC,
            self.angle_callback,
            10
        )
        self.get_logger().info(f"订阅规划话题: {PLANNING_TOPIC}")
        self.get_logger().info("🚀 已进入高频流式控制模式。")

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
        left_positions = list(target_positions[:NUM_JOINTS_PER_ARM])
        right_positions = list(target_positions[NUM_JOINTS_PER_ARM:])

        #3.处理订阅到的角度
        # 左臂 j3 (索引 2) 取反
        # 原始：left_j3
        # 处理后：-left_j3
        left_positions[2] = -left_positions[2]

        # 右臂 j1 ,j3(索引 0,2) 减去 PI
        # 原始：right_j1，j3
        # 处理后：right_j1 - PI,
        right_positions[0] = right_positions[0] - PI
        right_positions[2] = right_positions[2] - PI

        # 4. 分别发布左右臂轨迹
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
        # 关键修改: 使用动态计算的 actual_time_to_reach
        point.time_from_start = Duration(seconds=self.actual_time_to_reach).to_msg()
        
        trajectory_msg.points.append(point)
        
        # 发布消息
        publisher.publish(trajectory_msg)
        self.get_logger().debug(f'已发布 {arm_name} 臂目标。')


def main(args=None):
    rclpy.init(args=args)

    # --- 新增命令行参数解析 ---
    parser = argparse.ArgumentParser(description="双臂关节角度指令接收器")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0, # 默认速度 100%
        help="运动速度缩放因子 (0.01 - 1.0)。值越小，运动越慢 (执行时间越长)。",
    )
    parsed_args = parser.parse_args(args=sys.argv[1:])
    # --- 结束参数解析 ---

    commander = JointAngleSubscriber(speed_scaling=parsed_args.speed)
    
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