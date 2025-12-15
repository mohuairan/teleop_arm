import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from robo_interfaces.msg import SetAngle # 确保编译并source了该消息包

import numpy as np
import time
import sys
import os
import pygame
from keyboard_interface import KeyboardController

# 引入你的 IK 和控制器库
# 假设这些文件在当前目录或 python path 中
import casadi_ik
# 如果之前的 KeyboardController 代码在单独文件，请 import；这里为了完整性我合并在下面

# === 配置 ===
SCENE_XML_PATH = '/home/slwang/teleoperation_robot/viola_description/urdf/scene.xml'
ARM_XML_PATH = '/home/slwang/teleoperation_robot/viola_description/urdf/viola_description.xml'
EE_NAME = "link6" # 请确保与 URDF 一致

class IKControlNode(Node):
    def __init__(self):
        super().__init__('ik_control_node')
        
        # 1. 初始化发布者/订阅者
        # 发送给 robo_driver 的指令
        self.pub_cmd = self.create_publisher(SetAngle, 'set_angle_topic', 10)
        
        # 接收 robo_driver 发来的真实状态
        self.sub_state = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        
        # 2. 初始化算法模型
        self.controller = KeyboardController()
        self.arm = casadi_ik.Kinematics(EE_NAME)
        self.arm.buildFromMJCF(ARM_XML_PATH) # 或者 buildFromURDF
        
        # 3. 状态变量
        self.current_joint_rad = np.zeros(6) # 存储从 ROS 读到的真实角度 (弧度)
        self.last_ik_dof = np.zeros(self.arm.model.nq) # 存储上一次 IK 的解 (用于 Warm Start)
        self.robot_ready = False # 标志位：是否收到过一次真实状态
        
        # 4. 启动控制循环 (50Hz)
        self.timer = self.create_timer(0.02, self.control_loop)
        
        self.get_logger().info("IK Control Node Started. Waiting for joint_states...")

    def joint_state_callback(self, msg):
        """
        接收来自 robo_driver 的真实机械臂角度
        注意：robo_driver 发出的 joint_states 已经是弧度了吗？
        查看 robo_driver.py: 
        self.Servo.servoangle2jointstate -> 似乎已经转成弧度了 (degrees_to_radians)
        """
        if len(msg.position) >= 6:
            # 假设顺序是 joint1...joint6
            self.current_joint_rad = np.array(msg.position[:6])
            
            # 仅在第一次收到消息时，同步 IK 的初始位置
            if not self.robot_ready:
                self.sync_initial_pose()
                self.robot_ready = True

    def sync_initial_pose(self):
        """将控制器的目标吸附到机械臂当前的真实位置"""
        self.get_logger().info(f"Syncing to initial pose: {np.degrees(self.current_joint_rad)}")
        
        # 使用真实角度计算 FK
        tf_initial = self.arm.fk(self.current_joint_rad)
        
        # 更新控制器目标
        self.controller.x = tf_initial[0, 3]
        self.controller.y = tf_initial[1, 3]
        self.controller.z = tf_initial[2, 3]
        self.controller.R = tf_initial[:3, :3]
        
        # 更新 IK 初值
        self.last_ik_dof[:6] = self.current_joint_rad
        
    def build_transform(self, x, y, z, roll, pitch, yaw):
        R = self.controller._rpy_to_matrix(roll, pitch, yaw)
        tf = np.eye(4)
        tf[:3, :3] = R
        tf[:3, 3] = [x, y, z]
        return tf

    def control_loop(self):
        if not self.robot_ready:
            return
        
        if not self.controller.is_connected():
            rclpy.shutdown()
            return

        # 1. 获取输入
        # 注意：传入 self.last_ik_dof 作为运动学参考，而不是真实状态，防止震荡反馈
        self.controller.handle_input(self.arm, self.last_ik_dof[:6])
        x, y, z, roll, pitch, yaw = self.controller.get_pose_target()
        
        # 2. 构建目标
        tf_target = self.build_transform(x, y, z, roll, pitch, yaw)
        
        # 3. IK 解算
        # 使用上一次解作为 Warm Start
        try:
            dof, info = self.arm.ik(tf_target, current_arm_motor_q=self.last_ik_dof)
            
            if info['success']:
                self.last_ik_dof = dof # 更新内部状态
                
                # 4. 发布指令给 ROS
                self.publish_command(dof[:6])
            else:
                self.get_logger().warn("IK Failed")
                
        except Exception as e:
            self.get_logger().error(f"IK Error: {e}")

    def publish_command(self, target_rad):
        """将计算出的弧度打包发送给 robo_driver"""
        msg = SetAngle()
        
        # 转换单位：robo_driver 期望的是度 (float)，它内部会再乘10
        target_deg = np.degrees(target_rad)
        
        # 填充消息
        # 假设我们要控制 ID 0~5
        msg.servo_id = [0, 1, 2, 3, 4, 5]
        msg.target_angle = target_deg.tolist()
        
        # 时间控制：50Hz = 20ms。
        # 为了平滑，我们可以给稍微大一点的时间窗口，例如 30ms 或 40ms
        msg.time = [30] * 6 
        msg.speed = [0] * 6 # robo_driver 的时间模式下可能不看这个，或者视作加速度
        
        self.pub_cmd.publish(msg)

    def destroy_node(self):
        self.controller.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = IKControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()