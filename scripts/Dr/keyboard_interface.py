import mujoco
import numpy as np
import mujoco_viewer
import casadi_ik
import time
import pygame
import os


SCENE_XML_PATH = '/home/slwang/start_ai_eyes_arm/viola_description/urdf/scene.xml'
ARM_XML_PATH = '/home/slwang/start_ai_eyes_arm/viola_description/urdf/viola_description.xml'


class KeyboardController:
    """键盘控制器类（替代原XboxController）"""
    
    def __init__(self):
        # === 运动范围限制 ===
        self.x_min, self.x_max = -0.4, 0.4
        self.y_min, self.y_max = -0.4, 0.4
        self.z_min, self.z_max = 0.05, 0.35
        
        # === 灵敏度设置 (因为键盘是开关量，步长要设小一点) ===
        self.pos_step = 0.001  # 每次按键循环移动的距离 (米)
        self.ori_step = 0.005  # 每次按键循环旋转的角度 (弧度)
        
        self.running = True
        self.init_controller()
        
    def _rpy_to_matrix(self, roll, pitch, yaw):
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cy, -sy, 0],
                       [sy,  cy, 0],
                       [0,    0, 1]])
        Ry = np.array([[cp, 0, sp],
                       [0, 1, 0],
                       [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0],
                       [0, cr, -sr],
                       [0, sr,  cr]])
        return Rz @ Ry @ Rx

    def _matrix_to_rpy(self, R):
        pitch = -np.arcsin(R[2, 0])
        if abs(R[2, 0]) < 0.999999:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = 0
            yaw = np.arctan2(-R[0, 1], R[1, 1])
        return roll, pitch, yaw

    def _axis_angle_to_matrix(self, axis, angle):
        if angle == 0:
            return np.eye(3)
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K @ K)
        return R

    def init_controller(self):
        pygame.init()
        # 创建一个小窗口以接收键盘事件 (Pygame 必须有窗口才能监听键盘)
        self.screen = pygame.display.set_mode((300, 200))
        pygame.display.set_caption("Keyboard Teleop Control")
        
        # 在窗口上显示简单的提示
        font = pygame.font.SysFont(None, 24)
        text1 = font.render("Click here to control", True, (255, 255, 255))
        text2 = font.render("WASD: Move | QE: Up/Down", True, (255, 255, 255))
        self.screen.blit(text1, (20, 50))
        self.screen.blit(text2, (20, 80))
        pygame.display.flip()
        
        print("键盘控制器已启动。请确保聚焦在弹出的 Pygame 窗口上。")
        print("控制键位:")
        print("  位置: [W/S]前后  [A/D]左右  [Q/E]上下")
        print("  姿态: [U/O]Roll  [I/K]Pitch [J/L]Yaw")
        
    def is_connected(self):
        return self.running
        
    def handle_input(self, arm, current_qpos):
        # 处理退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return

        # 获取所有按键状态
        keys = pygame.key.get_pressed()
        
        # === 1. 计算局部坐标系下的位移 (dx, dy, dz) ===
        dx, dy, dz = 0.0, 0.0, 0.0
        
        # W/S -> X轴 (前后)
        if keys[pygame.K_w]: dx = 1.0
        if keys[pygame.K_s]: dx = -1.0
        
        # A/D -> Y轴 (左右)
        if keys[pygame.K_a]: dy = 1.0
        if keys[pygame.K_d]: dy = -1.0
        
        # Q/E -> Z轴 (上下)
        if keys[pygame.K_q]: dz = 1.0
        if keys[pygame.K_e]: dz = -1.0

        # 应用步长
        delta_local = np.array([dx, dy, dz]) * self.pos_step
        
        # 将局部位移转换为全局位移 (delta_world = R_current * delta_local)
        # 这样按 'W' 永远是沿着末端的前进方向走，而不是世界坐标系的X轴
        tf_current = arm.fk(current_qpos)
        R_ee = tf_current[:3, :3]
        delta_world = R_ee @ delta_local
        
        # 更新目标位置 (带限位)
        self.x = np.clip(self.x + delta_world[0], self.x_min, self.x_max)
        self.y = np.clip(self.y + delta_world[1], self.y_min, self.y_max)
        self.z = np.clip(self.z + delta_world[2], self.z_min, self.z_max)

        # === 2. 计算旋转增量 (roll, pitch, yaw) ===
        d_roll, d_pitch, d_yaw = 0.0, 0.0, 0.0
        
        # U/O -> Roll
        if keys[pygame.K_u]: d_roll = 1.0
        if keys[pygame.K_o]: d_roll = -1.0
        
        # I/K -> Pitch
        if keys[pygame.K_i]: d_pitch = 1.0
        if keys[pygame.K_k]: d_pitch = -1.0
        
        # J/L -> Yaw
        if keys[pygame.K_j]: d_yaw = 1.0
        if keys[pygame.K_l]: d_yaw = -1.0

        # 应用旋转矩阵更新
        if d_roll != 0 or d_pitch != 0 or d_yaw != 0:
            d_roll *= self.ori_step
            d_pitch *= self.ori_step
            d_yaw *= self.ori_step
            
            R_inc = (
                self._axis_angle_to_matrix(np.array([1,0,0]), d_roll) @
                self._axis_angle_to_matrix(np.array([0,1,0]), d_pitch) @
                self._axis_angle_to_matrix(np.array([0,0,1]), d_yaw)
            )
            self.R = self.R @ R_inc

    def get_pose_target(self):
        roll, pitch, yaw = self._matrix_to_rpy(self.R)
        return self.x, self.y, self.z, roll, pitch, yaw
        
    def cleanup(self):
        pygame.quit()


class RobotController(mujoco_viewer.CustomViewer):    
    def __init__(self, scene_path, arm_path, controller):
        super().__init__(scene_path, distance=1.5, azimuth=135, elevation=-30)
        self.controller = controller
        self.arm = casadi_ik.Kinematics("link6")
        self.arm.buildFromMJCF(arm_path)
        self.last_dof = np.zeros(self.arm.model.nq)
        self.frame_count = 0

        # 1. 获取当前关节角度 (通常是 XML 中定义的初始角度，或者全0)
        current_q = self.data.qpos[:6]
        
        # 2. 计算正运动学 (FK)，得到当前末端的实际位姿
        tf_initial = self.arm.fk(current_q)
        
        # 3. 覆盖控制器的初始目标，让它和机械臂当前状态同步
        start_pos = tf_initial[:3, 3]
        start_rot = tf_initial[:3, :3]
        
        print(f"初始化: 检测到机械臂当前位于: {start_pos}")
        print("已同步控制器目标。")

        self.controller.x = start_pos[0]
        self.controller.y = start_pos[1]
        self.controller.z = start_pos[2]
        self.controller.R = start_rot # 直接复制旋转矩阵
        
        # 初始化用于平滑的上一帧变量
        self.last_dof = current_q

    def runFunc(self):
        self.frame_count += 1
        
        # 检查控制器是否仍连接（对于键盘来说就是检查窗口是否关闭）
        if not self.controller.is_connected():
            print("控制器已断开")
            return

        # 处理输入并更新目标位姿
        self.controller.handle_input(self.arm, self.data.qpos[:6])
        x, y, z, roll, pitch, yaw = self.controller.get_pose_target()

        # 构建目标变换矩阵
        tf_target = self.build_transform(x, y, z, roll, pitch, yaw)
        
        # IK 解算
        dof, info = self.arm.ik(tf_target, current_arm_motor_q=self.last_dof)
        self.last_dof = dof

        # === 下面主要是打印调试信息，保持原样 ===
        qpos_actual = self.data.qpos[:6]
        qpos_theoretical = dof[:6]
        qpos_error = qpos_theoretical - qpos_actual

        tf_actual = self.arm.fk(qpos_actual)
        pos_actual = tf_actual[:3, 3]
        pos_target = tf_target[:3, 3]
        delta_pos = pos_target - pos_actual
        R_actual = tf_actual[:3, :3]
        R_target = tf_target[:3, :3]
        R_diff = R_target.T @ R_actual
        roll_err, pitch_err, yaw_err = self.controller._matrix_to_rpy(R_diff)

        print(f"\rTarget: [{x:.2f}, {y:.2f}, {z:.2f}] | "
              f"Err: Pos={np.linalg.norm(delta_pos):.3f} Ori={np.linalg.norm([roll_err, pitch_err, yaw_err]):.3f}", end="")

        # 误差保护逻辑 (前60帧不检查)
        if self.frame_count > 60:
            POS_ERR_LIMIT = 0.05 #稍微放宽一点以便调试
            ORI_ERR_LIMIT = 0.2
            pos_err_norm = np.linalg.norm(delta_pos)
            ori_err_norm = np.linalg.norm([roll_err, pitch_err, yaw_err])
            
            if pos_err_norm > POS_ERR_LIMIT or ori_err_norm > ORI_ERR_LIMIT:
                print(f"\n⚠️ [警告] 误差超限 (Pos: {pos_err_norm:.3f}, Ori: {ori_err_norm:.3f})，重置目标！")
                # 将目标重置为当前实际位置，防止机械臂乱飞
                self.controller.x, self.controller.y, self.controller.z = pos_actual
                self.controller.R = R_actual.copy()

        # 执行步进
        self.data.qpos[:6] = dof[:6]
        mujoco.mj_step(self.model, self.data)
        time.sleep(0.01)

    def build_transform(self, x, y, z, roll, pitch, yaw):
        R = self.controller._rpy_to_matrix(roll, pitch, yaw)
        tf = np.eye(4)
        tf[:3, :3] = R
        tf[:3, 3] = [x, y, z]
        return tf


if __name__ == "__main__":
    # 使用 KeyboardController
    controller = KeyboardController()
    
    try:
        robot = RobotController(SCENE_XML_PATH, ARM_XML_PATH, controller)
        robot.run_loop()
    except KeyboardInterrupt:
        print("\n程序已停止")
    finally:
        controller.cleanup()