import time
import asyncio
import numpy as np
import mujoco
import casadi_ik
from scipy.spatial.transform import Rotation as R

# === Vuer 导入 ===
from vuer import Vuer, VuerSession
from vuer.schemas import Urdf, PointLight, Scene, MotionControllers
from vuer.events import ClientEvent

# === 路径设置 (请确认路径正确) ===
# 建议将 viola.urdf 放在当前文件夹的 assets/ 目录下，方便 Vuer 读取
URDF_PATH = "viola_description/urdf/viola.urdf" 
ARM_XML_PATH = '/home/jodell/start_ai_eyes_arm/viola_description/urdf/viola_description.xml'

class VRRobotController:
    def __init__(self, arm_xml_path):
        # 1. 初始化 IK 解算器
        self.arm = casadi_ik.Kinematics("link6")
        self.arm.buildFromMJCF(arm_xml_path)
        
        # 2. 内部状态
        self.qpos = np.zeros(self.arm.model.nq)
        self.target_pos = None
        self.target_rot = None
        
        # 3. 遥操作状态 (Clutch)
        self.clutch_engaged = False
        self.hand_pose_ref = None
        self.robot_pose_ref = None
        self.scale = 1.0 

        self.init_robot_pose()

    def init_robot_pose(self):
        tf = self.arm.fk(self.qpos[:6])
        self.target_pos = tf[:3, 3]
        self.target_rot = tf[:3, :3]
        print(f"Robot Initial Pose: {self.target_pos}")

    def update_vr_input(self, hand_matrix, trigger_value):
        """
        被事件回调调用的更新函数
        hand_matrix: 4x4 numpy array
        trigger_value: float (0.0 - 1.0)
        """
        # 提取当前手柄位姿
        hand_pos_curr = hand_matrix[:3, 3]
        hand_rot_curr = hand_matrix[:3, :3]
        
        is_triggered = trigger_value > 0.5

        # --- 离合 (Clutch) 逻辑 ---
        if is_triggered:
            if not self.clutch_engaged:
                # [刚按下] 记录参考点
                self.clutch_engaged = True
                self.hand_pose_ref = (hand_pos_curr.copy(), hand_rot_curr.copy())
                self.robot_pose_ref = (self.target_pos.copy(), self.target_rot.copy())
                print(">>> Clutch Engaged")
            else:
                # [按住中] 计算相对运动
                ref_hand_pos, ref_hand_rot = self.hand_pose_ref
                ref_robot_pos, ref_robot_rot = self.robot_pose_ref

                # 计算手柄的相对位移 (在世界系)
                # 注意：Quest3 坐标系通常 Y 是上。如果方向不对，可能需要在这里调整轴。
                delta_pos = (hand_pos_curr - ref_hand_pos) * self.scale
                
                # 计算手柄的相对旋转 dR = R_curr * R_ref^T
                delta_rot = hand_rot_curr @ ref_hand_rot.T
                
                # 更新机器人目标
                self.target_pos = ref_robot_pos + delta_pos
                self.target_rot = delta_rot @ ref_robot_rot
        else:
            if self.clutch_engaged:
                # [松开]
                self.clutch_engaged = False
                print("<<< Clutch Released")

    def step(self):
        """执行一步 IK 计算"""
        if self.target_pos is None:
            return self.qpos

        tf_target = np.eye(4)
        tf_target[:3, :3] = self.target_rot
        tf_target[:3, 3] = self.target_pos

        # IK 解算 (使用上一帧角度作为猜测值)
        dof, info = self.arm.ik(tf_target, current_arm_motor_q=self.qpos)
        self.qpos = dof
        return self.qpos

# === Vuer 主程序设置 ===
app = Vuer()
robot = VRRobotController(ARM_XML_PATH)

# --- 关键修改 1: 使用 add_handler 监听控制器移动 ---
@app.add_handler("CONTROLLER_MOVE")
async def on_controller_move(event: ClientEvent, sess: VuerSession):
    """
    当 Quest 3 手柄移动时触发此函数
    event.value 包含了手柄的数据
    """
    data = event.value
    
    # 检查是否有右手数据 (我们通常用右手操作)
    right_hand = data.get("right")
    
    if right_hand:
        # 1. 获取位姿矩阵 (通常是 16 个浮点数的列表，列主序)
        matrix_list = right_hand.get("matrix")
        if matrix_list:
            # 转换为 4x4 numpy 矩阵 (需注意转置，WebXR通常是列主序，numpy是行主序)
            mat = np.array(matrix_list).reshape(4, 4).T 
            
            # 2. 获取扳机状态
            # 不同的 Vuer 版本数据结构可能略有不同，通常在 'buttons' 或直接在属性里
            # 如果没有直接的 'trigger' 字段，可能需要根据 log 调试结构
            # 假设结构为: {'trigger': {'value': 1.0, 'pressed': True}, ...} 或者是平铺的
            trigger = 0.0
            if "trigger" in right_hand:
                # 尝试获取 trigger 值
                t_data = right_hand["trigger"]
                if isinstance(t_data, dict):
                    trigger = t_data.get("value", 0.0)
                elif isinstance(t_data, (float, int)):
                    trigger = float(t_data)
            
            # 更新机器人控制器
            robot.update_vr_input(mat, trigger)

@app.spawn(start=True)
async def main(sess: VuerSession):
    print("Vuer Server Started. Open URL in Quest 3.")
    
    # --- 关键修改 2: 显式启用 MotionControllers 流 ---
    # stream=True 非常重要，否则不会触发 CONTROLLER_MOVE 事件
    sess.upsert @ Scene(
        children=[
            PointLight(position=[0, 2, 2], intensity=10),
            Urdf(src=URDF_PATH, key="robot", position=[0, 0, 0]),
            # 添加控制器组件
            MotionControllers(key="controllers", stream=True, left=True, right=True),
        ]
    )

    while True:
        # 执行机器人计算
        qpos = robot.step()
        
        # 更新前端模型显示
        joint_names = [f"joint{i+1}" for i in range(6)]
        joint_dict = {name: angle for name, angle in zip(joint_names, qpos[:6])}
        
        sess.update @ Urdf(key="robot", jointValues=joint_dict)
        
        # 必须有的 sleep，否则会阻塞事件循环
        await asyncio.sleep(0.01)