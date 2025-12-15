import mujoco
import numpy as np
import casadi_ik
import time
import threading
import asyncio
import cv2
import queue
import base64
from scipy.spatial.transform import Rotation as R

# 引入 Vuer 相关库
from vuer import Vuer
from vuer.events import ClientEvent
from vuer.schemas import DefaultScene, CameraView, Plane, Box

# 配置文件路径
SCENE_XML_PATH = '/home/jodell/start_ai_eyes_arm/viola_description/urdf/scene.xml'
ARM_XML_PATH = '/home/jodell/start_ai_eyes_arm/viola_description/urdf/viola_description.xml'

class VuerHeadController:
    def __init__(self, host='0.0.0.0', port=8012):
        # === 机械臂参数 ===
        self.x_min, self.x_max = -0.4, 0.6
        self.y_min, self.y_max = -0.5, 0.5
        self.z_min, self.z_max = 0.05, 0.6

        # === 状态变量 ===
        self.running = True
        self.first_pose_received = False
        self.vr_init_pos = None

        # 机械臂初始状态
        self.robot_init_pos = np.array([0.3, 0.0, 0.2])
        self.target_x = self.robot_init_pos[0]
        self.target_y = self.robot_init_pos[1]
        self.target_z = self.robot_init_pos[2]
        self.target_R = np.eye(3)

        self.T_vr_to_robot = R.from_euler('x', -90, degrees=True).as_matrix() @ \
                             R.from_euler('z', -90, degrees=True).as_matrix()

        # 通信队列（放 data URI 或 bytes）
        self.image_queue = queue.Queue(maxsize=1)

        # Vuer 服务器
        self.app = Vuer(host=host, port=port)
        self.proxy = None

        self.thread = threading.Thread(target=self._run_vuer_server, daemon=True)
        self.thread.start()

        print(f"🔥 Vuer 服务器启动中... 请访问 https://<电脑IP>:{port}")

    def _run_vuer_server(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        @self.app.spawn(start=True)
        async def main(proxy):
            self.proxy = proxy
            print("✅ [DEBUG] Vuer 客户端已连接")

            # === 1. 初始化场景 ===
            try:
                proxy.set @ DefaultScene(
                    children=[
                        Plane(
                            key="monitor",
                            args=[1.6, 0.9],
                            position=[0, 1.5, -1.5],
                            rotation=[0, 0, 0],
                            material=dict(color="white", side=2),
                            src="",  # 初始为空，后续用 data URI 更新
                        ),
                        Box(
                            key="ref_box",
                            args=[0.2, 0.2, 0.2],
                            position=[-0.5, 1.5, -1.0],
                            material=dict(color="red"),
                        ),
                        CameraView(
                            key="ego",
                            stream="frame",
                            position=[0, 0, 0],
                            rotation=[0, 0, 0],
                        )
                    ]
                )
                print("✅ [DEBUG] 场景已创建：前方屏幕 + 左侧红色方块")
            except Exception as e:
                print(f"⚠️ [WARN] 初始化场景出错: {e}")

            # === 2. 发图循环 ===
            async def image_sender_loop():
                while self.running:
                    try:
                        if self.proxy is None:
                            await asyncio.sleep(0.02)
                            continue

                        # 取出 data_uri / bytes
                        try:
                            data = None
                            if not self.image_queue.empty():
                                data = self.image_queue.get_nowait()
                        except Exception:
                            data = None

                        if data is not None:
                            # 优先尝试按 frame 流发送（低延迟）
                            try:
                                # 如果 proxy 支持 send_frame（部分 vuer 版本提供）
                                if hasattr(self.proxy, "send_frame"):
                                    # 如果 data 是 data URI，需把 base64 decode 回 bytes
                                    if isinstance(data, str) and data.startswith("data:image"):
                                        header, b64 = data.split(",", 1)
                                        frame_bytes = base64.b64decode(b64)
                                    else:
                                        frame_bytes = data
                                    # 发送帧；键名为 "frame" 与 CameraView(stream="frame") 对应
                                    try:
                                        self.proxy.send_frame("frame", frame_bytes)
                                    except Exception:
                                        # fallback to update Plane if send_frame fails
                                        self.proxy.update @ Plane(key="monitor", src=data)
                                else:
                                    # 回退到更新 Plane 的 src（data URI）
                                    self.proxy.update @ Plane(key="monitor", src=data)
                            except Exception as e:
                                # 单次错误不要中断循环
                                print(f"⚠️ [WARN] 发送图像失败: {e}")
                        await asyncio.sleep(0.016)
                    except Exception:
                        # 捕获外层所有异常防止任务退出
                        await asyncio.sleep(0.05)

            # 启动发送任务
            asyncio.create_task(image_sender_loop())

            # === 3. 头部追踪循环 ===
            while self.running:
                try:
                    event = await self.app.grab_event()
                    if event is None:
                        await asyncio.sleep(0.01)
                        continue
                    if hasattr(event, "etype") and event.etype == "CAMERA_MOVE":
                        # 收到头部矩阵
                        raw = event.value.get('matrix') if isinstance(event.value, dict) else None
                        if raw:
                            # event.value['matrix'] 是 16 元素的扁平列表（按你原代码）
                            try:
                                raw_matrix = np.array(raw).reshape(4, 4).T
                                vr_pos = raw_matrix[:3, 3]
                                self._update_pose(vr_pos)
                                # 仅做轻量打印
                                if int(time.time() * 10) % 50 == 0:
                                    print(f"📡 [DEBUG] CAMERA_MOVE 接收，pos={vr_pos}")
                            except Exception as e:
                                print(f"⚠️ [WARN] 解析 CAMERA_MOVE 矩阵失败: {e}")
                    else:
                        # 不是 CAMERA_MOVE 的事件可以忽略或打印
                        pass
                except Exception:
                    await asyncio.sleep(0.01)

    def _update_pose(self, vr_pos):
        if not self.first_pose_received:
            self.vr_init_pos = vr_pos
            self.first_pose_received = True
            print("✅ [系统] VR 原点已校准，开始控制机械臂！")
            return

        delta_pos_vr = vr_pos - self.vr_init_pos
        delta_pos_robot = self.T_vr_to_robot @ delta_pos_vr
        SCALE = 2.0
        delta_pos_robot *= SCALE

        self.target_x = np.clip(self.robot_init_pos[0] + delta_pos_robot[0], self.x_min, self.x_max)
        self.target_y = np.clip(self.robot_init_pos[1] + delta_pos_robot[1], self.y_min, self.y_max)
        self.target_z = np.clip(self.robot_init_pos[2] + delta_pos_robot[2], self.z_min, self.z_max)

    def broadcast_image(self, image_rgb):
        """
        接收 RGB numpy image (HxWx3), 将其编码为 JPEG bytes 并放入队列。
        重要：不要把 data-uri 放队列了，沉浸式模式需要原始 bytes。
        """
        try:
            # 转为 BGR 后编码 jpeg
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            ok, buffer = cv2.imencode('.jpg', image_bgr, encode_param)
            if not ok:
                print("⚠️ [WARN] JPEG 编码失败")
                return

            frame_bytes = buffer.tobytes()

            # 放入队列（覆盖旧帧）
            if self.image_queue.full():
                try:
                    self.image_queue.get_nowait()
                except Exception:
                    pass
            self.image_queue.put(frame_bytes)
        except Exception as e:
            print(f"⚠️ [WARN] broadcast_image 出错: {e}")


    def get_pose_target(self):
        # 返回 (x, y, z, roll, pitch, yaw)
        return self.target_x, self.target_y, self.target_z, 3.14, 0, 0

    def cleanup(self):
        self.running = False


# === 修改点：不再继承 mujoco_viewer，直接跑后台循环 ===
class HeadlessRobotController:
    def __init__(self, scene_path, arm_path, controller):
        self.controller = controller

        # 加载模型
        print(f"📂 [DEBUG] 正在加载模型: {scene_path}")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # IK 解算器
        self.arm = casadi_ik.Kinematics("link6")
        self.arm.buildFromMJCF(arm_path)
        self.last_dof = np.zeros(self.arm.model.nq)
        self.frame_count = 0

        # 初始化位置（读取当前 qpos）
        current_q = np.array(self.data.qpos[:6])
        try:
            tf_initial = self.arm.fk(current_q)
            self.controller.robot_init_pos = tf_initial[:3, 3]
            self.controller.target_R = tf_initial[:3, :3]
        except Exception as e:
            print(f"⚠️ [WARN] 初始 FK 失败: {e}")

        # 离屏渲染器
        self.offscreen_width = 640
        self.offscreen_height = 360
        self.renderer = mujoco.Renderer(self.model, height=self.offscreen_height, width=self.offscreen_width)

        # 相机（确保 default）
        self.offscreen_cam = mujoco.MjvCamera()
        try:
            # 设置默认 camera 参数，避免未初始化导致无像素
            mujoco.mjv_defaultCamera(self.offscreen_cam)
        except Exception:
            # 有些 mujoco Python 绑定可能没有 mjv_defaultCamera，捕获并继续
            pass

        self.offscreen_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.offscreen_cam.fixedcamid = -1
        self.offscreen_cam.lookat = [0, 0, 0.2]
        self.offscreen_cam.distance = 1.5
        self.offscreen_cam.azimuth = 135
        self.offscreen_cam.elevation = -30

        print("✅ [DEBUG] 仿真环境初始化完成 (Headless Mode)")

    def run_loop(self):
        print("🚀 [DEBUG] 开始物理仿真循环...")
        while self.controller.running:
            self.frame_count += 1

            try:
                # 1. IK 控制
                x, y, z, roll, pitch, yaw = self.controller.get_pose_target()
                tf_target = np.eye(4)
                tf_target[:3, :3] = self.controller.target_R
                tf_target[:3, 3] = [x, y, z]

                dof, info = self.arm.ik(tf_target, current_arm_motor_q=self.last_dof)
                self.last_dof = dof
                # 防止 qpos 长度不足异常
                try:
                    self.data.qpos[:6] = dof[:6]
                except Exception:
                    pass

                # 2. 物理步进
                mujoco.mj_step(self.model, self.data)

                # 3. 图像回传（每 3 帧发一次）
                if self.frame_count % 3 == 0:
                    try:
                        self.renderer.update_scene(self.data, camera=self.offscreen_cam)
                        pixels = self.renderer.render()  # 返回 ndarray HxWx3 RGB
                        # Debug 打印
                        if self.frame_count % 60 == 0:
                            print(f"🔄 [DEBUG] 帧 {self.frame_count} 已渲染")
                        # 直接广播
                        self.controller.broadcast_image(pixels)
                    except Exception as e:
                        print(f"Render Error: {e}")

                # 控制循环频率
                time.sleep(0.01)
            except Exception as e:
                print(f"🚨 [ERROR] 主循环捕获异常: {e}")
                time.sleep(0.05)


if __name__ == "__main__":
    import os
    # 尝试清理端口（Linux）
    try:
        os.system("fuser -k 8012/tcp")
    except Exception:
        pass

    controller = VuerHeadController(port=8012)
    try:
        robot = HeadlessRobotController(SCENE_XML_PATH, ARM_XML_PATH, controller)
        robot.run_loop()
    except KeyboardInterrupt:
        pass
    finally:
        controller.cleanup()
