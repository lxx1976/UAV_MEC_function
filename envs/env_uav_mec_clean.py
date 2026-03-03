# -*- coding: utf-8 -*-
"""
干净的 UAV 边缘计算环境
使用 function.py 中的模块化函数实现
"""

import numpy as np
from typing import List, Tuple, Dict, Any

# 导入我们的功能函数库
from . import function as fn
from .uav_comm_energy import RotorcraftParams


class UAVMECEnvironment:
    """
    UAV 移动边缘计算环境（干净版本）

    特点：
    - 使用 function.py 中的模块化函数
    - 代码结构清晰，易于理解和维护
    - 支持多 UAV 协作
    """

    def __init__(self, config: Dict[str, Any] = None):
        """初始化环境"""
        # 默认配置
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        # 提取关键参数
        self.num_uavs = self.config['num_uavs']
        self.num_terminals = self.config['num_terminals']
        self.ground_area = self.config['ground_area']
        self.height_min = self.config['height_min']
        self.height_max = self.config['height_max']
        self.time_slot = self.config['time_slot']
        self.max_episode_steps = self.config['max_episode_steps']
        self.communication_range = self.config['communication_range']

        # 初始化空间配置
        self.space_config = fn.initialize_3d_space(
            self.ground_area,
            self.height_min,
            self.height_max
        )

        # 初始化旋翼参数（用于能耗计算）
        self.rotor_params = RotorcraftParams()

        # 状态变量
        self.current_step = 0
        self.uav_positions = None
        self.uav_battery = None
        self.terminals = None
        self.connection_matrix = None
        self.offload_decisions = None

        # 观测和动作空间维度
        self.observation_space = self._calculate_obs_dim()
        self.action_space = self._calculate_action_dim()

        print(f"[OK] UAV MEC 环境初始化完成")
        print(f"   - UAV 数量: {self.num_uavs}")
        print(f"   - 终端数量: {self.num_terminals}")
        print(f"   - 地面区域: {self.ground_area}m x {self.ground_area}m")
        print(f"   - 飞行高度: {self.height_min}m - {self.height_max}m")
        print(f"   - 观测维度: {self.observation_space}")
        print(f"   - 动作维度: {self.action_space}")

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 基本参数
            'num_uavs': 2,
            'num_terminals': 6,
            'ground_area': 400.0,
            'height_min': 20.0,
            'height_max': 120.0,
            'initial_height': 70.0,
            'time_slot': 1.0,
            'max_episode_steps': 1000,

            # 通信参数
            'communication_range': 100.0,
            'carrier_frequency': 2.4e9,
            'bandwidth': 1e6,
            'transmit_power': 0.1,
            'noise_power_density': 4e-21,
            'antenna_gain': 2.0,

            # 计算参数
            'cpu_freq_uav': 5e9,
            'cpu_freq_ground': 1e9,
            'cpu_cycles_per_bit': 1000,
            'data_range': (102400.0, 204800.0),  # bits (100-200 KB)

            # 运动参数
            'max_horizontal_speed': 10.0,
            'max_vertical_speed': 10.0,
            'delta_h': 5.0,

            # 能量参数
            'battery_capacity': 324000.0,  # J (~90Wh)
            'energy_per_bit': 1e-9,

            # 安全参数
            'min_safe_distance': 5.0,
        }

    def _calculate_obs_dim(self) -> int:
        """计算观测空间维度"""
        # 自身状态: 3 (电量, 高度, 处理量)
        # 每个终端: 4 (方向x2 + 距离 + 剩余数据)
        # 其他UAV: 3 * (num_uavs - 1) (方向x2 + 距离)
        obs_dim = 3 + 4 * self.num_terminals + 3 * (self.num_uavs - 1)
        return obs_dim

    def _calculate_action_dim(self) -> int:
        """计算动作空间维度"""
        # 动作空间: 8种移动 × 2种服务决策 = 16
        return 16

    def reset(self) -> List[np.ndarray]:
        """重置环境"""
        self.current_step = 0

        # 1. 初始化 UAV 位置（使用 function.py）
        self.uav_positions = fn.generate_uav_initial_positions(
            self.num_uavs,
            self.ground_area,
            self.config['initial_height'],
            mode='center'
        )

        # 2. 初始化 UAV 电量
        self.uav_battery = np.full(self.num_uavs, self.config['battery_capacity'])

        # 3. 初始化终端任务（使用 function.py）
        self.terminals = fn.initialize_all_terminal_tasks(
            self.num_terminals,
            self.config['data_range'],
            self.config['cpu_cycles_per_bit']
        )

        # 为每个终端添加位置信息
        terminal_positions = fn.generate_terminal_positions(
            self.num_terminals,
            self.ground_area
        )
        for i, terminal in enumerate(self.terminals):
            terminal['position'] = terminal_positions[i]
            terminal['id'] = i

        # 4. 初始化连接矩阵
        self.connection_matrix = np.zeros((self.num_uavs, self.num_terminals), dtype=bool)
        self._update_connections()

        # 5. 初始化卸载决策矩阵
        self.offload_decisions = np.zeros((self.num_uavs, self.num_terminals), dtype=bool)

        # 6. 获取初始观测
        observations = self._get_observations()

        return observations

    def _update_connections(self):
        """更新 UAV 与终端的连接状态（使用 function.py）"""
        terminal_positions = np.array([t['position'] for t in self.terminals])
        self.connection_matrix = fn.update_connection_matrix(
            self.uav_positions,
            terminal_positions,
            self.communication_range
        )

    def _get_observations(self) -> List[np.ndarray]:
        """获取所有 UAV 的观测（使用 function.py）"""
        observations = []

        # 准备环境参数
        env_params = {
            'battery_capacity': self.config['battery_capacity'],
            'height_min': self.height_min,
            'height_max': self.height_max,
            'data_range': self.config['data_range'],
            'ground_area': self.ground_area
        }

        # 计算每个 UAV 的处理数据量（简化版本）
        uav_processing_data = {i: 0.0 for i in range(self.num_uavs)}

        # 为每个 UAV 构建观测
        for uav_id in range(self.num_uavs):
            obs = fn.construct_full_observation(
                uav_id,
                self.uav_positions,
                self.uav_battery,
                uav_processing_data,
                self.terminals,
                env_params
            )
            observations.append(obs)

        return observations

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], List[Dict]]:
        """执行一步动作"""
        self.current_step += 1

        # 初始化返回值
        rewards = np.zeros(self.num_uavs)
        dones = [False] * self.num_uavs
        infos = [{} for _ in range(self.num_uavs)]

        # 第一步：解析动作并执行移动
        for uav_id in range(self.num_uavs):
            if self.uav_battery[uav_id] <= 0:
                continue  # 电量耗尽的 UAV 无法行动

            # 解析动作（使用 function.py）
            movement_action, service_decision = fn.decode_action(actions[uav_id])
            horizontal_action, vertical_action = fn.parse_movement_action(movement_action)

            # 执行水平移动
            new_pos, move_distance = fn.calculate_horizontal_movement(
                self.uav_positions[uav_id],
                horizontal_action,
                self.config['max_horizontal_speed'],
                self.time_slot
            )

            # 执行垂直移动
            new_height, v_vertical = fn.calculate_vertical_movement(
                new_pos[2],
                vertical_action,
                self.config['max_vertical_speed'],
                self.time_slot,
                self.height_min,
                self.height_max,
                self.config['delta_h']
            )
            new_pos[2] = new_height

            # 检查边界违规
            boundary_violation = fn.check_boundary_violation(
                new_pos,
                self.ground_area,
                self.height_min,
                self.height_max
            )

            if boundary_violation:
                # 裁剪到合法范围
                new_pos = fn.clip_position_to_boundary(
                    new_pos,
                    self.ground_area,
                    self.height_min,
                    self.height_max
                )
                rewards[uav_id] -= 1.0  # 边界惩罚

            # 更新位置
            self.uav_positions[uav_id] = new_pos

            # 保存服务决策
            infos[uav_id]['service_decision'] = service_decision
            infos[uav_id]['boundary_violation'] = boundary_violation

        # 检查碰撞
        collision_pairs = fn.check_collision(
            self.uav_positions,
            self.config['min_safe_distance']
        )
        for uav_i, uav_j in collision_pairs:
            rewards[uav_i] -= 1.0
            rewards[uav_j] -= 1.0
            infos[uav_i]['collision'] = True
            infos[uav_j]['collision'] = True

        # 第二步：更新连接状态
        self._update_connections()

        # 第三步：处理任务卸载和计算
        uav_terminal_progress = {}
        for uav_id in range(self.num_uavs):
            if self.uav_battery[uav_id] <= 0:
                continue

            service_decision = infos[uav_id].get('service_decision', 0)
            if service_decision == 0:
                continue  # 不服务

            uav_terminal_progress[uav_id] = {}

            # 遍历所有连接的终端
            for term_id in range(self.num_terminals):
                if not self.connection_matrix[uav_id, term_id]:
                    continue  # 未连接

                terminal = self.terminals[term_id]
                if terminal['is_completed']:
                    continue  # 已完成

                # 计算上行速率（使用 function.py）
                uplink_rate = fn.calculate_uplink_rate(
                    self.uav_positions[uav_id],
                    terminal['position'],
                    transmit_power=self.config['transmit_power'],
                    bandwidth=self.config['bandwidth'],
                    carrier_frequency=self.config['carrier_frequency'],
                    noise_power_density=self.config['noise_power_density'],
                    antenna_gain=self.config['antenna_gain'],
                    mode='expected'
                )

                # 计算处理量（使用 function.py）
                processed_bits = fn.calculate_processed_bits_coupled(
                    offload_decision=True,
                    cpu_freq_uav=self.config['cpu_freq_uav'],
                    data_bits=terminal['remaining_data_bits'],
                    time_slot=self.time_slot,
                    uplink_rate=uplink_rate,
                    cpu_cycles=terminal['cpu_cycles_per_bit'],
                    cpu_freq_ground=self.config['cpu_freq_ground']
                )

                uav_terminal_progress[uav_id][term_id] = processed_bits
                rewards[uav_id] += 0.1  # 服务奖励

        # 第四步：更新终端任务进度（使用 function.py）
        self.terminals, completed_ids = fn.update_all_terminals_progress(
            self.terminals,
            uav_terminal_progress
        )

        # 任务完成奖励
        for term_id in completed_ids:
            for uav_id in range(self.num_uavs):
                if uav_id in uav_terminal_progress and term_id in uav_terminal_progress[uav_id]:
                    rewards[uav_id] += 5.0  # 完成任务奖励

        # 第五步：计算能耗并更新电量
        for uav_id in range(self.num_uavs):
            # 计算速度
            if self.current_step > 1:
                v_h, v_v = fn.calculate_velocity_from_positions(
                    self.uav_positions[uav_id],
                    self.uav_positions[uav_id],  # 简化：使用当前位置
                    self.time_slot
                )
            else:
                v_h, v_v = 0.0, 0.0

            # 计算推进能耗（使用 function.py）
            propulsion_energy, _ = fn.calculate_propulsion_energy(
                v_h, v_v, self.time_slot, self.rotor_params
            )

            # 计算通信能耗
            comm_energy = fn.calculate_communication_energy(
                self.config['transmit_power'],
                self.time_slot,
                efficiency=0.1
            )

            # 更新电量
            total_energy = propulsion_energy + comm_energy
            self.uav_battery[uav_id] -= total_energy

            if self.uav_battery[uav_id] <= 0:
                self.uav_battery[uav_id] = 0
                rewards[uav_id] -= 10.0  # 能量耗尽惩罚

        # 第六步：检查终止条件
        all_completed, completion_ratio = fn.check_all_tasks_completed(self.terminals)

        if all_completed:
            # 所有任务完成
            for uav_id in range(self.num_uavs):
                rewards[uav_id] += 50.0  # 任务完成奖励
                dones[uav_id] = True
        elif self.current_step >= self.max_episode_steps:
            # 超时
            dones = [True] * self.num_uavs
        elif all(self.uav_battery <= 0):
            # 所有 UAV 能量耗尽
            dones = [True] * self.num_uavs

        # 第七步：获取新观测
        observations = self._get_observations()

        # 添加额外信息
        for uav_id in range(self.num_uavs):
            infos[uav_id]['completion_ratio'] = completion_ratio
            infos[uav_id]['battery'] = self.uav_battery[uav_id]

        return observations, rewards.tolist(), dones, infos
