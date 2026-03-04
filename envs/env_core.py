import numpy as np

from .uav_comm_energy import RotorcraftParams
from .function import (
    calculate_communication_energy,      # 计算通信能耗（发射功率×时隙×效率）
    calculate_computation_energy,        # 计算计算能耗（处理数据量×每bit能耗）
    calculate_horizontal_movement,       # 计算水平移动后的位置（上下左右4个方向）
    calculate_processed_bits_coupled,    # 计算通信-计算耦合下的实际处理数据量（核心函数）
    calculate_propulsion_energy,         # 计算推进能耗（剖面+寄生+诱导+垂直爬升）
    calculate_total_energy_consumption,  # 计算总能耗（推进+计算+通信）
    calculate_uplink_rate,               # 计算上行速率（整合路径损耗、SNR、信道容量）
    calculate_velocity_from_positions,   # 从位置变化计算水平和垂直速度
    calculate_vertical_movement,         # 计算垂直移动后的高度（上升/下降/悬停）
    check_all_tasks_completed,           # 检查所有任务是否完成，返回完成状态和比例
    clip_position_to_boundary,           # 将位置裁剪到合法范围内（边界约束）
    construct_full_observation,          # 构建完整观测向量（自身+终端+其他UAV）
    decode_action_vector_distance_based, # 将连续动作向量解码为离散动作（移动+基于距离的服务）
    generate_terminal_positions,         # 生成地面终端位置（随机或基准+偏移）
    generate_uav_initial_positions,      # 生成UAV初始位置（中心/随机/网格）
    get_fixed_terminal_positions,        # 获取固定的终端基准位置
    initialize_all_terminal_tasks,       # 初始化所有终端任务（数据量、CPU周期等）
    update_all_terminals_progress,       # 更新所有终端任务进度（批量更新剩余数据量）
)


class EnvCore(object):
    """
    UAV-assisted edge computing environment core.

    Action (continuous 3D per UAV):
    - action[0]: mapped to movement action (7 bins)
    - action[1]: mapped to service decision (2 bins: 0=no service, 1=service)
    - action[2]: mapped to number of terminals to serve (4 bins: 0-3)
    """

    def __init__(self):
        # Multi-agent settings
        self.agent_num = 2
        self.num_terminals = 6
        self.action_dim = 3

        # Episode settings
        self.episode_limit = 1000
        self.time_slot = 1.0
        self.current_step = 0

        # Space and mobility settings
        self.ground_area = 400.0
        self.height_min = 20.0
        self.height_max = 120.0
        self.initial_height = 70.0
        self.max_horizontal_speed = 10.0
        self.max_vertical_speed = 5.0
        self.delta_h = 5.0

        # Communication and computation settings
        self.transmit_power = 0.2  # 0.2 W (increased from 0.1)
        self.bandwidth = 5e6  # 5 MHz (increased from 1 MHz)
        self.carrier_frequency = 2.4e9
        self.noise_power_density = 4e-21
        self.antenna_gain = 2.0
        
        # LoS probability model parameters (for air-ground channel)
        self.los_a = 9.61  # LoS probability fitting parameter a (urban scenario)
        self.los_b = 0.16  # LoS probability fitting parameter b
        
        # LoS/NLoS additional path loss (dB)
        self.eta_los = 1.0   # LoS additional loss
        self.eta_nlos = 20.0 # NLoS additional loss
        
        self.cpu_freq_terminal = 2e9  # 2 GHz - Terminal local CPU (weak)
        self.cpu_freq_uav = 5e9  # 5 GHz - UAV CPU (medium)
        self.cpu_freq_ground = 100e9  # 100 GHz - Ground server CPU (strong)
        self.cpu_cycles_per_bit = 1000
        self.data_range = (200000.0, 300000.0)  # KB = 200-300 MB (increased from 100-200 KB)

        # Energy settings
        self.rotor_params = RotorcraftParams()
        self.battery_capacity = 80000.0  # 80 kJ (increased from 20 kJ)
        self.energy_per_bit = 1e-6  # 1 μJ/bit (increased from 1 nJ/bit)

        # Reward settings
        self.reward_per_bit = 1e-7  # 每bit数据的奖励（降低10倍）
        self.energy_penalty = 1e-3  # 能耗惩罚系数
        self.completion_bonus_half = 5.0  # 完成50%的奖励
        self.completion_bonus_full = 10.0  # 完成100%的奖励
        self.service_reward = 0.1  # 提供卸载服务的小奖励
        self.invalid_service_penalty = 0.5  # 服务已完成终端的惩罚
        self.battery_depleted_penalty = 50.0  # 电池耗尽的惩罚
        self.timeout_penalty = 100.0  # 超时未完成所有任务的惩罚

        # Observation dim: self(3) + terminals(4*num_terminals) + other_uavs(3*(N-1))
        self.obs_dim = 3 + 4 * self.num_terminals + 3 * (self.agent_num - 1)

        # Runtime state
        self.uav_positions = None
        self.uav_battery = None
        self.uav_processing_data = None
        self.terminals = None

    def _build_terminal_states(self):
        # 获取固定的终端基准位置（从 function.py）
        base_positions = get_fixed_terminal_positions(
            num_terminals=self.num_terminals,
            ground_area=self.ground_area
        )
        
        # 生成终端位置（固定位置，无随机偏移）
        terminal_positions = generate_terminal_positions(
            num_terminals=self.num_terminals,
            ground_area=self.ground_area,
            base_positions=base_positions,
            variance=0.0,  # 0.0 = 完全固定，不随机偏移
        )
        
        terminal_tasks = initialize_all_terminal_tasks(
            num_terminals=self.num_terminals,
            data_range=self.data_range,
            cpu_cycles_per_bit=self.cpu_cycles_per_bit,
        )
        for term_id, task in enumerate(terminal_tasks):
            task["position"] = terminal_positions[term_id]
        return terminal_tasks

    def _get_obs(self):
        env_params = {
            "battery_capacity": self.battery_capacity,
            "height_min": self.height_min,
            "height_max": self.height_max,
            "data_range": self.data_range,
            "ground_area": self.ground_area,
        }
        obs = []
        for uav_id in range(self.agent_num):
            one_obs = construct_full_observation(
                uav_id=uav_id,
                uav_positions=self.uav_positions,
                uav_battery=self.uav_battery,
                uav_processing_data=self.uav_processing_data,
                terminals=self.terminals,
                env_params=env_params,
            ).astype(np.float32)
            obs.append(one_obs)
        return obs

    def reset(self):
        self.current_step = 0
        self.uav_positions = generate_uav_initial_positions(
            num_uavs=self.agent_num,
            ground_area=self.ground_area,
            initial_height=self.initial_height,
            mode="grid",
        )
        self.uav_battery = np.full(self.agent_num, self.battery_capacity, dtype=np.float64)
        self.uav_processing_data = {uav_id: 0.0 for uav_id in range(self.agent_num)}
        self.terminals = self._build_terminal_states()
        # 追踪每个终端是否已经获得过50%和100%完成奖励
        self.terminal_half_rewarded = [False] * self.num_terminals
        self.terminal_full_rewarded = [False] * self.num_terminals
        
        return self._get_obs()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(self.agent_num, -1)

        rewards = []
        dones = []
        infos = []

        uav_terminal_progress = {uav_id: {} for uav_id in range(self.agent_num)}
        selected_terminals = {}

        for uav_id in range(self.agent_num):
            action_vec = actions[uav_id]
            if action_vec.shape[0] < 3:
                pad = np.zeros(3 - action_vec.shape[0], dtype=np.float32)
                action_vec = np.concatenate([action_vec, pad], axis=0)

            # 解码动作：移动 + 服务决策 + 服务终端数量
            movement_action, horizontal_action, vertical_action, service_decision, num_terminals_to_serve = decode_action_vector_distance_based(
                action_vector=action_vec,
                num_terminals=self.num_terminals,
                movement_bins=7,
            )


            old_pos = self.uav_positions[uav_id].copy()
            # 计算移动后的位置
            moved_pos, _ = calculate_horizontal_movement(
                current_pos=old_pos.copy(),
                action=horizontal_action,
                max_speed=self.max_horizontal_speed,
                time_slot=self.time_slot,
            )
            new_height, _ = calculate_vertical_movement(
                current_height=old_pos[2],
                action=vertical_action,
                max_speed=self.max_vertical_speed,
                time_slot=self.time_slot,
                height_min=self.height_min,
                height_max=self.height_max,
                delta_h=self.delta_h,
            )
            moved_pos[2] = new_height
            moved_pos = clip_position_to_boundary(
                moved_pos, self.ground_area, self.height_min, self.height_max
            )
            self.uav_positions[uav_id] = moved_pos

            v_horizontal, v_vertical = calculate_velocity_from_positions(
                pos_old=old_pos,
                pos_new=moved_pos,
                time_slot=self.time_slot,
            )


            # 核心逻辑：基于距离选择服务的终端
            total_processed_bits = 0.0
            total_communication_energy = 0.0
            total_computation_energy = 0.0
            num_invalid_services = 0
            served_terminal_ids = []
            
            if service_decision and num_terminals_to_serve > 0:
                # 计算UAV到所有终端的距离
                distances = []
                for terminal_id in range(self.num_terminals):
                    terminal = self.terminals[terminal_id]
                    terminal_pos = terminal["position"]
                    distance = np.linalg.norm(moved_pos[:2] - terminal_pos[:2])
                    distances.append((terminal_id, distance))
                
                # 按距离排序（近到远）
                distances.sort(key=lambda x: x[1])
                
                # 选择最近的 num_terminals_to_serve 个终端
                for i in range(min(num_terminals_to_serve, len(distances))):
                    terminal_id, _ = distances[i]
                    terminal = self.terminals[terminal_id]
                    
                    # 检查终端是否已完成
                    if terminal.get("is_completed", False):
                        num_invalid_services += 1
                        continue
                    
                    # 计算上行速率
                    uplink_rate = calculate_uplink_rate(
                        uav_pos=moved_pos,
                        terminal_pos=terminal["position"],
                        transmit_power=self.transmit_power,
                        bandwidth=self.bandwidth,
                        carrier_frequency=self.carrier_frequency,
                        noise_power_density=self.noise_power_density,
                        antenna_gain=self.antenna_gain,
                        a=self.los_a,
                        b=self.los_b,
                        eta_los=self.eta_los,
                        eta_nlos=self.eta_nlos,
                        mode="expected",
                    )
                    
                    # 计算实际处理的数据量
                    processed_bits = calculate_processed_bits_coupled(
                        offload_decision=True,
                        cpu_freq_uav=self.cpu_freq_uav,
                        data_bits=terminal["remaining_data_bits"],
                        time_slot=self.time_slot,
                        uplink_rate=uplink_rate,
                        cpu_cycles=terminal["cpu_cycles_per_bit"],
                        cpu_freq_ground=self.cpu_freq_ground,
                    )
                    
                    # 记录该UAV对该终端的处理量
                    uav_terminal_progress[uav_id][terminal_id] = processed_bits
                    total_processed_bits += processed_bits
                    served_terminal_ids.append(terminal_id)
                    
                    # 累计计算能耗
                    computation_energy = calculate_computation_energy(
                        processed_bits=processed_bits,
                        energy_per_bit=self.energy_per_bit,
                    )
                    total_computation_energy += computation_energy
                    
                    # 累计通信能耗
                    communication_energy = calculate_communication_energy(
                        transmit_power=self.transmit_power,
                        time_slot=self.time_slot,
                        efficiency=1.0 if processed_bits > 0 else 0.0,
                    )
                    total_communication_energy += communication_energy
            
            # 记录选择服务的终端
            selected_terminals[uav_id] = served_terminal_ids

            # 累计该UAV的总处理数据量
            self.uav_processing_data[uav_id] += total_processed_bits


            # 计算能耗
            # 1. 推进能耗（飞行）
            propulsion_energy, _ = calculate_propulsion_energy(
                v_horizontal=v_horizontal,
                v_vertical=v_vertical,
                time_slot=self.time_slot,
                rotor_params=self.rotor_params,
            )
            
            # 2. 计算能耗（已在循环中累计）
            # 3. 通信能耗（已在循环中累计）
            
            # 总能耗
            total_energy = calculate_total_energy_consumption(
                propulsion_energy=propulsion_energy,
                computation_energy=total_computation_energy,
                communication_energy=total_communication_energy,
            )

            # 更新电池电量
            self.uav_battery[uav_id] = max(0.0, self.uav_battery[uav_id] - total_energy)

            # 计算奖励
            # 正奖励：处理数据量
            reward = total_processed_bits * self.reward_per_bit
                        
            # 正奖励：提供卸载服务（只要服务了就有小奖励）
            if len(served_terminal_ids) > 0:
                reward += self.service_reward * len(served_terminal_ids)
            
            # 负奖励：能耗
            reward -= total_energy * self.energy_penalty
            # 惩罚：服务已完成的终端
            if num_invalid_services > 0:
                reward -= self.invalid_service_penalty * num_invalid_services

            rewards.append([float(reward)])
            infos.append(
                {
                    "selected_terminals": served_terminal_ids,
                    "service_decision": bool(service_decision),
                    "num_terminals_to_serve": int(num_terminals_to_serve),
                    "num_served_terminals": len(served_terminal_ids),
                    "processed_bits": float(total_processed_bits),
                    "propulsion_energy_j": float(propulsion_energy),
                    "computation_energy_j": float(total_computation_energy),
                    "communication_energy_j": float(total_communication_energy),
                    "total_energy_j": float(total_energy),
                    "battery": float(self.uav_battery[uav_id]),
                    "num_invalid_services": int(num_invalid_services),
                }
            )

        self.terminals, completed_terminal_ids = update_all_terminals_progress(
            terminals=self.terminals,
            uav_terminal_progress=uav_terminal_progress,
            cpu_freq_terminal=self.cpu_freq_terminal,
            cpu_cycles_per_bit=self.cpu_cycles_per_bit,
            time_slot=self.time_slot,
        )
        # 计算完成奖励：两段式奖励（50%和100%）
        for uav_id in range(self.agent_num):
            served_terminal_ids = selected_terminals[uav_id]
            for term_id in served_terminal_ids:
                served_bits = uav_terminal_progress[uav_id].get(term_id, 0.0)
                if served_bits > 0.0:
                    terminal = self.terminals[term_id]
                    total_bits = terminal['total_data_bits']
                    remaining_bits = terminal['remaining_data_bits']
                    completion_ratio = 1.0 - (remaining_bits / total_bits)
                    
                    # 50%完成奖励
                    if completion_ratio >= 0.5 and not self.terminal_half_rewarded[term_id]:
                        rewards[uav_id][0] += self.completion_bonus_half
                        self.terminal_half_rewarded[term_id] = True
                    
                    # 100%完成奖励
                    if term_id in completed_terminal_ids and not self.terminal_full_rewarded[term_id]:
                        rewards[uav_id][0] += self.completion_bonus_full
                        self.terminal_full_rewarded[term_id] = True

        self.current_step += 1
        all_completed, completion_ratio = check_all_tasks_completed(self.terminals)
        out_of_battery = bool(np.any(self.uav_battery <= 0.0))
        timeout = self.current_step >= self.episode_limit
        episode_end = (
            timeout
            or all_completed
            or out_of_battery
        )
        # 添加终局惩罚      
        for uav_id in range(self.agent_num):
            # 电池耗尽惩罚
            if self.uav_battery[uav_id] <= 0.0:
                rewards[uav_id][0] -= self.battery_depleted_penalty
            
            # 超时未完成所有任务的惩罚
            if timeout and not all_completed:
                rewards[uav_id][0] -= self.timeout_penalty
            
            dones.append(bool(episode_end))
            infos[uav_id]["all_tasks_completed"] = bool(all_completed)
            infos[uav_id]["task_completion_ratio"] = float(completion_ratio)
            infos[uav_id]["episode_step"] = int(self.current_step)
            infos[uav_id]["out_of_battery"] = bool(self.uav_battery[uav_id] <= 0.0)
            infos[uav_id]["timeout"] = bool(timeout)

        obs = self._get_obs()
        return [obs, rewards, dones, infos]
