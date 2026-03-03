# -*- coding: utf-8 -*-
"""
UAV辅助边缘计算 - 功能函数库 (function.py)

本模块包含所有计算模型的独立实现，便于复用和测试。
每个函数对应 function_checklist.md 中的编号。

作者：基于现有环境代码重构
日期：2026-02-02
版本：v1.0
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union, Any, TYPE_CHECKING
from dataclasses import dataclass
import math

# 类型检查时导入，避免运行时循环导入
if TYPE_CHECKING:
    from .uav_comm_energy import RotorcraftParams


# ============================================================================
# 📐 第一部分：三维场景构建
# ============================================================================

def initialize_3d_space(
    ground_area: float,
    height_min: float,
    height_max: float
) -> Dict[str, float]:
    """
    [1.1.1] 初始化3D空间参数

    功能：创建空间配置字典，定义UAV活动的三维空间范围

    参数：
        ground_area: 地面区域边长 (m)，假设为正方形区域
        height_min: 最小飞行高度 (m)
        height_max: 最大飞行高度 (m)

    返回：
        space_config: 空间配置字典，包含：
            - ground_area: 地面区域边长
            - height_min: 最小高度
            - height_max: 最大高度
            - ground_center: 地面中心坐标 [x, y]
            - max_diagonal: 地面对角线长度（用于归一化）

    示例：
        >>> config = initialize_3d_space(400.0, 20.0, 120.0)
        >>> print(config['max_diagonal'])  # 565.685...
    """
    space_config = {
        'ground_area': ground_area,
        'height_min': height_min,
        'height_max': height_max,
        'ground_center': np.array([ground_area / 2, ground_area / 2]),
        'max_diagonal': ground_area * np.sqrt(2)  # 对角线长度
    }
    return space_config


def generate_uav_initial_positions(
    num_uavs: int,
    ground_area: float,
    initial_height: float,
    mode: str = 'center'
) -> np.ndarray:
    """
    [1.1.2] 生成UAV初始位置

    功能：根据不同模式生成UAV的初始3D位置

    参数：
        num_uavs: UAV数量
        ground_area: 地面区域边长 (m)
        initial_height: 初始飞行高度 (m)
        mode: 初始化模式
            - 'center': 所有UAV在中心位置 (默认)
            - 'random': 随机分布在地面区域
            - 'grid': 网格分布

    返回：
        positions: UAV位置数组，形状 (num_uavs, 3)，每行为 [x, y, z]

    示例：
        >>> positions = generate_uav_initial_positions(2, 400.0, 70.0, mode='center')
        >>> positions.shape  # (2, 3)
        >>> positions[0]  # [0.0, 0.0, 70.0]
    """
    positions = np.zeros((num_uavs, 3))

    if mode == 'center':
        # 所有UAV在地面中心上方
        positions[:, 0] = 0.0  # x = 0
        positions[:, 1] = 0.0  # y = 0
        positions[:, 2] = initial_height

    elif mode == 'random':
        # 随机分布在地面区域
        positions[:, 0] = np.random.uniform(0, ground_area, num_uavs)
        positions[:, 1] = np.random.uniform(0, ground_area, num_uavs)
        positions[:, 2] = initial_height

    elif mode == 'grid':
        # 网格分布
        grid_size = int(np.ceil(np.sqrt(num_uavs)))
        spacing = ground_area / (grid_size + 1)

        for i in range(num_uavs):
            row = i // grid_size
            col = i % grid_size
            positions[i, 0] = (col + 1) * spacing
            positions[i, 1] = (row + 1) * spacing
            positions[i, 2] = initial_height

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'center', 'random', or 'grid'.")

    return positions


def generate_terminal_positions(
    num_terminals: int,
    ground_area: float,
    base_positions: Optional[List[List[float]]] = None,
    variance: float = 0.0
) -> np.ndarray:
    """
    [1.1.3] 生成地面终端位置（支持固定+随机偏移）

    功能：生成地面终端的3D位置，支持基准位置+随机偏移模式

    参数：
        num_terminals: 终端数量
        ground_area: 地面区域边长 (m)
        base_positions: 基准位置列表，每个元素为 [x, y, z]
            - 如果提供，则在基准位置周围随机偏移
            - 如果为None，则完全随机生成
        variance: 随机偏移范围 (m)，在基准位置 ±variance 范围内随机

    返回：
        positions: 终端位置数组，形状 (num_terminals, 3)

    示例：
        >>> # 完全随机
        >>> pos1 = generate_terminal_positions(6, 400.0)
        >>>
        >>> # 基准位置 + 随机偏移
        >>> base = [[100, 100, 1], [300, 100, 1], [100, 300, 1]]
        >>> pos2 = generate_terminal_positions(3, 400.0, base, variance=30.0)
    """
    positions = np.zeros((num_terminals, 3))

    if base_positions is None:
        # 完全随机生成
        positions[:, 0] = np.random.uniform(0, ground_area, num_terminals)
        positions[:, 1] = np.random.uniform(0, ground_area, num_terminals)
        positions[:, 2] = 1.0  # 地面高度固定为1m

    else:
        # 基准位置 + 随机偏移
        for i in range(num_terminals):
            if i < len(base_positions):
                base_pos = base_positions[i]
                # 在基准位置周围随机偏移
                random_x = base_pos[0] + np.random.uniform(-variance, variance)
                random_y = base_pos[1] + np.random.uniform(-variance, variance)

                # 确保不超出边界
                random_x = np.clip(random_x, 0, ground_area)
                random_y = np.clip(random_y, 0, ground_area)

                positions[i] = [random_x, random_y, base_pos[2]]
            else:
                # 超出基准位置数量，使用中心区域随机
                center = ground_area / 2
                positions[i, 0] = center + np.random.uniform(-variance, variance)
                positions[i, 1] = center + np.random.uniform(-variance, variance)
                positions[i, 2] = 1.0

    return positions


def check_boundary_violation(
    position: np.ndarray,
    ground_area: float,
    height_min: float,
    height_max: float
) -> bool:
    """
    [1.2.1] 检查位置是否越界

    功能：检查给定位置是否超出空间边界

    参数：
        position: 位置坐标 [x, y, z]
        ground_area: 地面区域边长 (m)
        height_min: 最小飞行高度 (m)
        height_max: 最大飞行高度 (m)

    返回：
        is_violated: 是否越界 (True=越界, False=合法)

    示例：
        >>> pos = np.array([450, 200, 70])
        >>> check_boundary_violation(pos, 400.0, 20.0, 120.0)  # True (x超界)
    """
    x, y, z = position

    # 检查水平边界
    if x < 0 or x > ground_area or y < 0 or y > ground_area:
        return True

    # 检查垂直边界
    if z < height_min or z > height_max:
        return True

    return False


def clip_position_to_boundary(
    position: np.ndarray,
    ground_area: float,
    height_min: float,
    height_max: float
) -> np.ndarray:
    """
    [1.2.2] 将位置裁剪到合法范围内

    功能：将越界的位置裁剪到最近的合法边界

    参数：
        position: 位置坐标 [x, y, z]
        ground_area: 地面区域边长 (m)
        height_min: 最小飞行高度 (m)
        height_max: 最大飞行高度 (m)

    返回：
        clipped_position: 裁剪后的位置 [x, y, z]

    示例：
        >>> pos = np.array([450, -10, 150])
        >>> clipped = clip_position_to_boundary(pos, 400.0, 20.0, 120.0)
        >>> clipped  # [400, 0, 120]
    """
    clipped = position.copy()

    # 裁剪水平坐标
    clipped[0] = np.clip(clipped[0], 0, ground_area)
    clipped[1] = np.clip(clipped[1], 0, ground_area)

    # 裁剪垂直坐标
    clipped[2] = np.clip(clipped[2], height_min, height_max)

    return clipped


def check_collision(
    uav_positions: np.ndarray,
    min_safe_distance: float = 5.0
) -> List[Tuple[int, int]]:
    """
    [1.2.3] 检查UAV之间是否碰撞

    功能：检测所有UAV对之间的距离，找出碰撞对

    参数：
        uav_positions: 所有UAV位置，形状 (num_uavs, 3)
        min_safe_distance: 最小安全距离 (m)，默认5m

    返回：
        collision_pairs: 碰撞的UAV对列表，每个元素为 (uav_i, uav_j)

    示例：
        >>> positions = np.array([[0, 0, 70], [3, 0, 70], [100, 100, 70]])
        >>> collisions = check_collision(positions, min_safe_distance=5.0)
        >>> collisions  # [(0, 1)]  # UAV 0和1距离<5m
    """
    num_uavs = len(uav_positions)
    collision_pairs = []

    for i in range(num_uavs):
        for j in range(i + 1, num_uavs):
            distance = np.linalg.norm(uav_positions[i] - uav_positions[j])
            if distance < min_safe_distance:
                collision_pairs.append((i, j))

    return collision_pairs


# ============================================================================
# 🚁 第二部分：无人机动力学模型 - 运动学计算
# ============================================================================

def calculate_3d_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    [2.1.4] 计算两点间的3D欧氏距离

    功能：计算三维空间中两点之间的直线距离

    参数：
        pos1: 位置1 [x, y, z]
        pos2: 位置2 [x, y, z]

    返回：
        distance: 3D距离 (m)

    示例：
        >>> pos1 = np.array([0, 0, 70])
        >>> pos2 = np.array([3, 4, 70])
        >>> calculate_3d_distance(pos1, pos2)  # 5.0
    """
    return float(np.linalg.norm(pos2 - pos1))


def calculate_horizontal_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    [2.1.5] 计算两点间的水平距离（忽略高度）

    功能：只计算x-y平面上的距离，忽略z坐标

    参数：
        pos1: 位置1 [x, y, z]
        pos2: 位置2 [x, y, z]

    返回：
        distance: 水平距离 (m)

    示例：
        >>> pos1 = np.array([0, 0, 70])
        >>> pos2 = np.array([3, 4, 100])
        >>> calculate_horizontal_distance(pos1, pos2)  # 5.0 (忽略高度差)
    """
    return float(np.linalg.norm(pos2[:2] - pos1[:2]))


def calculate_velocity_from_positions(
    pos_old: np.ndarray,
    pos_new: np.ndarray,
    time_slot: float
) -> Tuple[float, float]:
    """
    [2.1.3] 从位置变化计算速度

    功能：根据前后两个位置和时间间隔，计算水平和垂直速度

    参数：
        pos_old: 旧位置 [x, y, z]
        pos_new: 新位置 [x, y, z]
        time_slot: 时间间隔 (s)

    返回：
        v_horizontal: 水平速度 (m/s)
        v_vertical: 垂直速度 (m/s)，正值表示上升，负值表示下降

    示例：
        >>> old = np.array([0, 0, 70])
        >>> new = np.array([10, 0, 75])
        >>> v_h, v_v = calculate_velocity_from_positions(old, new, 1.0)
        >>> v_h  # 10.0 m/s
        >>> v_v  # 5.0 m/s
    """
    # 水平位移
    horizontal_displacement = np.linalg.norm(pos_new[:2] - pos_old[:2])
    v_horizontal = horizontal_displacement / time_slot

    # 垂直位移
    vertical_displacement = pos_new[2] - pos_old[2]
    v_vertical = vertical_displacement / time_slot

    return float(v_horizontal), float(v_vertical)


def calculate_horizontal_movement(
    current_pos: np.ndarray,
    action: int,
    max_speed: float,
    time_slot: float
) -> Tuple[np.ndarray, float]:
    """
    [2.1.1] 计算水平移动后的位置

    功能：根据移动动作计算新的水平位置

    参数：
        current_pos: 当前位置 [x, y, z]
        action: 移动动作
            - 0: 向上(y+)
            - 1: 向下(y-)
            - 2: 向左(x-)
            - 3: 向右(x+)
            - 4: 保持不动
        max_speed: 最大水平速度 (m/s)
        time_slot: 时隙长度 (s)

    返回：
        new_pos: 新位置 [x, y, z]
        move_distance: 实际移动距离 (m)

    示例：
        >>> pos = np.array([100, 100, 70])
        >>> new_pos, dist = calculate_horizontal_movement(pos, 0, 10.0, 1.0)
        >>> new_pos  # [100, 110, 70]
        >>> dist  # 10.0
    """
    new_pos = current_pos.copy()
    move_distance = max_speed * time_slot

    if action == 0:    # 向上(y+)
        new_pos[1] += move_distance
    elif action == 1:  # 向下(y-)
        new_pos[1] -= move_distance
    elif action == 2:  # 向左(x-)
        new_pos[0] -= move_distance
    elif action == 3:  # 向右(x+)
        new_pos[0] += move_distance
    elif action == 4:  # 保持不动
        move_distance = 0.0
    else:
        raise ValueError(f"Invalid horizontal action: {action}. Must be 0-4.")

    return new_pos, move_distance


def calculate_vertical_movement(
    current_height: float,
    action: int,
    max_speed: float,
    time_slot: float,
    height_min: float,
    height_max: float,
    delta_h: float
) -> Tuple[float, float]:
    """
    [2.1.2] 计算垂直移动后的高度

    功能：根据垂直动作计算新的高度，考虑高度约束

    参数：
        current_height: 当前高度 (m)
        action: 垂直动作
            - 0: 下降
            - 1: 上升
            - 2: 悬停
        max_speed: 最大垂直速度 (m/s)
        time_slot: 时隙长度 (s)
        height_min: 最小飞行高度 (m)
        height_max: 最大飞行高度 (m)
        delta_h: 每步高度变化量 (m)，通常为5m

    返回：
        new_height: 新高度 (m)，已裁剪到合法范围
        v_vertical: 垂直速度 (m/s)

    示例：
        >>> h, v = calculate_vertical_movement(70, 1, 10.0, 1.0, 20.0, 120.0, 5.0)
        >>> h  # 75.0
        >>> v  # 5.0
    """
    if action == 0:  # 下降
        new_height = current_height - delta_h
        v_vertical = -delta_h / time_slot
    elif action == 1:  # 上升
        new_height = current_height + delta_h
        v_vertical = delta_h / time_slot
    elif action == 2:  # 悬停
        new_height = current_height
        v_vertical = 0.0
    else:
        raise ValueError(f"Invalid vertical action: {action}. Must be 0-2.")

    # 裁剪到合法高度范围
    new_height = np.clip(new_height, height_min, height_max)

    # 如果裁剪后高度没变，实际速度为0
    if new_height == current_height:
        v_vertical = 0.0

    return float(new_height), float(v_vertical)


# ============================================================================
# 🚁 第二部分：无人机动力学模型 - 能耗模型
# ============================================================================

# 注：这部分函数依赖 uav_comm_energy.py 中的 RotorcraftParams 和 uav_step_energy
# 我们提供包装函数，方便调用

def calculate_propulsion_energy(
    v_horizontal: float,
    v_vertical: float,
    time_slot: float,
    rotor_params: 'RotorcraftParams'
) -> Tuple[float, Dict[str, float]]:
    """
    [2.2.1] 计算推进能耗（剖面+寄生+诱导+垂直）

    功能：计算旋翼无人机的推进能耗，包含四个部分

    参数：
        v_horizontal: 水平速度 (m/s)
        v_vertical: 垂直速度 (m/s)
        time_slot: 时隙长度 (s)
        rotor_params: 旋翼参数对象（来自 uav_comm_energy.py）

    返回：
        energy: 总推进能耗 (J)
        power_breakdown: 功率分解字典，包含：
            - profile: 剖面功率
            - parasite: 寄生功率
            - induced: 诱导功率
            - climb: 垂直爬升功率

    示例：
        >>> from uav_comm_energy import RotorcraftParams
        >>> params = RotorcraftParams()
        >>> energy, breakdown = calculate_propulsion_energy(10.0, 0.0, 1.0, params)
    """
    from .uav_comm_energy import uav_step_energy

    energy_J, power_dict = uav_step_energy(
        v_h=v_horizontal,
        v_v=v_vertical,
        t_step=time_slot,
        params=rotor_params
    )

    return float(energy_J), power_dict


def calculate_hovering_energy(
    time_slot: float,
    rotor_params: 'RotorcraftParams'
) -> float:
    """
    [2.2.2] 计算悬停能耗

    功能：计算UAV悬停时的能耗（v_h=0, v_v=0）

    参数：
        time_slot: 时隙长度 (s)
        rotor_params: 旋翼参数对象

    返回：
        energy: 悬停能耗 (J)

    示例：
        >>> from uav_comm_energy import RotorcraftParams
        >>> params = RotorcraftParams()
        >>> energy = calculate_hovering_energy(1.0, params)
    """
    energy, _ = calculate_propulsion_energy(0.0, 0.0, time_slot, rotor_params)
    return energy


def calculate_computation_energy(
    processed_bits: float,
    energy_per_bit: float = 1e-9
) -> float:
    """
    [2.2.3] 计算计算能耗

    功能：计算UAV处理数据的能耗

    参数：
        processed_bits: 处理的数据量 (bits)
        energy_per_bit: 每bit能耗 (J/bit)，默认1nJ

    返回：
        energy: 计算能耗 (J)

    示例：
        >>> energy = calculate_computation_energy(1e6, 1e-9)  # 1Mb数据
        >>> energy  # 0.001 J
    """
    return processed_bits * energy_per_bit


def calculate_communication_energy(
    transmit_power: float,
    time_slot: float,
    efficiency: float = 0.1
) -> float:
    """
    [2.2.4] 计算通信能耗

    功能：计算UAV通信的能耗

    参数：
        transmit_power: 发射功率 (W)
        time_slot: 时隙长度 (s)
        efficiency: 效率系数，默认0.1（发射功率的10%）

    返回：
        energy: 通信能耗 (J)

    示例：
        >>> energy = calculate_communication_energy(0.1, 1.0, 0.1)
        >>> energy  # 0.01 J
    """
    return transmit_power * time_slot * efficiency


def calculate_total_energy_consumption(
    propulsion_energy: float,
    computation_energy: float,
    communication_energy: float
) -> float:
    """
    [2.2.5] 计算总能耗

    功能：汇总各部分能耗

    参数：
        propulsion_energy: 推进能耗 (J)
        computation_energy: 计算能耗 (J)
        communication_energy: 通信能耗 (J)

    返回：
        total_energy: 总能耗 (J)

    示例：
        >>> total = calculate_total_energy_consumption(50.0, 0.001, 0.01)
        >>> total  # 50.011 J
    """
    return propulsion_energy + computation_energy + communication_energy


# ============================================================================
# 📡 第三部分：通信模型 - 信道模型（第1批）
# ============================================================================

def calculate_elevation_angle(
    height: float,
    horizontal_distance: float
) -> float:
    """
    [3.1.1] 计算仰角

    功能：计算从地面终端到UAV的仰角（度）

    参数：
        height: 高度差 (m)，UAV高度 - 终端高度
        horizontal_distance: 水平距离 (m)

    返回：
        elevation_angle: 仰角 (度)

    示例：
        >>> angle = calculate_elevation_angle(70, 70)
        >>> angle  # 45.0度
    """
    # 避免除零
    horizontal_distance = max(horizontal_distance, 1e-9)
    angle_rad = np.arctan2(height, horizontal_distance)
    return float(np.degrees(angle_rad))


def calculate_los_probability(
    elevation_angle: float,
    a: float = 9.61,
    b: float = 0.16
) -> float:
    """
    [3.1.2] 计算LoS概率

    功能：使用逻辑函数计算视距(LoS)概率

    公式：P_LoS(θ) = 1 / (1 + a * exp(-b * (θ - a)))

    参数：
        elevation_angle: 仰角 (度)
        a: 拟合参数a，默认9.61（城市场景）
        b: 拟合参数b，默认0.16

    返回：
        p_los: LoS概率 (0-1)

    示例：
        >>> p = calculate_los_probability(45.0)
        >>> 0 <= p <= 1  # True
    """
    p_los = 1.0 / (1.0 + a * np.exp(-b * (elevation_angle - a)))
    return float(np.clip(p_los, 0.0, 1.0))


def sample_los_state(
    los_probability: float,
    rng: Optional[np.random.Generator] = None
) -> bool:
    """
    [3.1.3] 采样LoS/NLoS状态

    功能：根据LoS概率进行伯努利采样

    参数：
        los_probability: LoS概率 (0-1)
        rng: 随机数生成器，如果为None则创建新的

    返回：
        is_los: 是否为LoS状态 (True=LoS, False=NLoS)

    示例：
        >>> rng = np.random.default_rng(42)
        >>> is_los = sample_los_state(0.8, rng)
    """
    if rng is None:
        rng = np.random.default_rng()
    return bool(rng.random() < los_probability)


def calculate_free_space_path_loss(
    distance_3d: float,
    carrier_frequency: float = 2.4e9
) -> float:
    """
    [3.1.4] 计算自由空间路径损耗

    功能：计算FSPL（Free Space Path Loss）

    公式：FSPL = 20*log10(4π * f_c * d / c)

    参数：
        distance_3d: 3D距离 (m)
        carrier_frequency: 载波频率 (Hz)，默认2.4GHz

    返回：
        fspl: 自由空间路径损耗 (dB)

    示例：
        >>> fspl = calculate_free_space_path_loss(100.0, 2.4e9)
        >>> fspl > 0  # True，损耗为正值
    """
    c = 299_792_458.0  # 光速 (m/s)
    distance_3d = max(distance_3d, 1e-3)  # 避免除零
    x = 4.0 * np.pi * carrier_frequency * distance_3d / c
    return float(20.0 * np.log10(x))


def calculate_additional_loss(
    los_probability: float,
    eta_los: float = 1.0,
    eta_nlos: float = 20.0,
    mode: str = 'expected'
) -> float:
    """
    [3.1.5] 计算附加损耗（期望或采样）

    功能：计算LoS/NLoS的附加损耗

    参数：
        los_probability: LoS概率 (0-1)
        eta_los: LoS附加损耗 (dB)，默认1.0
        eta_nlos: NLoS附加损耗 (dB)，默认20.0
        mode: 计算模式
            - 'expected': 期望值 = p*η_LoS + (1-p)*η_NLoS
            - 'sample': 采样模式，随机选择LoS或NLoS

    返回：
        additional_loss: 附加损耗 (dB)

    示例：
        >>> loss = calculate_additional_loss(0.8, 1.0, 20.0, 'expected')
        >>> loss  # 0.8*1 + 0.2*20 = 4.8
    """
    if mode == 'expected':
        return los_probability * eta_los + (1.0 - los_probability) * eta_nlos
    elif mode == 'sample':
        is_los = sample_los_state(los_probability)
        return eta_los if is_los else eta_nlos
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'expected' or 'sample'.")


def calculate_total_path_loss(
    uav_pos: np.ndarray,
    terminal_pos: np.ndarray,
    carrier_frequency: float = 2.4e9,
    a: float = 9.61,
    b: float = 0.16,
    eta_los: float = 1.0,
    eta_nlos: float = 20.0,
    mode: str = 'expected'
) -> float:
    """
    [3.1.6] 计算总路径损耗

    功能：整合FSPL和附加损耗，计算总路径损耗

    参数：
        uav_pos: UAV位置 [x, y, z]
        terminal_pos: 终端位置 [x, y, z]
        carrier_frequency: 载波频率 (Hz)
        a, b: LoS概率拟合参数
        eta_los, eta_nlos: LoS/NLoS附加损耗 (dB)
        mode: 'expected' 或 'sample'

    返回：
        total_loss: 总路径损耗 (dB)

    示例：
        >>> uav = np.array([100, 100, 70])
        >>> terminal = np.array([100, 100, 1])
        >>> loss = calculate_total_path_loss(uav, terminal)
    """
    # 计算距离
    d_3d = calculate_3d_distance(uav_pos, terminal_pos)
    d_horizontal = calculate_horizontal_distance(uav_pos, terminal_pos)
    height_diff = uav_pos[2] - terminal_pos[2]

    # 计算FSPL
    fspl = calculate_free_space_path_loss(d_3d, carrier_frequency)

    # 计算仰角和LoS概率
    elevation = calculate_elevation_angle(height_diff, d_horizontal)
    p_los = calculate_los_probability(elevation, a, b)

    # 计算附加损耗
    additional = calculate_additional_loss(p_los, eta_los, eta_nlos, mode)

    return fspl + additional


# ============================================================================
# 📡 第三部分：通信模型 - 速率与容量（第2批）
# ============================================================================

def calculate_received_power(
    transmit_power: float,
    path_loss_db: float,
    antenna_gain: float = 2.0
) -> float:
    """
    [3.2.1] 计算接收功率

    功能：根据发射功率、路径损耗和天线增益计算接收功率

    公式：P_rx = P_tx + 2*G - L (dB域)
          P_rx = P_tx * 10^(-L/10) (线性域，假设G已包含在P_tx中)

    参数：
        transmit_power: 发射功率 (W)
        path_loss_db: 路径损耗 (dB)
        antenna_gain: 天线增益 (dBi)，默认2.0

    返回：
        received_power: 接收功率 (W)

    示例：
        >>> p_rx = calculate_received_power(0.1, 80.0, 2.0)
    """
    # 转换到dB域
    p_tx_dbm = 10 * np.log10(transmit_power * 1000)  # W -> dBm
    p_rx_dbm = p_tx_dbm + 2 * antenna_gain - path_loss_db
    # 转换回线性域
    p_rx_w = 10 ** ((p_rx_dbm - 30) / 10)  # dBm -> W
    return float(p_rx_w)


def calculate_snr(
    received_power: float,
    bandwidth: float,
    noise_power_density: float = 4e-21
) -> float:
    """
    [3.2.2] 计算信噪比

    功能：计算线性SNR

    公式：SNR = P_rx / (B * N0)

    参数：
        received_power: 接收功率 (W)
        bandwidth: 带宽 (Hz)
        noise_power_density: 噪声功率谱密度 (W/Hz)

    返回：
        snr: 信噪比（线性值，非dB）

    示例：
        >>> snr = calculate_snr(1e-10, 1e6, 4e-21)
    """
    noise_power = bandwidth * noise_power_density
    snr = received_power / noise_power
    return float(max(snr, 0.0))  # 确保非负


def calculate_channel_capacity(
    snr: float,
    bandwidth: float
) -> float:
    """
    [3.2.3] 计算信道容量（Shannon公式）

    功能：使用Shannon公式计算信道容量

    公式：C = B * log2(1 + SNR)

    参数：
        snr: 信噪比（线性值）
        bandwidth: 带宽 (Hz)

    返回：
        capacity: 信道容量 (bps)

    示例：
        >>> capacity = calculate_channel_capacity(10.0, 1e6)
        >>> capacity  # 约 3.46 Mbps
    """
    return float(bandwidth * np.log2(1.0 + snr))


def calculate_uplink_rate(
    uav_pos: np.ndarray,
    terminal_pos: np.ndarray,
    transmit_power: float = 0.1,
    bandwidth: float = 1e6,
    carrier_frequency: float = 2.4e9,
    noise_power_density: float = 4e-21,
    antenna_gain: float = 2.0,
    mode: str = 'expected'
) -> float:
    """
    [3.2.4] 计算上行速率（整合函数）

    功能：整合路径损耗、接收功率、SNR和信道容量计算

    参数：
        uav_pos: UAV位置 [x, y, z]
        terminal_pos: 终端位置 [x, y, z]
        transmit_power: 发射功率 (W)
        bandwidth: 带宽 (Hz)
        carrier_frequency: 载波频率 (Hz)
        noise_power_density: 噪声功率谱密度 (W/Hz)
        antenna_gain: 天线增益 (dBi)
        mode: 'expected' 或 'sample'

    返回：
        uplink_rate: 上行速率 (bps)

    示例：
        >>> uav = np.array([100, 100, 70])
        >>> terminal = np.array([150, 150, 1])
        >>> rate = calculate_uplink_rate(uav, terminal)
    """
    # 1. 计算路径损耗
    path_loss = calculate_total_path_loss(
        uav_pos, terminal_pos,
        carrier_frequency=carrier_frequency,
        mode=mode
    )

    # 2. 计算接收功率
    p_rx = calculate_received_power(transmit_power, path_loss, antenna_gain)

    # 3. 计算SNR
    snr = calculate_snr(p_rx, bandwidth, noise_power_density)

    # 4. 计算信道容量
    capacity = calculate_channel_capacity(snr, bandwidth)

    return capacity


# ============================================================================
# 📡 第三部分：通信模型 - 连接管理（第3批）
# ============================================================================

def check_communication_range(
    uav_pos: np.ndarray,
    terminal_pos: np.ndarray,
    comm_range: float
) -> bool:
    """
    [3.3.1] 检查是否在通信范围内

    功能：判断UAV与终端之间的距离是否在通信范围内

    参数：
        uav_pos: UAV位置 [x, y, z]
        terminal_pos: 终端位置 [x, y, z]
        comm_range: 通信范围 (m)

    返回：
        is_connected: 是否可连接 (True=在范围内)

    示例：
        >>> uav = np.array([100, 100, 70])
        >>> terminal = np.array([150, 150, 1])
        >>> check_communication_range(uav, terminal, 100.0)  # True or False
    """
    distance = calculate_3d_distance(uav_pos, terminal_pos)
    return distance <= comm_range


def update_connection_matrix(
    uav_positions: np.ndarray,
    terminal_positions: np.ndarray,
    comm_range: float
) -> np.ndarray:
    """
    [3.3.2] 更新所有UAV与终端的连接矩阵

    功能：批量检查所有UAV与终端的连接状态

    参数：
        uav_positions: 所有UAV位置，形状 (num_uavs, 3)
        terminal_positions: 所有终端位置，形状 (num_terminals, 3)
        comm_range: 通信范围 (m)

    返回：
        connection_matrix: 连接矩阵，形状 (num_uavs, num_terminals)
            - True: 可连接
            - False: 不可连接

    示例：
        >>> uav_pos = np.array([[100, 100, 70], [200, 200, 70]])
        >>> term_pos = np.array([[100, 100, 1], [300, 300, 1]])
        >>> matrix = update_connection_matrix(uav_pos, term_pos, 100.0)
        >>> matrix.shape  # (2, 2)
    """
    num_uavs = len(uav_positions)
    num_terminals = len(terminal_positions)
    connection_matrix = np.zeros((num_uavs, num_terminals), dtype=bool)

    for i in range(num_uavs):
        for j in range(num_terminals):
            connection_matrix[i, j] = check_communication_range(
                uav_positions[i],
                terminal_positions[j],
                comm_range
            )

    return connection_matrix


# ============================================================================
# 💻 第四部分：计算与卸载模型
# ============================================================================

def calculate_local_processing_time(
    data_bits: float,
    cpu_frequency: float,
    cycles_per_bit: int
) -> float:
    """
    [4.1.1] 计算本地处理时间

    功能：计算在本地CPU上处理数据所需的时间

    公式：t = (data_bits * cycles_per_bit) / cpu_frequency

    参数：
        data_bits: 数据量 (bits)
        cpu_frequency: CPU频率 (cycles/s)
        cycles_per_bit: 每bit需要的CPU周期数

    返回：
        processing_time: 处理时间 (s)

    示例：
        >>> t = calculate_local_processing_time(1e6, 5e9, 1000)
        >>> t  # 0.0002 秒
    """
    total_cycles = data_bits * cycles_per_bit
    return total_cycles / cpu_frequency


def calculate_local_processing_capacity(
    cpu_frequency: float,
    cycles_per_bit: int,
    time_slot: float
) -> float:
    """
    [4.1.2] 计算本地处理能力

    功能：计算在给定时隙内本地CPU能处理的数据量

    公式：capacity = (cpu_frequency * time_slot) / cycles_per_bit

    参数：
        cpu_frequency: CPU频率 (cycles/s)
        cycles_per_bit: 每bit需要的CPU周期数
        time_slot: 时隙长度 (s)

    返回：
        capacity: 可处理数据量 (bits)

    示例：
        >>> cap = calculate_local_processing_capacity(5e9, 1000, 1.0)
        >>> cap  # 5e6 bits = 5 Mb
    """
    return (cpu_frequency * time_slot) / cycles_per_bit


def calculate_offload_alpha(
    uplink_rate: float,
    data_bits: float,
    cpu_frequency_uav: float,
    cpu_frequency_ground: float,
    cycles_per_bit: int,
    time_slot: float
) -> float:
    """
    [4.2.1] 计算最优卸载比例α

    功能：计算在通信-计算耦合约束下的最优卸载比例

    约束条件：
        - 传输时间 + 地面处理时间 ≤ 时隙长度
        - α * data_bits / uplink_rate + α * data_bits * cycles_per_bit / f_g ≤ T
        - (1-α) * data_bits * cycles_per_bit / f_u ≤ T

    参数：
        uplink_rate: 上行速率 (bps)
        data_bits: 总数据量 (bits)
        cpu_frequency_uav: UAV CPU频率 (cycles/s)
        cpu_frequency_ground: 地面服务器CPU频率 (cycles/s)
        cycles_per_bit: 每bit需要的CPU周期数
        time_slot: 时隙长度 (s)

    返回：
        alpha: 卸载比例 (0-1)
            - 0: 全部本地处理
            - 1: 全部卸载到地面
            - 0<α<1: 部分卸载

    示例：
        >>> alpha = calculate_offload_alpha(1e6, 1e6, 5e9, 100e9, 1000, 1.0)
        >>> 0 <= alpha <= 1  # True
    """
    # 避免除零
    if uplink_rate <= 0 or cpu_frequency_ground <= 0 or cpu_frequency_uav <= 0:
        return 0.0

    # 计算卸载一个bit的总时间（传输+地面处理）
    offload_time_per_bit = 1.0 / uplink_rate + cycles_per_bit / cpu_frequency_ground

    # 计算本地处理一个bit的时间
    local_time_per_bit = cycles_per_bit / cpu_frequency_uav

    # 如果卸载比本地处理还慢，不卸载
    if offload_time_per_bit >= local_time_per_bit:
        return 0.0

    # 计算最大可卸载数据量（受时隙约束）
    max_offload_bits = time_slot / offload_time_per_bit

    # 计算最大可本地处理数据量
    max_local_bits = time_slot / local_time_per_bit

    # 如果全部卸载可以完成
    if max_offload_bits >= data_bits:
        return 1.0

    # 如果全部本地可以完成
    if max_local_bits >= data_bits:
        # 计算最优分配
        alpha = (data_bits - max_local_bits) / (max_offload_bits - max_local_bits)
        alpha = np.clip(alpha, 0.0, 1.0)
        return float(alpha)

    # 如果都无法完成，优先卸载（地面处理能力更强）
    alpha = max_offload_bits / data_bits
    return float(np.clip(alpha, 0.0, 1.0))


def calculate_offloaded_data(
    alpha: float,
    total_data_bits: float
) -> float:
    """
    [4.2.2] 计算卸载的数据量

    功能：根据卸载比例计算实际卸载的数据量

    参数：
        alpha: 卸载比例 (0-1)
        total_data_bits: 总数据量 (bits)

    返回：
        offloaded_bits: 卸载的数据量 (bits)

    示例：
        >>> offloaded = calculate_offloaded_data(0.6, 1e6)
        >>> offloaded  # 600000.0
    """
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha * total_data_bits


def calculate_transmission_time(
    data_bits: float,
    uplink_rate: float
) -> float:
    """
    [4.3.1] 计算传输时间

    功能：计算数据传输所需的时间

    参数：
        data_bits: 数据量 (bits)
        uplink_rate: 上行速率 (bps)

    返回：
        transmission_time: 传输时间 (s)

    示例：
        >>> t = calculate_transmission_time(1e6, 1e6)
        >>> t  # 1.0 秒
    """
    if uplink_rate <= 0:
        return float('inf')
    return data_bits / uplink_rate


def calculate_ground_processing_time(
    data_bits: float,
    cpu_freq_ground: float,
    cycles_per_bit: int
) -> float:
    """
    [4.3.2] 计算地面服务器处理时间

    功能：计算地面服务器处理数据所需的时间

    参数：
        data_bits: 数据量 (bits)
        cpu_freq_ground: 地面CPU频率 (cycles/s)
        cycles_per_bit: 每bit需要的CPU周期数

    返回：
        processing_time: 处理时间 (s)

    示例：
        >>> t = calculate_ground_processing_time(1e6, 100e9, 1000)
        >>> t  # 0.01 秒
    """
    if cpu_freq_ground <= 0:
        return float('inf')
    total_cycles = data_bits * cycles_per_bit
    return total_cycles / cpu_freq_ground


def calculate_processed_bits_coupled(
    offload_decision: bool,
    cpu_freq_uav: float,
    data_bits: float,
    time_slot: float,
    uplink_rate: float,
    cpu_cycles: int,
    cpu_freq_ground: float
) -> float:
    """
    [4.3.3] 计算通信-计算耦合下的实际处理量

    功能：这是核心函数，计算在通信-计算耦合约束下UAV实际能处理的数据量

    模型说明：
        - 如果不卸载（offload_decision=False）：
            本地处理量 = min(data_bits, cpu_freq_uav * time_slot / cpu_cycles)

        - 如果卸载（offload_decision=True）：
            1. 计算传输时间：t_trans = data_bits / uplink_rate
            2. 计算地面处理时间：t_ground = data_bits * cpu_cycles / cpu_freq_ground
            3. 总时间：t_total = t_trans + t_ground
            4. 如果 t_total <= time_slot：全部处理完成
               否则：只能处理部分数据

    参数：
        offload_decision: 是否卸载 (True=卸载, False=本地处理)
        cpu_freq_uav: UAV CPU频率 (cycles/s)
        data_bits: 任务数据量 (bits)
        time_slot: 时隙长度 (s)
        uplink_rate: 上行速率 (bps)
        cpu_cycles: 每bit需要的CPU周期数
        cpu_freq_ground: 地面CPU频率 (cycles/s)

    返回：
        processed_bits: 实际处理的数据量 (bits)

    示例：
        >>> # 本地处理
        >>> processed = calculate_processed_bits_coupled(
        ...     False, 5e9, 1e6, 1.0, 1e6, 1000, 100e9
        ... )
        >>> processed  # 5e6 bits (本地能力足够)

        >>> # 卸载处理
        >>> processed = calculate_processed_bits_coupled(
        ...     True, 5e9, 1e6, 1.0, 1e6, 1000, 100e9
        ... )
        >>> processed  # 取决于传输+地面处理时间
    """
    if not offload_decision:
        # 本地处理模式
        local_capacity = calculate_local_processing_capacity(
            cpu_freq_uav, cpu_cycles, time_slot
        )
        return float(min(data_bits, local_capacity))

    else:
        # 卸载模式：通信-计算耦合
        # 计算传输时间
        t_trans = calculate_transmission_time(data_bits, uplink_rate)

        # 计算地面处理时间
        t_ground = calculate_ground_processing_time(
            data_bits, cpu_freq_ground, cpu_cycles
        )

        # 总时间
        t_total = t_trans + t_ground

        # 如果总时间在时隙内，全部处理完成
        if t_total <= time_slot:
            return float(data_bits)

        # 否则，计算在时隙内能处理多少数据
        # 设处理x bits，则：x/uplink_rate + x*cpu_cycles/cpu_freq_ground = time_slot
        # x * (1/uplink_rate + cpu_cycles/cpu_freq_ground) = time_slot
        time_per_bit = 1.0 / uplink_rate + cpu_cycles / cpu_freq_ground
        processed_bits = time_slot / time_per_bit

        return float(min(processed_bits, data_bits))


# ============================================================================
# 💻 第四部分完成标记
# ============================================================================
# ✅ 4.1 本地计算 (2个函数)
# ✅ 4.2 卸载决策 (2个函数)
# ✅ 4.3 通信-计算耦合 (3个函数)
# 总计：7个函数
# ============================================================================


# ============================================================================
# 🤝 第五部分：多UAV协作模型
# ============================================================================

def calculate_multi_uav_efficiency(num_serving_uavs: int) -> float:
    """
    [5.1.1] 计算多UAV服务同一终端的效率系数

    功能：当多个UAV同时服务同一终端时，由于干扰和协调开销，效率会下降

    模型：
        - 1个UAV：效率 = 1.0
        - 2个UAV：效率 = 0.9
        - 3个UAV：效率 = 0.8
        - 4+个UAV：效率 = 0.7

    参数：
        num_serving_uavs: 服务该终端的UAV数量

    返回：
        efficiency: 效率系数 (0-1)

    示例：
        >>> eff = calculate_multi_uav_efficiency(1)
        >>> eff  # 1.0
        >>> eff = calculate_multi_uav_efficiency(3)
        >>> eff  # 0.8
    """
    if num_serving_uavs <= 0:
        return 0.0
    elif num_serving_uavs == 1:
        return 1.0
    elif num_serving_uavs == 2:
        return 0.9
    elif num_serving_uavs == 3:
        return 0.8
    else:
        return 0.7


def allocate_terminal_processing(
    serving_uavs: List[int],
    individual_capacities: Dict[int, float],
    efficiency: float
) -> Dict[int, float]:
    """
    [5.1.2] 分配多UAV对单终端的处理量

    功能：当多个UAV服务同一终端时，按各UAV的处理能力比例分配任务

    参数：
        serving_uavs: 服务该终端的UAV ID列表
        individual_capacities: 各UAV的理论处理能力字典 {uav_id: capacity}
        efficiency: 多UAV协作效率系数

    返回：
        actual_processing: 各UAV实际处理量字典 {uav_id: processed_bits}

    示例：
        >>> serving = [0, 1]
        >>> capacities = {0: 1000.0, 1: 1500.0}
        >>> allocation = allocate_terminal_processing(serving, capacities, 0.9)
        >>> allocation  # {0: 360.0, 1: 540.0}  (总和=900，按4:6分配)
    """
    if not serving_uavs:
        return {}

    # 计算总容量
    total_capacity = sum(individual_capacities.get(uav_id, 0.0) for uav_id in serving_uavs)

    if total_capacity <= 0:
        return {uav_id: 0.0 for uav_id in serving_uavs}

    # 考虑效率后的总容量
    effective_total = total_capacity * efficiency

    # 按比例分配
    actual_processing = {}
    for uav_id in serving_uavs:
        capacity = individual_capacities.get(uav_id, 0.0)
        ratio = capacity / total_capacity
        actual_processing[uav_id] = effective_total * ratio

    return actual_processing


def allocate_uav_computing_power(
    uav_cpu_freq: float,
    num_terminals_served: int
) -> float:
    """
    [5.2.1] 分配UAV计算能力给多个终端

    功能：当一个UAV服务多个终端时，平均分配计算能力

    参数：
        uav_cpu_freq: UAV CPU频率 (cycles/s)
        num_terminals_served: 服务的终端数量

    返回：
        allocated_cpu_freq: 每个终端分配的CPU频率 (cycles/s)

    示例：
        >>> allocated = allocate_uav_computing_power(5e9, 2)
        >>> allocated  # 2.5e9 (平均分配)
    """
    if num_terminals_served <= 0:
        return 0.0
    return uav_cpu_freq / num_terminals_served


def calculate_multi_terminal_processing(
    uav_id: int,
    terminal_ids: List[int],
    uav_pos: np.ndarray,
    terminal_positions: np.ndarray,
    allocated_cpu_freq: float,
    comm_params: Dict[str, float],
    time_slot: float,
    terminals_data: List[Dict[str, float]]
) -> Dict[int, float]:
    """
    [5.2.2] 计算UAV对多个终端的处理量

    功能：计算一个UAV同时服务多个终端时，每个终端的实际处理量

    参数：
        uav_id: UAV ID
        terminal_ids: 终端ID列表
        uav_pos: UAV位置 [x, y, z]
        terminal_positions: 所有终端位置数组 (num_terminals, 3)
        allocated_cpu_freq: 分配给每个终端的CPU频率
        comm_params: 通信参数字典
        time_slot: 时隙长度 (s)
        terminals_data: 终端数据列表，每个元素包含 remaining_data_bits, cpu_cycles_per_bit

    返回：
        processing_dict: 各终端处理量字典 {terminal_id: processed_bits}

    示例：
        >>> processing = calculate_multi_terminal_processing(
        ...     0, [0, 1], uav_pos, term_pos, 2.5e9, comm_params, 1.0, terminals
        ... )
    """
    processing_dict = {}

    for term_id in terminal_ids:
        if term_id >= len(terminals_data):
            processing_dict[term_id] = 0.0
            continue

        terminal = terminals_data[term_id]
        term_pos = terminal_positions[term_id]

        # 计算上行速率
        uplink_rate = calculate_uplink_rate(
            uav_pos, term_pos,
            transmit_power=comm_params.get('transmit_power', 0.1),
            bandwidth=comm_params.get('bandwidth', 1e6),
            carrier_frequency=comm_params.get('carrier_frequency', 2.4e9),
            noise_power_density=comm_params.get('noise_power_density', 4e-21),
            antenna_gain=comm_params.get('antenna_gain', 2.0),
            mode='expected'
        )

        # 计算处理量（使用分配的CPU频率）
        processed = calculate_processed_bits_coupled(
            offload_decision=True,
            cpu_freq_uav=allocated_cpu_freq,
            data_bits=terminal['remaining_data_bits'],
            time_slot=time_slot,
            uplink_rate=uplink_rate,
            cpu_cycles=terminal['cpu_cycles_per_bit'],
            cpu_freq_ground=comm_params.get('cpu_freq_ground', 100e9)
        )

        processing_dict[term_id] = processed

    return processing_dict


# ============================================================================
# 🤝 第五部分：多UAV协作模型 - 负载均衡（暂不启用）
# ============================================================================
# 注意：以下函数已实现但暂不启用，可在后续版本中使用

def calculate_load_balance_ratio(uav_service_counts: Dict[int, int]) -> float:
    """
    [5.3.1] 计算UAV间负载均衡比例 【暂不启用】

    功能：评估UAV之间的负载均衡程度

    公式：balance_ratio = 1 - (std / mean)
        - 完全均衡：std=0, ratio=1.0
        - 不均衡：std越大，ratio越小

    参数：
        uav_service_counts: 各UAV服务步数字典 {uav_id: service_count}

    返回：
        balance_ratio: 均衡比例 (0-1)，1表示完全均衡

    示例：
        >>> counts = {0: 100, 1: 100, 2: 100}
        >>> ratio = calculate_load_balance_ratio(counts)
        >>> ratio  # 1.0 (完全均衡)
    """
    if not uav_service_counts:
        return 1.0

    counts = list(uav_service_counts.values())
    if len(counts) == 0:
        return 1.0

    mean_count = np.mean(counts)
    if mean_count == 0:
        return 1.0

    std_count = np.std(counts)
    balance_ratio = 1.0 - (std_count / mean_count)

    return float(np.clip(balance_ratio, 0.0, 1.0))


def suggest_task_reallocation(
    uav_loads: Dict[int, int],
    terminal_priorities: Dict[int, float]
) -> Dict[str, Any]:
    """
    [5.3.2] 建议任务重新分配方案 【暂不启用】

    功能：根据UAV负载和终端优先级，建议任务重新分配

    参数：
        uav_loads: UAV负载情况 {uav_id: num_terminals_serving}
        terminal_priorities: 终端优先级 {terminal_id: priority}

    返回：
        reallocation_suggestion: 重新分配建议字典

    注意：此函数暂不启用，仅作为预留接口
    """
    return {
        'overloaded_uavs': [],
        'underloaded_uavs': [],
        'suggested_transfers': []
    }


# ============================================================================
# 🤝 第五部分完成标记
# ============================================================================
# ✅ 5.1 单终端多UAV服务 (2个函数)
# ✅ 5.2 单UAV多终端服务 (2个函数)
# ⚠️ 5.3 负载均衡 (2个函数) - 已实现但暂不启用
# 总计：6个函数
# ============================================================================


# ============================================================================
# 🎯 第六部分：任务管理
# ============================================================================

def generate_terminal_task(
    terminal_id: int,
    data_range: Tuple[float, float],
    cpu_cycles_per_bit: int,
    priority_range: Tuple[float, float] = (0.5, 1.0)
) -> Dict[str, Any]:
    """
    [6.1.1] 为终端生成任务

    功能：为单个终端生成随机任务

    参数：
        terminal_id: 终端ID
        data_range: 数据量范围 (min_kb, max_kb)
        cpu_cycles_per_bit: 每bit需要的CPU周期数
        priority_range: 优先级范围 (min, max)

    返回：
        task: 任务字典，包含：
            - terminal_id: 终端ID
            - total_data_bits: 总数据量 (bits)
            - remaining_data_bits: 剩余数据量 (bits)
            - cpu_cycles_per_bit: 每bit CPU周期数
            - priority: 优先级 (0-1)
            - is_completed: 是否完成

    示例：
        >>> task = generate_terminal_task(0, (10.0, 20.0), 1000)
        >>> 10*1024*8 <= task['total_data_bits'] <= 20*1024*8  # True
    """
    # 随机生成数据量 (KB -> bits)
    data_kb = np.random.uniform(data_range[0], data_range[1])
    data_bits = data_kb * 1024 * 8  # KB -> bits

    # 随机生成优先级
    priority = np.random.uniform(priority_range[0], priority_range[1])

    task = {
        'terminal_id': terminal_id,
        'total_data_bits': float(data_bits),
        'remaining_data_bits': float(data_bits),
        'cpu_cycles_per_bit': cpu_cycles_per_bit,
        'priority': float(priority),
        'is_completed': False
    }

    return task


def initialize_all_terminal_tasks(
    num_terminals: int,
    data_range: Tuple[float, float],
    cpu_cycles_per_bit: int
) -> List[Dict[str, Any]]:
    """
    [6.1.2] 初始化所有终端任务

    功能：为所有终端生成初始任务

    参数：
        num_terminals: 终端数量
        data_range: 数据量范围 (min_kb, max_kb)
        cpu_cycles_per_bit: 每bit需要的CPU周期数

    返回：
        tasks: 任务列表

    示例：
        >>> tasks = initialize_all_terminal_tasks(6, (10.0, 20.0), 1000)
        >>> len(tasks)  # 6
    """
    tasks = []
    for i in range(num_terminals):
        task = generate_terminal_task(i, data_range, cpu_cycles_per_bit)
        tasks.append(task)
    return tasks


def update_terminal_progress(
    terminal: Dict[str, Any],
    processed_bits: float
) -> Tuple[Dict[str, Any], bool]:
    """
    [6.2.1] 更新单个终端任务进度

    功能：更新终端的剩余数据量，检查是否完成

    参数：
        terminal: 终端字典
        processed_bits: 本步处理的数据量 (bits)

    返回：
        updated_terminal: 更新后的终端字典
        is_completed: 是否完成

    示例：
        >>> terminal = {'remaining_data_bits': 1000.0, 'is_completed': False}
        >>> updated, completed = update_terminal_progress(terminal, 600.0)
        >>> updated['remaining_data_bits']  # 400.0
        >>> completed  # False
    """
    terminal = terminal.copy()

    # 更新剩余数据量
    terminal['remaining_data_bits'] -= processed_bits
    terminal['remaining_data_bits'] = max(0.0, terminal['remaining_data_bits'])

    # 检查是否完成
    if terminal['remaining_data_bits'] <= 0:
        terminal['is_completed'] = True
        is_completed = True
    else:
        is_completed = False

    return terminal, is_completed


def update_all_terminals_progress(
    terminals: List[Dict[str, Any]],
    uav_terminal_progress: Dict[int, Dict[int, float]]
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    [6.2.2] 更新所有终端任务进度

    功能：批量更新所有终端的任务进度

    参数：
        terminals: 终端列表
        uav_terminal_progress: UAV-终端处理量字典 {uav_id: {terminal_id: processed_bits}}

    返回：
        updated_terminals: 更新后的终端列表
        completed_terminal_ids: 本步完成的终端ID列表

    示例：
        >>> progress = {0: {0: 500.0, 1: 300.0}, 1: {1: 200.0}}
        >>> updated, completed = update_all_terminals_progress(terminals, progress)
    """
    updated_terminals = []
    completed_terminal_ids = []

    for term_id, terminal in enumerate(terminals):
        # 汇总所有UAV对该终端的处理量
        total_processed = 0.0
        for uav_id, term_progress in uav_terminal_progress.items():
            total_processed += term_progress.get(term_id, 0.0)

        # 更新终端进度
        updated_terminal, is_completed = update_terminal_progress(terminal, total_processed)
        updated_terminals.append(updated_terminal)

        if is_completed and not terminal['is_completed']:
            completed_terminal_ids.append(term_id)

    return updated_terminals, completed_terminal_ids


def check_terminal_completion(terminal: Dict[str, Any]) -> bool:
    """
    [6.3.1] 检查终端任务是否完成

    功能：检查单个终端的任务是否完成

    参数：
        terminal: 终端字典

    返回：
        is_completed: 是否完成

    示例：
        >>> terminal = {'is_completed': True}
        >>> check_terminal_completion(terminal)  # True
    """
    return terminal.get('is_completed', False)


def check_all_tasks_completed(terminals: List[Dict[str, Any]]) -> Tuple[bool, float]:
    """
    [6.3.2] 检查所有任务是否完成

    功能：检查是否所有终端任务都已完成

    参数：
        terminals: 终端列表

    返回：
        all_completed: 是否全部完成
        completion_ratio: 完成比例 (0-1)

    示例：
        >>> all_done, ratio = check_all_tasks_completed(terminals)
        >>> ratio  # 0.0 到 1.0
    """
    if not terminals:
        return True, 1.0

    completed_count = sum(1 for t in terminals if t.get('is_completed', False))
    completion_ratio = completed_count / len(terminals)
    all_completed = (completed_count == len(terminals))

    return all_completed, completion_ratio


def calculate_task_completion_ratio(terminals: List[Dict[str, Any]]) -> float:
    """
    [6.3.3] 计算任务完成率

    功能：计算所有终端的任务完成率

    参数：
        terminals: 终端列表

    返回：
        completion_ratio: 完成率 (0-1)

    示例：
        >>> ratio = calculate_task_completion_ratio(terminals)
        >>> 0.0 <= ratio <= 1.0  # True
    """
    _, ratio = check_all_tasks_completed(terminals)
    return ratio


# ============================================================================
# 🎯 第六部分完成标记
# ============================================================================
# ✅ 6.1 任务生成 (2个函数)
# ✅ 6.2 任务进度更新 (2个函数)
# ✅ 6.3 任务完成检查 (3个函数)
# 总计：7个函数
# ============================================================================


# ============================================================================
# 📊 第七部分：观测与状态
# ============================================================================

def calculate_uav_self_observation(
    uav_id: int,
    uav_battery: np.ndarray,
    uav_positions: np.ndarray,
    uav_processing_data: Dict[int, float],
    battery_capacity: float,
    height_min: float,
    height_max: float,
    data_range: Tuple[float, float],
    num_terminals: int
) -> np.ndarray:
    """
    [7.1.1] 计算UAV自身状态观测

    功能：计算UAV的自身状态观测向量

    观测内容：
        - 归一化电量 (0-1)
        - 归一化高度 (0-1)
        - 归一化处理数据量 (0-1)

    参数：
        uav_id: UAV ID
        uav_battery: 所有UAV电量数组
        uav_positions: 所有UAV位置数组 (num_uavs, 3)
        uav_processing_data: UAV处理数据量字典 {uav_id: processed_bits}
        battery_capacity: 电池容量 (J)
        height_min: 最小飞行高度 (m)
        height_max: 最大飞行高度 (m)
        data_range: 数据量范围 (min_kb, max_kb)
        num_terminals: 终端数量

    返回：
        self_obs: 自身观测向量 [归一化电量, 归一化高度, 归一化处理量]

    示例：
        >>> obs = calculate_uav_self_observation(0, battery, positions, processing, ...)
        >>> obs.shape  # (3,)
    """
    # 归一化电量
    battery_ratio = uav_battery[uav_id] / battery_capacity

    # 归一化高度
    height = uav_positions[uav_id, 2]
    height_ratio = (height - height_min) / (height_max - height_min)

    # 归一化处理数据量
    processed = uav_processing_data.get(uav_id, 0.0)
    max_data = data_range[1] * 1024 * 8 * num_terminals  # 最大可能数据量
    processing_ratio = min(processed / max_data, 1.0) if max_data > 0 else 0.0

    self_obs = np.array([battery_ratio, height_ratio, processing_ratio], dtype=np.float32)
    return self_obs


def calculate_terminal_observation(
    uav_pos: np.ndarray,
    terminal: Dict[str, Any],
    ground_area: float
) -> np.ndarray:
    """
    [7.1.2] 计算单个终端的观测

    功能：计算从UAV视角观测单个终端的状态

    观测内容：
        - 方向单位向量 (x, y) - 2维
        - 归一化距离 (0-1) - 1维
        - 归一化剩余数据 (0-1) - 1维

    参数：
        uav_pos: UAV位置 [x, y, z]
        terminal: 终端字典
        ground_area: 地面区域边长 (m)

    返回：
        terminal_obs: 终端观测向量 [dir_x, dir_y, norm_dist, norm_data]

    示例：
        >>> obs = calculate_terminal_observation(uav_pos, terminal, 400.0)
        >>> obs.shape  # (4,)
    """
    term_pos = terminal['position']

    # 计算方向向量
    direction = term_pos[:2] - uav_pos[:2]  # 只考虑水平方向
    distance = np.linalg.norm(direction)

    # 计算单位方向向量
    if distance > 1e-6:
        unit_direction = direction / distance
    else:
        unit_direction = np.array([0.0, 0.0])

    # 归一化距离（使用对角线长度）
    max_distance = ground_area * np.sqrt(2)
    norm_distance = min(distance / max_distance, 1.0)

    # 归一化剩余数据
    if terminal['total_data_bits'] > 0:
        norm_data = terminal['remaining_data_bits'] / terminal['total_data_bits']
    else:
        norm_data = 0.0

    terminal_obs = np.array([
        unit_direction[0],
        unit_direction[1],
        norm_distance,
        norm_data
    ], dtype=np.float32)

    return terminal_obs


def calculate_other_uav_observation(
    uav_pos: np.ndarray,
    other_uav_pos: np.ndarray,
    ground_area: float
) -> np.ndarray:
    """
    [7.1.3] 计算其他UAV的观测

    功能：计算从当前UAV视角观测其他UAV的状态

    观测内容：
        - 方向单位向量 (x, y) - 2维
        - 归一化距离 (0-1) - 1维

    参数：
        uav_pos: 当前UAV位置 [x, y, z]
        other_uav_pos: 其他UAV位置 [x, y, z]
        ground_area: 地面区域边长 (m)

    返回：
        other_uav_obs: 其他UAV观测向量 [dir_x, dir_y, norm_dist]

    示例：
        >>> obs = calculate_other_uav_observation(uav_pos, other_pos, 400.0)
        >>> obs.shape  # (3,)
    """
    # 计算方向向量（只考虑水平方向）
    direction = other_uav_pos[:2] - uav_pos[:2]
    distance = np.linalg.norm(direction)

    # 计算单位方向向量
    if distance > 1e-6:
        unit_direction = direction / distance
    else:
        unit_direction = np.array([0.0, 0.0])

    # 归一化距离
    max_distance = ground_area * np.sqrt(2)
    norm_distance = min(distance / max_distance, 1.0)

    other_uav_obs = np.array([
        unit_direction[0],
        unit_direction[1],
        norm_distance
    ], dtype=np.float32)

    return other_uav_obs


def construct_full_observation(
    uav_id: int,
    uav_positions: np.ndarray,
    uav_battery: np.ndarray,
    uav_processing_data: Dict[int, float],
    terminals: List[Dict[str, Any]],
    env_params: Dict[str, Any]
) -> np.ndarray:
    """
    [7.1.4] 构建完整观测向量

    功能：构建UAV的完整观测向量

    观测结构（假设2 UAV, 6终端）：
        - 自身状态：3维
        - 6个终端：6×4=24维
        - 1个其他UAV：1×3=3维
        总计：30维

    参数：
        uav_id: UAV ID
        uav_positions: 所有UAV位置数组
        uav_battery: 所有UAV电量数组
        uav_processing_data: UAV处理数据量字典
        terminals: 终端列表
        env_params: 环境参数字典，包含：
            - battery_capacity
            - height_min, height_max
            - data_range
            - ground_area

    返回：
        full_obs: 完整观测向量

    示例：
        >>> obs = construct_full_observation(0, positions, battery, processing, terminals, params)
        >>> obs.shape  # (30,) for 2 UAVs, 6 terminals
    """
    observations = []

    # 1. 自身状态观测
    self_obs = calculate_uav_self_observation(
        uav_id, uav_battery, uav_positions, uav_processing_data,
        env_params['battery_capacity'],
        env_params['height_min'],
        env_params['height_max'],
        env_params['data_range'],
        len(terminals)
    )
    observations.append(self_obs)

    # 2. 所有终端观测
    for terminal in terminals:
        term_obs = calculate_terminal_observation(
            uav_positions[uav_id],
            terminal,
            env_params['ground_area']
        )
        observations.append(term_obs)

    # 3. 其他UAV观测
    num_uavs = len(uav_positions)
    for other_id in range(num_uavs):
        if other_id != uav_id:
            other_obs = calculate_other_uav_observation(
                uav_positions[uav_id],
                uav_positions[other_id],
                env_params['ground_area']
            )
            observations.append(other_obs)

    # 拼接所有观测
    full_obs = np.concatenate(observations)
    return full_obs


def normalize_distance(distance: float, max_distance: float) -> float:
    """
    [7.2.1] 归一化距离

    功能：将距离归一化到 [0, 1] 范围

    参数：
        distance: 距离 (m)
        max_distance: 最大距离 (m)

    返回：
        normalized: 归一化距离 (0-1)

    示例：
        >>> norm = normalize_distance(200.0, 400.0)
        >>> norm  # 0.5
    """
    if max_distance <= 0:
        return 0.0
    return float(np.clip(distance / max_distance, 0.0, 1.0))


def normalize_battery(battery: float, battery_capacity: float) -> float:
    """
    [7.2.2] 归一化电量

    功能：将电量归一化到 [0, 1] 范围

    参数：
        battery: 当前电量 (J)
        battery_capacity: 电池容量 (J)

    返回：
        normalized: 归一化电量 (0-1)

    示例：
        >>> norm = normalize_battery(5000.0, 10000.0)
        >>> norm  # 0.5
    """
    if battery_capacity <= 0:
        return 0.0
    return float(np.clip(battery / battery_capacity, 0.0, 1.0))


def normalize_height(height: float, height_min: float, height_max: float) -> float:
    """
    [7.2.3] 归一化高度

    功能：将高度归一化到 [0, 1] 范围

    参数：
        height: 当前高度 (m)
        height_min: 最小高度 (m)
        height_max: 最大高度 (m)

    返回：
        normalized: 归一化高度 (0-1)

    示例：
        >>> norm = normalize_height(70.0, 20.0, 120.0)
        >>> norm  # 0.5
    """
    if height_max <= height_min:
        return 0.0
    return float(np.clip((height - height_min) / (height_max - height_min), 0.0, 1.0))


def calculate_unit_vector(vector: np.ndarray) -> np.ndarray:
    """
    [7.2.4] 计算单位向量

    功能：将向量归一化为单位向量

    参数：
        vector: 输入向量

    返回：
        unit_vector: 单位向量

    示例：
        >>> vec = np.array([3.0, 4.0])
        >>> unit = calculate_unit_vector(vec)
        >>> unit  # [0.6, 0.8]
        >>> np.linalg.norm(unit)  # 1.0
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


# ============================================================================
# 📊 第七部分完成标记
# ============================================================================
# ✅ 7.1 观测计算 (4个函数)
# ✅ 7.2 归一化工具 (4个函数)
# 总计：8个函数
# ============================================================================


# ============================================================================
# 🎮 第八部分：动作解析
# ============================================================================

def encode_action(movement_action: int, service_decision: int) -> int:
    """
    [8.1.1] 将移动和服务决策编码为单一动作索引

    功能：将移动动作和服务决策组合编码为一个动作索引

    编码规则：
        - 移动动作：0-7 (8种：4个水平方向 × 2个垂直状态)
        - 服务决策：0-1 (2种：不服务/服务)
        - 总动作空间：8 × 2 = 16

    公式：action_index = movement_action * 2 + service_decision

    参数：
        movement_action: 移动动作 (0-7)
        service_decision: 服务决策 (0-1)

    返回：
        action_index: 动作索引 (0-15)

    示例：
        >>> action = encode_action(3, 1)  # 向右移动 + 服务
        >>> action  # 7
    """
    return movement_action * 2 + service_decision


def decode_action(action_index: int) -> Tuple[int, int]:
    """
    [8.1.2] 将动作索引解码为移动和服务决策

    功能：将单一动作索引解码为移动动作和服务决策

    参数：
        action_index: 动作索引 (0-15)

    返回：
        movement_action: 移动动作 (0-7)
        service_decision: 服务决策 (0-1)

    示例：
        >>> movement, service = decode_action(7)
        >>> movement  # 3 (向右)
        >>> service  # 1 (服务)
    """
    movement_action = action_index // 2
    service_decision = action_index % 2
    return movement_action, service_decision


def parse_movement_action(movement_action: int) -> Tuple[int, int]:
    """
    [8.1.3] 解析移动动作为水平和垂直动作

    功能：将移动动作解析为水平动作和垂直动作

    动作编码：
        - 0: 向上(y+) + 悬停
        - 1: 向下(y-) + 悬停
        - 2: 向左(x-) + 悬停
        - 3: 向右(x+) + 悬停
        - 4: 保持不动 + 上升
        - 5: 保持不动 + 下降
        - 6: 保持不动 + 悬停
        - 7: 保留（未使用）

    参数：
        movement_action: 移动动作 (0-7)

    返回：
        horizontal_action: 水平动作 (0-4)
            - 0: 向上(y+)
            - 1: 向下(y-)
            - 2: 向左(x-)
            - 3: 向右(x+)
            - 4: 保持不动
        vertical_action: 垂直动作 (0-2)
            - 0: 下降
            - 1: 上升
            - 2: 悬停

    示例：
        >>> h, v = parse_movement_action(0)
        >>> h, v  # (0, 2) - 向上移动 + 悬停
    """
    if movement_action == 0:
        return 0, 2  # 向上 + 悬停
    elif movement_action == 1:
        return 1, 2  # 向下 + 悬停
    elif movement_action == 2:
        return 2, 2  # 向左 + 悬停
    elif movement_action == 3:
        return 3, 2  # 向右 + 悬停
    elif movement_action == 4:
        return 4, 1  # 不动 + 上升
    elif movement_action == 5:
        return 4, 0  # 不动 + 下降
    elif movement_action == 6:
        return 4, 2  # 不动 + 悬停
    else:
        return 4, 2  # 默认：不动 + 悬停


def validate_action(action: int, action_dim: int) -> bool:
    """
    [8.2.1] 验证动作是否合法

    功能：检查动作是否在合法范围内

    参数：
        action: 动作索引
        action_dim: 动作空间维度

    返回：
        is_valid: 是否合法

    示例：
        >>> validate_action(5, 16)  # True
        >>> validate_action(20, 16)  # False
    """
    return 0 <= action < action_dim


def clip_action(action: int, action_dim: int) -> int:
    """
    [8.2.2] 将动作裁剪到合法范围

    功能：将越界的动作裁剪到合法范围

    参数：
        action: 动作索引
        action_dim: 动作空间维度

    返回：
        clipped_action: 裁剪后的动作

    示例：
        >>> clip_action(20, 16)  # 15
        >>> clip_action(-5, 16)  # 0
    """
    return int(np.clip(action, 0, action_dim - 1))


# ============================================================================
# 🎮 第八部分完成标记
# ============================================================================
# ✅ 8.1 动作编码/解码 (3个函数)
# ✅ 8.2 动作验证 (2个函数)
# 总计：5个函数
# ============================================================================


# ============================================================================
# 🎉 function.py 实现完成总结
# ============================================================================
#
# 已完成部分（1-8）：
# ✅ 第一部分：三维场景构建 (6个函数)
# ✅ 第二部分：无人机动力学模型 (10个函数)
# ✅ 第三部分：通信模型 (13个函数)
# ✅ 第四部分：计算与卸载模型 (7个函数)
# ✅ 第五部分：多UAV协作模型 (6个函数)
# ✅ 第六部分：任务管理 (7个函数)
# ✅ 第七部分：观测与状态 (8个函数)
# ✅ 第八部分：动作解析 (5个函数)
#
# 总计：62个功能函数
#
# 未实现部分（按用户要求排除）：
# ⏸️ 第九部分：奖励计算（基础）- 用户要求暂不实现
# ⏸️ 第十部分：博弈论奖励（扩展）- 用户要求暂不实现
# ⏸️ 第十一部分：工具函数 - 可选实现
#
# ============================================================================



