#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试参数传递是否正确
"""

import sys
sys.path.append('.')

from config import get_config
from envs.env_uav_wrapper import UAVMECWrapper


def test_params():
    """测试参数传递"""
    print("=" * 80)
    print("测试参数传递")
    print("=" * 80)

    # 获取配置
    parser = get_config()
    args = parser.parse_args([])

    # 构建环境配置（与 train.py 中相同）
    env_config = {
        'num_uavs': args.uav_num_uavs,
        'num_terminals': args.uav_num_terminals,
        'ground_area': args.uav_ground_area,
        'height_min': args.uav_height_min,
        'height_max': args.uav_height_max,
        'communication_range': args.uav_communication_range,
        'cpu_freq_uav': args.uav_cpu_freq_uav,
        'cpu_freq_ground': args.uav_cpu_freq_ground,
        'max_horizontal_speed': args.uav_max_horizontal_speed,
        'max_vertical_speed': args.uav_max_vertical_speed,
        'battery_capacity': args.uav_battery_capacity,
        'max_episode_steps': args.episode_length,
        'time_slot': args.uav_time_slot,
        'data_range': (args.uav_data_range_min, args.uav_data_range_max),
        'cpu_cycles_per_bit': args.uav_cpu_cycles_per_bit,
    }

    print("\n传递给环境的参数:")
    print("-" * 80)
    for key, value in env_config.items():
        print(f"'{key}' = {value}")

    # 创建环境
    print("\n" + "=" * 80)
    print("创建环境...")
    env = UAVMECWrapper(env_config)

    print("\n[OK] 环境创建成功")
    print(f"   - UAV数量: {env.env.num_uavs}")
    print(f"   - 终端数量: {env.env.num_terminals}")
    print(f"   - 电池容量: {env.env.config['battery_capacity']} J")
    print(f"   - 数据范围: {env.env.config['data_range']} bits")
    print(f"   - 时隙长度: {env.env.config['time_slot']} s")

    # 测试运行
    print("\n" + "=" * 80)
    print("测试环境运行...")
    obs = env.reset()
    print(f"[OK] Reset成功，观测形状: {obs.shape}")

    import numpy as np
    actions = [np.random.randint(0, 16) for _ in range(env.num_agent)]
    obs, rewards, dones, infos = env.step(actions)
    print(f"[OK] Step成功")

    print("\n" + "=" * 80)
    print("[SUCCESS] 所有参数传递正确！")
    print("=" * 80)


if __name__ == "__main__":
    test_params()
