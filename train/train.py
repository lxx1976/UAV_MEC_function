"""
# @Time    : 2021/6/30 10:07 下午
# @Author  : hezhiqiang
# @Email   : tinyzqh@163.com
# @File    : train.py
"""

# !/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "light_mappo-main"))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from config import get_config
from envs.env_wrappers import DummyVecEnv

"""Train script for MPEs."""


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # 根据 env_name 选择环境
            if all_args.env_name == "MyEnv":
                # 原始示例环境
                from envs.env_discrete import DiscreteActionEnv
                env = DiscreteActionEnv()
            elif all_args.env_name == "UAVMECEnv":
                # UAV MEC 环境
                from envs.env_uav_wrapper import UAVMECWrapper

                # 构建环境配置
                env_config = {
                    'num_uavs': all_args.uav_num_uavs,
                    'num_terminals': all_args.uav_num_terminals,
                    'ground_area': all_args.uav_ground_area,
                    'height_min': all_args.uav_height_min,
                    'height_max': all_args.uav_height_max,
                    'communication_range': all_args.uav_communication_range,
                    'cpu_freq_uav': all_args.uav_cpu_freq_uav,
                    'cpu_freq_ground': all_args.uav_cpu_freq_ground,
                    'max_horizontal_speed': all_args.uav_max_horizontal_speed,
                    'max_vertical_speed': all_args.uav_max_vertical_speed,
                    'battery_capacity': all_args.uav_battery_capacity,
                    'max_episode_steps': all_args.episode_length,
                    'time_slot': all_args.uav_time_slot,
                    'data_range': (all_args.uav_data_range_min, all_args.uav_data_range_max),
                    'cpu_cycles_per_bit': all_args.uav_cpu_cycles_per_bit,
                }
                env = UAVMECWrapper(env_config)
            else:
                raise ValueError(f"Unknown environment: {all_args.env_name}")

            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])#并行训练一共五轮在config中有


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # 根据 env_name 选择环境（与训练环境保持一致）
            if all_args.env_name == "MyEnv":
                # 原始示例环境
                from envs.env_discrete import DiscreteActionEnv
                env = DiscreteActionEnv()
            elif all_args.env_name == "UAVMECEnv":
                # UAV MEC 环境
                from envs.env_uav_wrapper import UAVMECWrapper

                # 构建环境配置
                env_config = {
                    'num_uavs': all_args.uav_num_uavs,
                    'num_terminals': all_args.uav_num_terminals,
                    'ground_area': all_args.uav_ground_area,
                    'height_min': all_args.uav_height_min,
                    'height_max': all_args.uav_height_max,
                    'communication_range': all_args.uav_communication_range,
                    'cpu_freq_uav': all_args.uav_cpu_freq_uav,
                    'cpu_freq_ground': all_args.uav_cpu_freq_ground,
                    'max_horizontal_speed': all_args.uav_max_horizontal_speed,
                    'max_vertical_speed': all_args.uav_max_vertical_speed,
                    'battery_capacity': all_args.uav_battery_capacity,
                    'max_episode_steps': all_args.episode_length,
                    'time_slot': all_args.uav_time_slot,
                    'data_range': (all_args.uav_data_range_min, all_args.uav_data_range_max),
                    'cpu_cycles_per_bit': all_args.uav_cpu_cycles_per_bit,
                }
                env = UAVMECWrapper(env_config)
            else:
                raise ValueError(f"Unknown environment: {all_args.env_name}")

            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="MyEnv", help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (
            all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False
        ), "check recurrent policy!"
    else:
        raise NotImplementedError

    assert (
        all_args.share_policy == True and all_args.scenario_name == "simple_speaker_listener"
    ) == False, "The simple_speaker_listener scenario can not use shared policy. Please check the config.py."

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():#为了避免和历史进程中有重复，所以每次都会检查从run1开始存不存在这个文件，如果有的话就创建新的，但其实就是创建了一个runX文件夹。
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        from runner.shared.env_runner import EnvRunner as Runner
    else:
        from runner.separated.env_runner import EnvRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
