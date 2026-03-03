"""
UAV MEC Environment Wrapper
将 UAVMECEnvironment 适配到 MAPPO 训练框架
"""

import gym
from gym import spaces
import numpy as np
from .env_uav_mec_clean import UAVMECEnvironment


class UAVMECWrapper(object):
    """
    UAV MEC 环境的包装器
    使其兼容 MAPPO 训练框架的接口要求
    """

    def __init__(self, config=None):
        """
        初始化环境

        Args:
            config: 环境配置字典，如果为 None 则使用默认配置
        """
        # 创建底层环境
        self.env = UAVMECEnvironment(config)

        # 获取智能体数量
        self.num_agent = self.env.num_uavs

        # 获取观测和动作维度
        self.signal_obs_dim = self.env.observation_space
        self.signal_action_dim = self.env.action_space

        # 动作输入格式（True: 数字 0-15, False: one-hot 向量）
        self.discrete_action_input = True

        # 配置空间
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        # 计算共享观测维度（所有智能体观测拼接）
        share_obs_dim = self.signal_obs_dim * self.num_agent

        # 为每个智能体配置空间
        for agent_idx in range(self.num_agent):
            # 动作空间：离散动作 0-15
            self.action_space.append(spaces.Discrete(self.signal_action_dim))

            # 观测空间：连续观测
            self.observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.signal_obs_dim,),
                    dtype=np.float32,
                )
            )

        # 共享观测空间（用于 centralized critic）
        self.share_observation_space = [
            spaces.Box(
                low=-np.inf,
                high=+np.inf,
                shape=(share_obs_dim,),
                dtype=np.float32
            )
            for _ in range(self.num_agent)
        ]

    def step(self, actions):
        """
        执行一步动作

        Args:
            actions: 动作列表或数组
                - 如果是 one-hot 编码，需要转换为数字
                - 如果已经是数字，直接使用

        Returns:
            observations: 观测数组 (num_agents, obs_dim)
            rewards: 奖励数组 (num_agents,)
            dones: 完成标志数组 (num_agents,)
            infos: 信息字典列表
        """
        # 处理动作格式
        if isinstance(actions, np.ndarray):
            if len(actions.shape) > 1 and actions.shape[-1] > 1:
                # one-hot 编码，转换为数字
                actions = [np.argmax(a) for a in actions]
            else:
                # 已经是数字格式
                actions = actions.flatten().tolist()

        # 调用底层环境
        observations, rewards, dones, infos = self.env.step(actions)

        # 转换为 numpy 数组
        observations = np.stack(observations)
        rewards = np.array(rewards)
        dones = np.array(dones)

        return observations, rewards, dones, infos

    def reset(self):
        """
        重置环境

        Returns:
            observations: 初始观测数组 (num_agents, obs_dim)
        """
        observations = self.env.reset()
        return np.stack(observations)

    def close(self):
        """关闭环境"""
        pass

    def render(self, mode="rgb_array"):
        """渲染环境（暂未实现）"""
        pass

    def seed(self, seed):
        """设置随机种子"""
        np.random.seed(seed)


if __name__ == "__main__":
    # 测试 wrapper
    print("Testing UAV MEC Wrapper...")

    env = UAVMECWrapper()
    print(f"Number of agents: {env.num_agent}")
    print(f"Observation dim: {env.signal_obs_dim}")
    print(f"Action dim: {env.signal_action_dim}")

    # 测试 reset
    obs = env.reset()
    print(f"Reset observation shape: {obs.shape}")

    # 测试 step
    actions = [np.random.randint(0, 16) for _ in range(env.num_agent)]
    obs, rewards, dones, infos = env.step(actions)
    print(f"Step observation shape: {obs.shape}")
    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}")

    print("Wrapper test passed!")
