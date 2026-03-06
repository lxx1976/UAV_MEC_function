"""
# @Time    : 2021/7/2 5:22 下午
# @Author  : hezhiqiang
# @Email   : tinyzqh@163.com
# @File    : env_discrete.py
"""

import gym
from gym import spaces
import numpy as np
from envs.env_core import EnvCore


class DiscreteActionEnv(object):
    """
    离散动作环境包装器
    Wrapper for discrete action environment using MultiDiscrete action space.

    动作空间：MultiDiscrete([7, 2, 4])
    - 维度0: 移动动作 (0-6)
        0: 悬停
        1: 向上移动
        2: 向下移动
        3: 向左移动
        4: 向右移动
        5: 向前移动
        6: 向后移动
    - 维度1: 服务决策 (0-1)
        0: 不提供服务
        1: 提供服务
    - 维度2: 服务终端数量 (0-3)
        0: 服务0个终端
        1: 服务1个终端
        2: 服务2个终端
        3: 服务3个终端
    """

    def __init__(self):
        self.env = EnvCore(use_discrete_action=True)  # 使用离散动作空间
        self.num_agent = self.env.agent_num

        self.signal_obs_dim = self.env.obs_dim

        # 离散动作空间：[移动(7), 服务决策(2), 服务数量(4)]
        self.discrete_action_input = True
        self.movable = True

        # 配置空间
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0

        for agent in range(self.num_agent):
            # 离散动作空间：MultiDiscrete
            # [移动动作(0-6), 服务决策(0-1), 服务终端数(0-3)]
            discrete_action_space = spaces.MultiDiscrete([7, 2, 4])
            self.action_space.append(discrete_action_space)

            # 观测空间
            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.signal_obs_dim,),
                    dtype=np.float32,
                )
            )

        # 共享观测空间（用于中心化Critic）
        self.share_observation_space = [
            spaces.Box(
                low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32
            )
            for _ in range(self.num_agent)
        ]

    def step(self, actions):
        """
        执行动作

        参数:
            actions: 离散动作数组
                shape = (n_threads, n_agents, 3) for MultiDiscrete
                或 shape = (n_agents, 3) for single thread
                每个动作是 [移动(0-6), 服务(0-1), 数量(0-3)]

        返回:
            obs: 观测
            rewards: 奖励
            dones: 完成标志
            infos: 信息字典
        """
        # 确保actions是正确的格式
        actions = np.asarray(actions, dtype=np.int32)

        # 调用环境的step函数
        results = self.env.step(actions)
        obs, rews, dones, infos = results

        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        """重置环境"""
        obs = self.env.reset()
        return np.stack(obs)

    def close(self):
        """关闭环境"""
        pass

    def render(self, mode="rgb_array"):
        """渲染环境"""
        pass

    def seed(self, seed):
        """设置随机种子"""
        np.random.seed(seed)


class MultiDiscrete:
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """

    def __init__(self, array_of_param_array):
        super().__init__()
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):
        """Returns a array with one sample from each discrete action space"""
        # For each row: round(random .* (max - min) + min, 0)
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.0), random_array) + self.low)]

    def contains(self, x):
        return (
            len(x) == self.num_discrete_space
            and (np.array(x) >= self.low).all()
            and (np.array(x) <= self.high).all()
        )

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


if __name__ == "__main__":
    DiscreteActionEnv().step(actions=None)
