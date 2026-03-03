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
    对于离散动作环境的封装
    Wrapper for discrete action environment.
    """

    def __init__(self):
        self.env = EnvCore()
        self.num_agent = self.env.agent_num

        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim

        # 如果为 False，则动作是一个 N 维的 one-hot 向量。if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False

        self.movable = True  

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0  #所有智能体的单独观测维度相加之后的总长度。实现“全局共享观测”
        total_action_space = []
        for agent_idx in range(self.num_agent):  #在enc_core.py中定义,开始为每一个智能体分配动作空间等等
            # physical action space
            u_action_space = spaces.Discrete(self.signal_action_dim)  # 5个离散的动作，动作维度在env_core.py中定义

            # if self.movable:
            total_action_space.append(u_action_space)
            '''
            现在的写法：
            只把每个 agent 的 action_space 分开存，不关心“联合空间”。

            注释掉的写法：
            其实是支持把所有 agent 的动作空间合成为一个总的动作空间，可以简化多智能体训练时的动作分布建模，适用于某些联合建模的算法（比如单网络输出所有 agent 行为）。
            '''
            # total action space
            # if len(total_action_space) > 1:
            #     # all action spaces are discrete, so simplify to MultiDiscrete action space
            #     if all(
            #         [
            #             isinstance(act_space, spaces.Discrete)
            #             for act_space in total_action_space
            #         ]
            #     ):
            #         act_space = MultiDiscrete(
            #             [[0, act_space.n - 1] for act_space in total_action_space]
            #         )
            #     else:
            #         act_space = spaces.Tuple(total_action_space)
            # self.action_space.append(act_space)
            # else:
            self.action_space.append(total_action_space[agent_idx])

            # observation space
            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.signal_obs_dim,),
                    dtype=np.float32,
                )
            )  # [-inf,inf]

        self.share_observation_space = [        #共享观测空间
            spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agent)
        ]

    def step(self, actions):
        """
        输入actions维度假设：
        # actions shape = (5, 2, 5)
        # 5个线程的环境，里面有2个智能体，每个智能体的动作是一个one_hot的5维编码
        Input actions dimension assumption:
        # actions shape = (5, 2, 5)
        # 5 threads of the environment, with 2 intelligent agents inside, and each intelligent agent's action is a 5-dimensional one_hot encoding
        """

        results = self.env.step(actions)
        obs, rews, dones, infos = results
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obs = self.env.reset()#这里调用了envcore的reset功能
        return np.stack(obs)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        pass


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
