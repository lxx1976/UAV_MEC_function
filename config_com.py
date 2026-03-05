import argparse


def get_config():
    """
    The configuration parser for common hyperparameters of all environment.
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["rmappo", "mappo", "rmappg", "mappg", "trpo"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch
        --cuda
            by default True, will use GPU to train; or else will use CPU;
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.

    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or else will use concatenated local obs.

    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer.

    Network parameters:
        --share_policy
            by default True, all agents will share the same network; set to make training agents use different policies.
        --use_centralized_V
            by default True, use centralized training mode; or else will decentralized training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards.
        --use_valuenorm
            by default True, use running mean and std to normalize rewards.
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs.
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases. or else, will use xavier uniform inilialization.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers ( default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.

    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)

    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.
        --huber_delta <float>
            coefficient of huber loss.

    PPG parameters:
        --aux_epoch <int>
            number of auxiliary epochs. (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)

    Run parameters：
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate

    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.

    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.

    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.

    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    parser = argparse.ArgumentParser(
        description="onpolicy", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str, default="mappo", choices=["rmappo", "mappo"])

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="scheme_c_v3",
        help="an identifier to distinguish different experiment.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=True,
        help="use GPU to train; by default will use GPU unless --no_cuda is specified;",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        default=False,
        help="force use CPU even if GPU is available;",
    )
    parser.add_argument(
        "--cuda_deterministic",
        action="store_false",
        default=True,
        help="by default, make sure random seed effective. if set, bypass such function.",
    )
    parser.add_argument(
        "--n_training_threads",
        type=int,
        default=2,
        help="Number of torch threads for training",
    )
    parser.add_argument(
        "--n_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for training rollouts",
    )
    parser.add_argument(
        "--n_eval_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for evaluating rollouts",
    )
    parser.add_argument(
        "--n_render_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for rendering rollouts",
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=3000000,
        help="Number of environment steps to train (default: 10e6)",
    )
    parser.add_argument(
        "--user_name",
        type=str,
        default="marl",
        help="[for wandb usage], to specify user's name for simply collecting training data.",
    )

    # env parameters
    parser.add_argument("--env_name", type=str, default="UAVEdgeComputingSchemeC", help="specify the name of environment")
    parser.add_argument(
        "--use_obs_instead_of_state",
        action="store_true",
        default=False,
        help="Whether to use global state or concatenated obs",
    )

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int, default=1000, help="Max length for any episode")

    # network parameters
    parser.add_argument(
        "--share_policy",
        action="store_false",
        default=False,
        help="Whether agent share the same policy",
    )
    parser.add_argument(
        "--use_centralized_V",
        action="store_false",
        default=True,
        help="Whether to use centralized V function",
    )
    parser.add_argument(
        "--stacked_frames",
        type=int,
        default=1,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--use_stacked_frames",
        action="store_true",
        default=False,
        help="Whether to use stacked_frames",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--layer_N",
        type=int,
        default=3,
        help="Number of layers for actor/critic networks",
    )
    parser.add_argument("--use_ReLU", action="store_false", default=True, help="Whether to use ReLU")
    parser.add_argument(
        "--use_popart",
        action="store_true",
        default=False,
        help="by default False, use PopArt to normalize rewards.",
    )
    parser.add_argument(
        "--use_valuenorm",
        action="store_false",
        default=True,
        help="by default True, use running mean and std to normalize rewards.",
    )
    parser.add_argument(
        "--use_feature_normalization",
        action="store_false",
        default=True,
        help="Whether to apply layernorm to the inputs",
    )
    parser.add_argument(
        "--use_orthogonal",
        action="store_false",
        default=True,
        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases",
    )
    parser.add_argument("--gain", type=float, default=0.01, help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument(
        "--use_naive_recurrent_policy",
        action="store_true",
        default=False,
        help="Whether to use a naive recurrent policy",
    )
    parser.add_argument(
        "--use_recurrent_policy",
        action="store_false",
        default=False,
        help="use a recurrent policy",
    )
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument(
        "--data_chunk_length",
        type=int,
        default=10,
        help="Time length of chunks used to train a recurrent_policy",
    )

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 5e-4)")
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=0.001,
        help="critic learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--opti_eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=10, help="number of ppo epochs (default: 15)")
    parser.add_argument(
        "--use_clipped_value_loss",
        action="store_false",
        default=True,
        help="by default, clip loss value. If set, do not clip loss value.",
    )
    parser.add_argument(
        "--clip_param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--num_mini_batch",
        type=int,
        default=4,
        help="number of batches for ppo (default: 1)",
    )
    parser.add_argument(
        "--entropy_coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value_loss_coef",
        type=float,
        default=1,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--use_max_grad_norm",
        action="store_false",
        default=True,
        help="by default, use max norm of gradients. If set, do not use.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=10.0,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument(
        "--use_gae",
        action="store_false",
        default=True,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="gae lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--use_proper_time_limits",
        action="store_true",
        default=False,
        help="compute returns taking into account time limits",
    )
    parser.add_argument(
        "--use_huber_loss",
        action="store_false",
        default=True,
        help="by default, use huber loss. If set, do not use huber loss.",
    )
    parser.add_argument(
        "--use_value_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in value loss.",
    )
    parser.add_argument(
        "--use_policy_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in policy loss.",
    )
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument(
        "--use_linear_lr_decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the learning rate",
    )
    parser.add_argument(
        "--use_entropy_decay",
        action="store_true",
        default=False,
        help="use entropy coefficient decay to reduce exploration over time",
    )
    parser.add_argument(
        "--entropy_decay_rate",
        type=float,
        default=0.997,
        help="decay rate for entropy coefficient (default: 0.99)",
    )
    parser.add_argument(
        "--min_entropy_coef",
        type=float,
        default=0.00001,
        help="minimum entropy coefficient value (default: 0.001)",
    )
    
    # UAV MEC environment specific parameters
    parser.add_argument(
        "--uav_space_x",
        type=int,
        default=10,
        help="UAV MEC environment x dimension (default: 10)",
    )
    parser.add_argument(
        "--uav_space_y", 
        type=int,
        default=10,
        help="UAV MEC environment y dimension (default: 10)",
    )
    parser.add_argument(
        "--uav_space_z",
        type=int, 
        default=5,
        help="UAV MEC environment z dimension (default: 5)",
    )
    parser.add_argument(
        "--uav_num_uavs",
        type=int,
        default=3,
        help="Number of UAVs in the environment (default: 3)",
    )
    parser.add_argument(
        "--uav_num_base_stations",
        type=int,
        default=4, 
        help="Number of base stations (default: 4)",
    )
    parser.add_argument(
        "--uav_num_users",
        type=int,
        default=8,
        help="Number of mobile users (default: 8)",
    )
    parser.add_argument(
        "--uav_total_tasks",
        type=int,
        default=50,
        help="Total number of tasks to complete (default: 50)",
    )
    parser.add_argument(
        "--uav_task_generation_rate",
        type=float,
        default=0.3,
        help="Task generation probability per user per step (default: 0.3)",
    )
    parser.add_argument(
        "--uav_max_speed",
        type=float,
        default=2.0,
        help="Maximum UAV movement distance per time step (default: 2.0)",
    )
    parser.add_argument(
        "--uav_communication_range",
        type=float,
        default=5.0,
        help="UAV communication range (default: 5.0)",
    )
    parser.add_argument(
        "--uav_min_safe_distance",
        type=float,
        default=1.0,
        help="Minimum safe distance between UAVs (default: 1.0)",
    )
    # UAVMEC奖励设置
    parser.add_argument(
        "--reward_task_completion",
        type=float,
        default=0,  # Changed from 10.0 to -10.01
        help="Reward weight for task completion (default: 1)",
    )
    parser.add_argument(
        "--reward_energy_efficiency",
        type=float,
        default=0,  # Already correct
        help="Reward weight for energy efficiency (default: 1)",
    )
    parser.add_argument(
        "--reward_latency_penalty",
        type=float,
        default=0, # Already correct
        help="Reward weight for latency penalty (default: 1)",
    )
    parser.add_argument(
        "--reward_collision_penalty",
        type=float,
        default=0,  # Already correct
        help="Reward weight for collision penalty (default: 1)",
    )
    parser.add_argument(
        "--reward_boundary_penalty",
        type=float,
        default=0,  # Already correct
        help="Reward weight for boundary violation penalty (default: 1)",
    )
    parser.add_argument(
        "--reward_cooperation_bonus",
        type=float,
        default=0,  # Changed from 2.0 to -2.0
        help="Reward weight for cooperation bonus (default: 1)",
    )

    # UAV Edge Computing environment specific parameters
    parser.add_argument(
        "--edge_num_uavs",
        type=int,
        default=2,
        help="Number of UAVs in edge computing environment (default: 2)",
    )
    parser.add_argument(
        "--edge_num_terminals",
        type=int,
        default=6,
        help="Number of ground terminals (default: 6)",
    )
    parser.add_argument(
        "--edge_ground_area",
        type=float,
        default=400.0,
        help="Ground area size in meters (default: 400.0)",
    )
    parser.add_argument(
        "--edge_height_min",
        type=float,
        default=20.0,
        help="Minimum UAV flight height in meters (default: 20.0)",
    )
    parser.add_argument(
        "--edge_height_max",
        type=float,
        default=120.0,
        help="Maximum UAV flight height in meters (default: 120.0)",
    )
    parser.add_argument(
        "--edge_time_slot",
        type=float,
        default=1.0,
        help="Time slot duration in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--edge_data_min",
        type=float,
        default=10.0,
        help="Minimum task data size in KB (default: 10.0)",
    )
    parser.add_argument(
        "--edge_data_max",
        type=float,
        default=20.0,
        help="Maximum task data size in KB (default: 20.0)",
    )
    parser.add_argument(
        "--edge_cpu_cycles_per_bit",
        type=int,
        default=1000,
        help="CPU cycles required per bit (default: 1000)",
    )
    parser.add_argument(
        "--edge_f_u",
        type=float,
        default=5e9,
        help="UAV CPU frequency in cycles/s (default: 5e9)",
    )
    parser.add_argument(
        "--edge_f_g",
        type=float,
        default=1e9,
        help="Ground server CPU frequency in cycles/s (default: 1e9)",
    )
    parser.add_argument(
        "--edge_max_horizontal_speed",
        type=float,
        default=10.0,
        help="Maximum UAV horizontal speed in m/s (default: 10.0)",
    )
    parser.add_argument(
        "--edge_max_vertical_speed",
        type=float,
        default=10.0,
        help="Maximum UAV vertical speed in m/s (default: 10.0)",
    )
    parser.add_argument(
        "--edge_communication_range",
        type=float,
        default=100.0,
        help="UAV communication range in meters (default: 100.0)",
    )

    # UAV Edge Computing reward parameters
    parser.add_argument(
        "--edge_reward_task_completion",
        type=float,
        default=10.0,
        help="Reward weight for task completion (default: 10.0)",
    )
    parser.add_argument(
        "--edge_reward_approach_terminal",
        type=float,
        default=0.1,
        help="Reward for approaching target terminal (default: 0.1)",
    )
    parser.add_argument(
        "--edge_reward_energy_efficiency",
        type=float,
        default=1.0,
        help="Reward weight for energy efficiency (default: 1.0)",
    )
    parser.add_argument(
        "--edge_reward_energy_penalty",
        type=float,
        default=-0.0001,  # 从-0.01改为-0.0001，减少100倍
        help="Penalty for energy consumption (default: -0.0001)",
    )
    parser.add_argument(
        "--edge_reward_boundary_penalty",
        type=float,
        default=-10.0,
        help="Penalty for boundary violation (default: -10.0)",
    )
    parser.add_argument(
        "--edge_reward_all_tasks_bonus",
        type=float,
        default=100.0,
        help="Bonus for completing all tasks (default: 100.0)",
    )
    parser.add_argument(
        "--edge_reward_no_processing_penalty",
        type=float,
        default=-2.0,
        help="Penalty when UAV processes no data in a step (default: -2.0)",
    )

    # Nash Equilibrium Game Theory reward parameters
    parser.add_argument(
        "--nash_w_task_progress",
        type=float,
        default=1.0,
        help="Nash reward weight for task progress (default: 1.0)",
    )
    parser.add_argument(
        "--nash_w_connection_quality",
        type=float,
        default=0.3,
        help="Nash reward weight for connection quality (default: 0.3)",
    )
    parser.add_argument(
        "--nash_w_energy_efficiency",
        type=float,
        default=0.2,
        help="Nash reward weight for energy efficiency (default: 0.2)",
    )
    parser.add_argument(
        "--nash_w_terminal_competition",
        type=float,
        default=-0.5,
        help="Nash reward weight for terminal competition penalty (default: -0.5)",
    )
    parser.add_argument(
        "--nash_w_collision_avoidance",
        type=float,
        default=-2.0,
        help="Nash reward weight for collision avoidance (default: -2.0)",
    )
    parser.add_argument(
        "--nash_w_coverage_coordination",
        type=float,
        default=0.4,
        help="Nash reward weight for coverage coordination (default: 0.4)",
    )
    parser.add_argument(
        "--nash_w_load_balance",
        type=float,
        default=0.3,
        help="Nash reward weight for load balance (default: 0.3)",
    )
    parser.add_argument(
        "--nash_task_completion_bonus",
        type=float,
        default=50.0,
        help="Nash bonus for task completion (default: 50.0)",
    )
    parser.add_argument(
        "--nash_boundary_penalty",
        type=float,
        default=-5.0,
        help="Nash penalty for boundary violation (default: -5.0)",
    )
    parser.add_argument(
        "--nash_collision_penalty",
        type=float,
        default=-5.0,
        help="Nash penalty for collision (default: -5.0)",
    )
    parser.add_argument(
        "--nash_epsilon",
        type=float,
        default=0.1,
        help="Nash equilibrium tolerance (default: 0.1)",
    )
    parser.add_argument(
        "--best_response_threshold",
        type=float,
        default=0.05,
        help="Best response threshold for Nash adjustment (default: 0.05)",
    )
    parser.add_argument(
        "--use_all_params_reward",
        action="store_true",
        default=False,
        help="Use the all-params reward path for UAVEdgeComputingNash (default: False)",
    )

    # Scheme C (Cooperative Auction) reward parameters
    parser.add_argument(
        "--scheme_c_auction_win_reward",
        type=float,
        default=0.10,
        help="Scheme C: Reward for winning auction and serving terminal (default: 0.10)",
    )
    parser.add_argument(
        "--scheme_c_auction_participate_reward",
        type=float,
        default=0.02,
        help="Scheme C: Reward for participating in auction (default: 0.02)",
    )
    parser.add_argument(
        "--scheme_c_auction_payment_ratio",
        type=float,
        default=0.1,
        help="Scheme C: Payment ratio for auction winner (default: 0.1)",
    )
    parser.add_argument(
        "--scheme_c_task_completion_max",
        type=float,
        default=20.0,
        help="Scheme C: Maximum task completion reward (default: 20.0)",
    )
    parser.add_argument(
        "--scheme_c_task_completion_min",
        type=float,
        default=5.0,
        help="Scheme C: Minimum task completion reward (default: 5.0)",
    )
    parser.add_argument(
        "--scheme_c_decay_start_step",
        type=int,
        default=800,
        help="Scheme C: Step to start reward decay (default: 800)",
    )
    parser.add_argument(
        "--scheme_c_coop_bonus_perfect",
        type=float,
        default=5.0,
        help="Scheme C: Perfect cooperation bonus (default: 5.0)",
    )
    parser.add_argument(
        "--scheme_c_coop_bonus_good",
        type=float,
        default=3.0,
        help="Scheme C: Good cooperation bonus (default: 3.0)",
    )
    parser.add_argument(
        "--scheme_c_coop_bonus_fair",
        type=float,
        default=1.0,
        help="Scheme C: Fair cooperation bonus (default: 1.0)",
    )

    # UAV Edge Computing visualization parameters
    parser.add_argument(
        "--enable_visualization",
        action="store_true",
        default=False,
        help="Enable real-time 2D visualization of the UAV environment (default: False)",
    )
    parser.add_argument(
        "--visualization_start_episode",
        type=int,
        default=0,
        help="Start visualization from this episode number (0 means from beginning, default: 0)",
    )
    parser.add_argument(
        "--data_record_interval",
        type=int,
        default=None,
        help="Data recording interval in training mode (steps, None for auto: 50 for train, 1 for test)",
    )
    parser.add_argument(
        "--data_max_episodes",
        type=int,
        default=None,
        help="Maximum episodes to record in training mode (None for auto: 10 for train, 999 for test)",
    )
    # save parameters
    parser.add_argument(
        "--save_interval",
        type=int,
        default=8,
        help="time duration between contiunous twice models saving.",
    )

    # log parameters
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="time duration between contiunous twice log printing.",
    )

    # eval parameters
    parser.add_argument(
        "--use_eval",
        action="store_true",
        default=False,
        help="by default, do not start evaluation. If set`, start evaluation alongside with training.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=4,
        help="time duration between contiunous twice evaluation progress.",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=32,
        help="number of episodes of a single evaluation.",
    )

    # render parameters
    parser.add_argument(
        "--save_gifs",
        action="store_true",
        default=False,
        help="by default, do not save render video. If set, save video.",
    )
    parser.add_argument(
        "--use_render",
        action="store_true",
        default=False,
        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.",
    )
    parser.add_argument(
        "--render_episodes",
        type=int,
        default=5,
        help="the number of episodes to render a given env",
    )
    parser.add_argument(
        "--ifi",
        type=float,
        default=0.1,
        help="the play interval of each rendered image in saved video.",
    )

    # pretrained parameters
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="by default None. set the path to pretrained model.",
    )

    return parser
