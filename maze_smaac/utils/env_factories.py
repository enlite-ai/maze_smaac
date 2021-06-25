"""Contains environment factory functions. This is Maze-specific and unrelated to the SMAAC approach. """
from typing import Optional, Union, Type, Mapping, Any

from grid2op.Reward import BaseReward

from maze_smaac.env.core_env import Grid2OpCoreEnvironment
from maze_smaac.env.maze_env import L2RPNMazeEnv
from maze_smaac.reward.reward_aggregator import RewardAggregator
from maze_smaac.space_interfaces.dict_action_conversion import MazeSMAACActionConversion
from maze_smaac.space_interfaces.dict_observation_conversion import MazeSMAACObservationConversion


def maze_smaac_env_factory(power_grid: str, mask: int, mask_hi: int, danger: float, n_history: int,
                           bus_threshold: float, reward_scale: float, seed: Optional[int],
                           reward: Union[Type[BaseReward], str, Mapping[str, Any]]) -> L2RPNMazeEnv:
    """
    Helper function to instantiate the MazeEnv

    :param power_grid: The power grid to use in the grid2op env.
    :param mask: Lower substation mask parameter.
    :param mask_hi: Upper substation mask parameter.
    :param danger: The danger threshold above which to act. delta_t in the paper.
    :param n_history: The number of observations to stack.
    :param bus_threshold: Corresponds to delta_t in Yoon et al. (2021) - Winning the l2rpn challenge:
           power grid management via semi markov afterstate actor critic.
    :param reward_scale:
    :param seed: Seed to use for the grid2op env. Should be None for training.
    :param reward: The reward configuration.
    """

    # Initialize the core environment
    core_env = Grid2OpCoreEnvironment(
        power_grid=power_grid,
        difficulty='competition',
        reward=reward,
        reward_aggregator=RewardAggregator(reward_scale=reward_scale)
    )
    # set seed
    if seed:
        core_env.wrapped_env.seed(seed)

    # Set grid2op parameters
    core_env.wrapped_env.deactivate_forecast()
    core_env.wrapped_env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = 3
    core_env.wrapped_env.parameters.NB_TIMESTEP_RECONNECTION = 12
    core_env.wrapped_env.parameters.NB_TIMESTEP_COOLDOWN_LINE = 3
    core_env.wrapped_env.parameters.NB_TIMESTEP_COOLDOWN_SUB = 3
    core_env.wrapped_env.parameters.HARD_OVERFLOW_THRESHOLD = 200.0

    # Init action and observation conversion interfaces.
    # Note that the action_conversion must be instantiated before the observation_conversion since the latter needs
    # access to the member variable action_conversion.n
    action_conversion = MazeSMAACActionConversion(
        env=core_env.wrapped_env, mask=mask, mask_hi=mask_hi, danger=danger,
        bus_threshold=bus_threshold
    )

    observation_conversion = MazeSMAACObservationConversion(
        env=core_env.wrapped_env, mask=mask, mask_hi=mask_hi, dim_topo=int(core_env.wrapped_env.dim_topo),
        danger=danger, n_history=n_history, output_dim=action_conversion.n)

    # initialize the maze environment
    maze_env = L2RPNMazeEnv(core_env=core_env, action_conversion=action_conversion,
                            observation_conversion=observation_conversion)

    return maze_env
