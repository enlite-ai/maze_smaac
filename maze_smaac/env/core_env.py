"""Contains the core env for the l2rpn-challenge. This is Maze-specific and unrelated to the SMAAC approach. """
import os
from collections import ChainMap
from typing import Tuple, Dict, Any, Union, Mapping, Type, Optional

import grid2op
import numpy as np
from grid2op.Action import PlayableAction
from grid2op.Observation.CompleteObservation import CompleteObservation
from grid2op.Reward import BaseReward
from maze.core.annotations import override
from maze.core.env.core_env import CoreEnv
from maze.core.env.structured_env import StepKeyType, ActorID
from maze.core.events.pubsub import Pubsub
from maze.core.rendering.renderer import Renderer
from maze.core.utils.factory import Factory
from maze.utils.bcolors import BColors

import maze_smaac
from maze_smaac.env.events import ActionEvents, GridEvents, RewardEvents
from maze_smaac.env.kpi_calculator import Grid2OpKpiCalculator
from maze_smaac.env.l2rpn_renderer import L2RPNRenderer
from maze_smaac.reward.reward_aggregator import BaseRewardAggregator


class Grid2OpCoreEnvironment(CoreEnv):
    """Core env for the l2rpn challenge.

    :param power_grid: The name fo the grid2op environment setting.
    :param difficulty: The difficulty level of the env.
    :param reward: The reward configuration.
    :param reward_aggregator: The reward aggregator used.
    """

    def __init__(self,
                 power_grid: Union[str, grid2op.Environment.Environment],
                 difficulty: Union[int, str],
                 reward: Union[Type[BaseReward], str, Mapping[str, Any]],
                 reward_aggregator: Union[BaseRewardAggregator, str, Mapping[str, Any]]):
        super().__init__()

        self.wrapped_env: Optional[grid2op.Environment.Environment] = None
        """the external grid2op environment"""

        # init pubsub for event to reward routing
        self.pubsub = Pubsub(self.context.event_service)

        # init reward function(s)
        self._init_reward(reward)

        # setup environment
        self.power_grid = power_grid
        self._init_env(power_grid, difficulty)
        self._setup_env()

        # init reward aggregator
        self._init_reward_aggregator(reward_aggregator)

        # KPIs calculation
        self.kpi_calculator = Grid2OpKpiCalculator()

        # Rendering
        self.renderer = L2RPNRenderer(self.wrapped_env.observation_space)

        self._current_state: Optional[CompleteObservation] = None

    def _setup_env(self) -> None:
        """Setup environment.
        """
        self.action_events = self.pubsub.create_event_topic(ActionEvents)
        self.grid_events = self.pubsub.create_event_topic(GridEvents)
        self.reward_events = self.pubsub.create_event_topic(RewardEvents)

    def _init_env(self, power_grid: Union[str, grid2op.Environment.Environment], difficulty: Union[int, str]) -> None:
        """Instantiate power grid environment from problem instance identifier.

        :param power_grid: Power grid problem instance identifier or instance.
        :param difficulty: The difficulty level of the env
        """
        possible_difficulties = [0, 1, 2, '0', '1', '2', 'competition']
        assert difficulty in possible_difficulties, f'The difficulty should be in {possible_difficulties}'
        if isinstance(power_grid, str):
            # combine all reward classes in a single dict
            rewards = ChainMap(self.reward_classes, self.kpi_classes)

            try:
                import lightsim2grid
                backend = lightsim2grid.LightSimBackend()
                backend = {'backend': backend}
            except ImportError:
                BColors.print_colored('Lightsim2grid backend could not be found. Using Pandas instead', BColors.WARNING)
                backend = {}

            # make env from dataset
            if power_grid == 'l2rpn_wcci_2020':
                dataset = os.path.join(maze_smaac.__path__._path[0], 'data', 'l2rpn_wcci_2020')
            else:
                raise ValueError("dataset must be set from path for smaac env")
            self.wrapped_env: grid2op.Environment.Environment = grid2op.make(dataset,
                                                                             test=True,
                                                                             reward_class=self.reward_class,
                                                                             other_rewards=dict(rewards),
                                                                             difficulty=str(difficulty), **backend)

        else:
            self.wrapped_env: grid2op.Environment.Environment = power_grid
            BColors.print_colored('env difficulty could not be applied since the env was passed as an instance',
                                  color=BColors.WARNING)

    def _init_reward(self, reward: Union[Type[BaseReward], str, Mapping[str, Any]]) -> None:
        """Instantiate rewards and "KPIs" for the environment.

        :param reward: The reward to use.
        """
        self.reward_classes = dict()
        self.kpi_classes = dict()

        if isinstance(reward, type):
            self.reward_class = reward

        elif isinstance(reward, str):
            self.reward_class = Factory(base_type=BaseReward).type_from_name(reward)

        # handle mapping type
        else:
            self.reward_class = Factory(base_type=BaseReward).type_from_name(reward["_target_"])

            if "rewards" in reward:
                for i, v in enumerate(reward["rewards"]):
                    _reward_class = Factory(base_type=BaseReward).type_from_name(v["_target_"])

                    # check if reward is a kpi score in reality
                    if v.get("kpi"):
                        self.kpi_classes[v["name"]] = _reward_class

                    # reward is not a kpi
                    if "name" in v:
                        self.reward_classes[v["name"]] = _reward_class
                    else:
                        self.reward_classes[f"reward_{i + 1}"] = _reward_class

    def _init_reward_aggregator(self, reward_aggregator: Union[BaseRewardAggregator, str, Mapping[str, Any]]) -> None:
        """Instantiate reward aggregator.

        :param reward_aggregator: The reward aggregator object.
        """
        self.reward_aggregator = Factory(base_type=BaseRewardAggregator).instantiate(reward_aggregator)
        self.pubsub.register_subscriber(self.reward_aggregator)

    @override(CoreEnv)
    def step(self, execution: PlayableAction) -> Tuple[CompleteObservation, np.array, bool, Dict[Any, Any]]:
        """Just passes the relevant information to the grid2op wrapped env.

        :param execution: Environment action to take.
        :return: state, reward, done, info
        """

        # step forward wrapped env
        self._current_state, reward, done, info = self.wrapped_env.step(execution)

        # fire l2rpn reward event
        self.reward_events.l2rpn_reward(reward=reward)

        # fire action events
        if info["is_illegal"]:
            redispatch = bool(info.get("is_illegal_redisp")) or bool(info.get("is_dispatching_illegal"))
            self.action_events.illegal_action_performed(redispatch=redispatch, reconnect=info["is_illegal_reco"])

        if info["is_ambiguous"]:
            self.action_events.ambiguous_action_performed()

        # log impact of action
        impact = execution.impact_on_objects()
        has_impact = impact["has_impact"]
        if not has_impact:
            self.action_events.noop_action_performed()

        if impact["topology"]["changed"]:
            self.action_events.topology_action_performed()

        if impact["redispatch"]["changed"]:
            self.action_events.redispatch_action_performed()

        # check for overflows
        for line_id, overflow in enumerate(self._current_state.timestep_overflow):
            if overflow > 0:
                self.grid_events.power_line_overload(line_id=line_id, rho=self._current_state.rho[line_id])

        # log reason for done
        if done:
            if not len(info["exception"]):
                exception_string = "none"
            else:
                exception_string = type(info["exception"][0]).__name__

            self.grid_events.done(exception_string)
        else:
            if len(info["exception"]) > 0:
                exception_string = type(info["exception"][0]).__name__
                self.grid_events.not_done_exception(exception_string)

        # fire additional reward events for logging
        if "rewards" in info:
            for reward_name, reward in info["rewards"].items():
                _kpi = bool(self.kpi_classes.get(reward_name))
                self.reward_events.other_reward(name=reward_name, reward=reward, is_kpi=_kpi)

        # accumulate reward
        reward = self.reward_aggregator.summarize_reward()

        return self._current_state, reward, done, info

    @override(CoreEnv)
    def get_maze_state(self) -> CompleteObservation:
        """Return current state of the environment.
        """
        return self._current_state

    @override(CoreEnv)
    def get_kpi_calculator(self) -> Grid2OpKpiCalculator:
        """KPIs are supported."""
        return self.kpi_calculator

    @override(CoreEnv)
    def reset(self):
        """Resets the environment"""
        self._current_state = self.wrapped_env.reset()
        return self._current_state

    @override(CoreEnv)
    def close(self) -> None:
        """No additional cleanup necessary."""
        pass

    @override(CoreEnv)
    def seed(self, seed) -> None:
        """Seeds the environment. Pass here since this is implemented differently for l2rpn."""
        pass

    @override(CoreEnv)
    def get_serializable_components(self) -> Dict[str, Any]:
        """Serializable all components used within the grid2op env.

        :return: a list of serialized components
        """
        return {}

    @override(CoreEnv)
    def get_renderer(self) -> Renderer:
        """Wrapped renderer for the grid2op env."""
        return self.renderer

    @override(CoreEnv)
    def actor_id(self) -> ActorID:
        """Currently implemented as single policy, single actor env.

        :return: Actor and id, in this case always 0,0
        """
        return ActorID(step_key=0, agent_id=0)

    @override(CoreEnv)
    def is_actor_done(self) -> bool:
        """The actors of this env are never done.

        :return: In this case always False
        """
        return False

    @property
    @override(CoreEnv)
    def agent_counts_dict(self) -> Dict[StepKeyType, int]:
        """Interface definition for core environments forming the basis for actual RL trainable environments.
        """
        return {0: 1}
