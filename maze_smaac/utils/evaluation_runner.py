""" Contains EvaluationRunner for the L2RPN env. This is Maze-specific and unrelated to the SMAAC approach. """

from maze.core.annotations import override
from maze.core.log_events.log_events_writer_registry import LogEventsWriterRegistry
from maze.core.log_events.log_events_writer_tsv import LogEventsWriterTSV
from maze.core.log_stats.log_stats import register_log_stats_writer
from maze.core.log_stats.log_stats_writer_console import LogStatsWriterConsole
from maze.core.rollout.rollout_runner import RolloutRunner
from maze.core.trajectory_recording.writers.trajectory_writer_file import TrajectoryWriterFile
from maze.core.trajectory_recording.writers.trajectory_writer_registry import TrajectoryWriterRegistry
from maze.core.utils.factory import ConfigType, CollectionOfConfigType
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.trajectory_recording_wrapper import TrajectoryRecordingWrapper

from maze_smaac.utils.smaac_rollout_evaluator import SMAACRolloutEvaluator


class EvaluationRunner(RolloutRunner):
    """Runs rollout in the local process. Useful for short rollouts or debugging.

    Trajectory, event logs and stats are recorded into the working directory managed by hydra (provided
    that the relevant wrappers are present.)

    :param deterministic: Whether to use argmax policy for rollouts
    :param use_test_set: Whether to use test set chronics for evaluation
    """

    def __init__(self, deterministic: bool, use_test_set: bool):
        # For this evaluation runner, fix some parameters.
        n_episodes = 1
        max_episode_steps = 1000000
        record_trajectory = False
        record_event_logs = False
        super().__init__(n_episodes, max_episode_steps, record_trajectory, record_event_logs)

        self.deterministic = deterministic
        self.use_test_set = use_test_set
        self.progress_bar = None

    @override(RolloutRunner)
    def run_with(self, env: ConfigType, wrappers: CollectionOfConfigType, agent: ConfigType):
        """Run the rollout sequentially in the main process."""
        env, agent = self.init_env_and_agent(env, wrappers, self.max_episode_steps, agent, self.input_dir,
                                             self.maze_seeding.generate_env_instance_seed(),
                                             self.maze_seeding.generate_agent_instance_seed())

        # Set up the wrappers
        # Hydra handles working directory
        register_log_stats_writer(LogStatsWriterConsole())
        if not isinstance(env, LogStatsWrapper):
            env = LogStatsWrapper.wrap(env, logging_prefix="rollout_data")
        if self.record_event_logs:
            LogEventsWriterRegistry.register_writer(LogEventsWriterTSV(log_dir="./event_logs"))
        if self.record_trajectory:
            TrajectoryWriterRegistry.register_writer(TrajectoryWriterFile(log_dir="./trajectory_data"))
            if not isinstance(env, TrajectoryRecordingWrapper):
                env = TrajectoryRecordingWrapper.wrap(env)

        # Initialize evaluator (n_eval_episodes will be provided before evaluation.yaml)
        self.evaluator = SMAACRolloutEvaluator(eval_env=env, n_episodes=self.n_episodes, model_selection=None,
                                               deterministic=self.deterministic,
                                               use_test_set=self.use_test_set)
        self.evaluator.evaluate(agent)

    def update_progress(self):
        """Called on episode end to update a simple progress indicator."""
        self.progress_bar.update()
