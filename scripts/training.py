"""
Training script for Maze adaptation of SMAAC for L2RPN.

To see how to further configure and customize your training runs, check out
- the Getting Started notebooks: https://github.com/enlite-ai/maze/tree/main/tutorials/notebooks.
- the documentation: https://maze-rl.readthedocs.io/en/latest/.
"""

from maze.api.run_context import RunContext
from data_acquisition import get_data

if __name__ == '__main__':
    get_data(base_path="maze_smaac/data/")

    # Train policy.
    # Alternatively we can train with Maze' CLI interface by invoking:
    # maze-run -cn conf_train env=maze_smaac model=maze_smaac_nets wrappers=maze_smaac_train +experiment=sac_train
    rc_train = RunContext(
        silent=True,
        env="maze_smaac",
        model="maze_smaac_nets",
        wrappers="maze_smaac_train",  # Use "maze_smaac_debug" for debug mode.
        experiment="sac_train",  # Use "sac_dev" for debug mode.
    )
    rc_train.train(n_epochs=1)

    # Evaluate policy.
    rc_eval = RunContext(
        run_dir=rc_train.run_dir,
        algorithm="sac",
        env="maze_smaac",
        model="maze_smaac_nets",
        wrappers="maze_smaac_rollout",
        configuration="test"
    )

    # Evaluate trained policy.
    stats = rc_eval.evaluate(
        _target_="maze_smaac.utils.smaac_rollout_evaluator.SMAACRolloutEvaluator",
        deterministic=True,
        use_test_set=True,
        n_episodes=1
    )

    # Roll out trained policy.
    env = rc_eval.env_factory()
    obs = env.reset()
    done = False
    i_steps = 0

    while not done:
        action = rc_eval.compute_action(obs)
        obs, reward, done, info = env.step(action)
        i_steps += 1

