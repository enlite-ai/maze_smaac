""" Contains L2RPN MazeEnv and env factory. This is Maze-specific and unrelated to the SMAAC approach. """
from maze.core.env.maze_env import MazeEnv

from maze_smaac.env.core_env import Grid2OpCoreEnvironment
from maze_smaac.space_interfaces.dict_action_conversion import MazeSMAACActionConversion
from maze_smaac.space_interfaces.dict_observation_conversion import MazeSMAACObservationConversion


class L2RPNMazeEnv(MazeEnv[Grid2OpCoreEnvironment]):
    """ L2RPN MazeEnv """
    def __init__(self,
                 core_env: Grid2OpCoreEnvironment,
                 action_conversion: MazeSMAACActionConversion,
                 observation_conversion: MazeSMAACObservationConversion):
        super(L2RPNMazeEnv, self).__init__(core_env=core_env,
                                           action_conversion_dict={0: action_conversion},
                                           observation_conversion_dict={0: observation_conversion})
