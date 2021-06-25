"""Contains reward implementation adopted from
 https://github.com/KAIST-AILab/SMAAC/blob/53c52f35dfa9224d1adfc5d3e9e67912b7cf0f1b/custom_reward.py """
import grid2op
from grid2op.Reward.BaseReward import BaseReward


class LossReward(BaseReward):
    """Computes reward as the scaled ratio between total load and total production.
    """

    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = -1.0
        self.reward_illegal = -0.5
        self.reward_max = 1.0

    def __call__(self, action: grid2op.Action.BaseAction, env: grid2op.Environment.BaseEnv, has_error: bool,
                 is_done: bool, is_illegal: bool, is_ambiguous: bool):
        if has_error:
            if is_illegal or is_ambiguous:
                return self.reward_illegal
            elif is_done:
                return self.reward_min
        gen_p, *_ = env.backend.generators_info()
        load_p, *_ = env.backend.loads_info()
        reward = (load_p.sum() / gen_p.sum() * 10. - 9.) * 0.1
        return reward
