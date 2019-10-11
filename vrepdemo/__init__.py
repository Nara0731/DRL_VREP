from gym.envs.registration import register

register(
    id='vrep-hopper-v0',
    entry_point='vrepdemo.envs:HopperVrepEnv',
)

register(
    id='vrep-twolink-v0',
    entry_point='vrepdemo.envs:TwoLinkVrepEnv',
)

register(
    id='vrep-twolink1-v0',
    entry_point='vrepdemo.envs:TwoLink1VrepEnv',
)

register(
    id='vrep-twolink2-v0',
    entry_point='vrepdemo.envs:TwoLink2VrepEnv',
)

register(
    id='vrep-sixlink1-v0',
    entry_point='vrepdemo.envs:SixLink1VrepEnv',
)

register(
    id='vrep-sixlink2-v0',
    entry_point='vrepdemo.envs:SixLink2VrepEnv',
)

register(
    id='vrep-twolinkball-v0',
    entry_point='vrepdemo.envs:TwoLinkBallVrepEnv',
)

register(
    id='vrep-twolinkball2-v0',
    entry_point='vrepdemo.envs:TwoLinkBall2VrepEnv',
)

