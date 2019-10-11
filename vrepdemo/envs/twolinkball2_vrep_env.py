from vrepdemo.vrep_env import vrep_env
import os

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

vrep_scenes_path = '/home/ubuntu/pytorch-a2c-ppo-acktr/vrepdemo/scenes'
RAD2DEG = 180/np.pi


class TwoLinkBall2VrepEnv(vrep_env.VrepEnv):
    metadata = {'render.modes': [],}

    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=vrep_scenes_path+'/twolinkball2.ttt',
                 ):
        vrep_env.VrepEnv.__init__(
            self,
            server_addr,
            server_port,
            scene_path,
        )
        self.random_start = False
        joint_names = ['joint1', 'joint2']
        shape_names = ['link1', 'link2', 'Circle', 'Sphere']

        self.oh_joint = list(map(self.get_object_handle, joint_names))
        self.oh_shape = list(map(self.get_object_handle, shape_names))

        # One action per joint
        dim_act = len(self.oh_joint)
        # Multiple dimensions per shape
        dim_obs = 12

        high_act = np.inf * np.ones([dim_act])
        high_obs = np.inf * np.ones([dim_obs])

        self.action_space = gym.spaces.Box(-high_act, high_act)
        self.observation_space = gym.spaces.Box(-high_obs, high_obs)

        self.joints_max_velocity = 100*np.pi/180  # 第二个关节期望0.04rad/s，也就是2.29°
        self.power = 5

        self.seed()

        print('TwoLinkVrepEnv: initialized')

    def _make_observation(self):
        """Get observation from v-rep and stores in self.observation
        """

        lst_o = []

        # Ball Position
        X1 = self.obj_get_position(self.oh_shape[-1])
        lst_o += X1[0:3]

        # Circle Position
        X2 = self.obj_get_position(self.oh_shape[-2])
        lst_o += X2[0:3]

        # Ball Velocity
        dX1, dXr1 = self.obj_get_velocity(self.oh_shape[-1])
        lst_o += dX1

        # Circle Velocity
        dX2, dXr2 = self.obj_get_velocity(self.oh_shape[-1])
        lst_o += dX2

        self.observation = np.array(lst_o).astype('float32')

    def _make_action(self, a):
        """Send action to v-rep
        """
        for i_oh, i_a in zip(self.oh_joint, a):
            # 期望速度
            self.obj_set_force(i_oh, self.power)
            self.obj_set_velocity(i_oh, self.joints_max_velocity * np.clip(i_a, -1, +1))

    def step(self, action):
        # Clip xor Assert
        # actions = np.clip(actions,-self.joints_max_velocity, self.joints_max_velocity)
        action = np.clip(action, -1, 1)
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Actuate
        self._make_action(action)
        # self._make_action(action*self.joints_max_velocity)
        # Step
        self.step_simulation()
        # Observe
        self._make_observation()

        # Reward
        pos_ball = self.observation[0:3]
        pos_circle = self.observation[3:6]
        vec = pos_ball - pos_circle
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.linalg.norm(action)
        lam1 = 0.1
        lam2 = 0.1

        reward = lam1 * reward_dist + lam2 * reward_ctrl

        # Early stop
        vec_threshold = 0.1
        ball_z = 0.40

        # Catch the ball
        done1 = np.linalg.norm(vec) < vec_threshold
        # Miss the ball
        done2 = pos_ball[2] < ball_z

        done = False
        if done1:
            done = done1

        if done2:
            done = done2
            reward -= 100

        return self.observation, reward, done, {}

    def reset(self):
        if self.sim_running:
            self.stop_simulation()
        self.start_simulation()

        # First action is random: emulate random initialization
        if self.random_start:
            factor = self.np_random.uniform(low=0, high=0.02, size=(1,))[0]
            action = self.action_space.sample() * factor
            self._make_action(action)
            self.step_simulation()

        self._make_observation()
        return self.observation

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


def main(args):
    env = TwoLinkBall2VrepEnv()

    # env.reset()
    # for i in range(100):
    #     fF = env.obj_read_force_sensor1(env.force)
    # print(fF)
    # env.start_simulation()
    # for i in range(10):
    #     env.step_simulation()
    #     fF = env.obj_read_force_sensor(env.force)
    #     print(fF)

    # observation = env.reset()
    # print(observation)

    for i_episode in range(4):
        observation = env.reset()
        print(observation)
        total_reward = 0
        for t in range(200):
            action = env.action_space.sample()
            print(action)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        print("Episode finished after {} timesteps.\tTotal reward: {}".format(t + 1, total_reward))
    env.close()
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))





