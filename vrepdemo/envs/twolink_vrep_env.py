from vrepdemo.vrep_env import vrep_env
import os

vrep_scenes_path = '/home/ubuntu/pytorch-a2c-ppo-acktr/vrepdemo/scenes'
#vrep_scenes_path = '../examples/scenes'


import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

RAD2DEG = 180/np.pi


class TwoLinkVrepEnv(vrep_env.VrepEnv):
    metadata = {'render.modes': [],}

    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=vrep_scenes_path+'/twolink.ttt',
                 ):
        vrep_env.VrepEnv.__init__(
            self,
            server_addr,
            server_port,
            scene_path,
        )
        self.random_start = False
        joint_names = ['joint1', 'joint2']
        shape_names = ['Link1', 'Link2', 'tip']

        self.oh_joint = list(map(self.get_object_handle, joint_names))
        self.oh_shape = list(map(self.get_object_handle, shape_names))

        self.force = self.get_object_handle('Force')
        # 第一次 初始化 调用, 因为没有开vrep.simxSynchronousTrigger(clientID)，力传感器没有读数
        # fF = 0
        # fF = self.obj_read_force_sensor(self.force)
        # print('Init Foece:', fF)

        # One action per joint
        dim_act = len(self.oh_joint)
        # Multiple dimensions per shape
        dim_obs = 9

        high_act = np.inf * np.ones([dim_act])
        high_obs = np.inf * np.ones([dim_obs])

        self.action_space = gym.spaces.Box(-high_act, high_act)
        self.observation_space = gym.spaces.Box(-high_obs, high_obs)

        self.joints_max_velocity = 30*np.pi/180  # 第二个关节期望0.04rad/s，也就是2.29°
        # self.joints_max_velocity = 9999
        self.power = 50

        self.seed()

        print('TwoLinkVrepEnv: initialized')

    def _make_observation(self):
        """Get observation from v-rep and stores in self.observation
        """
        # F
        # 第一次 初始化 调用, 因为没有开vrep.simxSynchronousTrigger(clientID)，力传感器没有读数
        # 要放在最前面，不然过一会就不能调用呢
        fF = self.obj_read_force_sensor(self.force)


        lst_o = []
        # q1, q2
        for i_oh in self.oh_joint:
            lst_o += [self.obj_get_joint_angle(i_oh)]
        # dq1, dq2
        for i_oh in self.oh_joint:
            lst_o += [self.obj_get_joint_velocity(i_oh)[0]]   # 虽然为1维，但是是元组
        # X, Z
        X = self.obj_get_position(self.oh_shape[-1])
        lst_o += [X[0]]
        lst_o += [X[2]]
        # dX, dZ
        dX, dXr = self.obj_get_velocity(self.oh_shape[-1])
        lst_o += [dX[0]]
        lst_o += [dX[2]]

        if fF is not None and fF is not 0:
            lst_o += [fF[0][2] - 9.81]
        else:
            lst_o += [0]

        self.observation = np.array(lst_o).astype('float32')

    def _make_action(self, a):
        """Send action to v-rep
        """
        # a = np.clip(a, -1, 1)
        #self.communicationPause()
        for i_oh, i_a in zip(self.oh_joint, a):
            # self.obj_set_velocity(i_oh, i_a)
            # 这个地方应该设置两种
            # 设置电机力矩：也就是把 电机的幅度*power给电机
            # 设置期望速度：也就是把 电机的符号*joints_max_velocity 给期望速度
            # vrep.simxSetJointTargetVelocity(clientID, jointHandle[0], vel, vrep.simx_opmode_streaming)
            # vrep.simxSetJointForce(clientID, jointHandle[0], tor, vrep.simx_opmode_streaming)
            ## 电机力矩
            a_f = np.abs(np.clip(i_a, -1, +1))   # 提取大小
            a_s = np.sign(i_a)                   # 提取符号
            self.obj_set_velocity(i_oh, self.joints_max_velocity * a_s)
            self.obj_set_force(i_oh, self.power * a_f)

            ## 期望速度
            # self.obj_set_force(i_oh, self.power)
            # self.obj_set_velocity(i_oh, self.joints_max_velocity * np.clip(i_a, -1, +1))

        #self.communicationGoon()

    def step(self, action):
        # Clip xor Assert
        # actions = np.clip(actions,-self.joints_max_velocity, self.joints_max_velocity)
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Actuate
        self._make_action(action)
        # self._make_action(action*self.joints_max_velocity)
        # Step
        self.step_simulation()
        # Observe
        self._make_observation()

        # Reward
        target_fz = 10
        target_vx = 0.02
        fz = self.observation[-1]
        vx = self.observation[-3]
        ex = target_vx - vx
        ez = target_fz - fz
        lam1 = 10
        lam2 = 0.07  # 精确到cm

        reward = -(lam1*np.linalg.norm(ex) + lam2*np.linalg.norm(ez))

        # Early stop
        target_x = 0.63 + 0.03
        target_z = 0.09

        #epx = 0.03
        epz = 0.02
        x = self.observation[4]
        z = self.observation[5]

        done1 = (x > target_x)  # and (np.abs(target_z - z) < epz)
        done2 = (z > 0.11 or z < 0.07)

        done = False
        if done1:
            done = done1

        if done2:
            done = done2
            reward -= 500

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

        # 读取状态之前，需要走一步，不然没办法读取力传感器
        self.step_simulation()
        self._make_observation()
        return self.observation

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


def main(args):
    env = TwoLinkVrepEnv()

    #env.reset()
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





