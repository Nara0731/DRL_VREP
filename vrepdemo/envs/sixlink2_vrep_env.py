from vrepdemo.vrep_env import vrep_env
import os

vrep_scenes_path = '/home/ubuntu/pytorch-a2c-ppo-acktr/vrepdemo/scenes'

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

RAD2DEG = 180/np.pi


class SixLink2VrepEnv(vrep_env.VrepEnv):
    metadata = {'render.modes': [],}

    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=vrep_scenes_path+'/UR2.ttt',
                 ):
        vrep_env.VrepEnv.__init__(
            self,
            server_addr,
            server_port,
            scene_path,
        )
        self.random_start = False
        joint_names = ['UR10_joint1', 'UR10_joint2', 'UR10_joint3', 'UR10_joint4', 'UR10_joint5', 'UR10_joint6']
        shape_names = ['UR10_link2', 'UR10_link3', 'UR10_link4', 'UR10_link5', 'UR10_link6', 'UR10_link7',
                       'Contactbody']

        self.oh_joint = list(map(self.get_object_handle, joint_names))
        self.oh_shape = list(map(self.get_object_handle, shape_names))

        self.force = self.get_object_handle('UR10_connection')
        # 第一次 初始化 调用, 因为没有开vrep.simxSynchronousTrigger(clientID)，力传感器没有读数
        # fF = 0
        # fF = self.obj_read_force_sensor(self.force)
        # print('Init Foece:', fF)

        # One action per joint
        self.dim_act = len(self.oh_joint)
        # Multiple dimensions per shape
        self.dim_obs = 20

        high_act = np.ones([self.dim_act])
        high_obs = np.inf * np.ones([self.dim_obs])

        self.action_space = gym.spaces.Box(-high_act, high_act)
        self.observation_space = gym.spaces.Box(-high_obs, high_obs)

        self.joints_max_velocity = np.array([2, 5, 5, 2, 10, 2])*np.pi/180  # 第二个关节期望0.04rad/s，也就是2.29°
        #self.joints_max_velocity = 9999
        self.power = np.array([100, 100, 110, 10, 50, 10])
        # self.joint_max_position = 2*np.pi/180

        self.seed()

        print('SixLinkVrepEnv: initialized')

    def _make_observation(self):
        """Get observation from v-rep and stores in self.observation
        """
        # F
        # 第一次 初始化 调用, 因为没有开vrep.simxSynchronousTrigger(clientID)，力传感器没有读数
        # 要放在最前面，不然过一会就不能调用呢
        fF = self.obj_read_force_sensor(self.force)


        lst_o = []
        # q1 ~ q6
        for i_oh in self.oh_joint:
            lst_o += [self.obj_get_joint_angle(i_oh)]
        # dq1 ~ dq6
        for i_oh in self.oh_joint:
            lst_o += [self.obj_get_joint_velocity(i_oh)[0]]   # 虽然为1维，但是是元组
        # X, Z
        X = self.obj_get_position(self.oh_shape[-1])
        Rx = self.obj_get_orientation(self.oh_shape[-1])
        lst_o += X
        lst_o += Rx
        # dX
        dX, dXr = self.obj_get_velocity(self.oh_shape[-1])
        lst_o += [dX[1]]

        if fF is not None and fF is not 0:
            lst_o.append(np.abs(fF[0][2]+0.385))
        else:
            lst_o += [0]

        self.observation = np.array(lst_o).astype('float32')

    def _make_action(self, a):
        """Send action to v-rep
        """
        # a = np.clip(a, -1, 1)
        for i_oh, i_a, i_power, i_velocity in zip(self.oh_joint, a, self.power, self.joints_max_velocity):
            # self.obj_set_velocity(i_oh, i_a)
            # 这个地方应该设置两种
            # 设置电机力矩：也就是把 电机的幅度*power给电机
            # 设置期望速度：也就是把 电机的符号*joints_max_velocity 给期望速度
            # a_f = np.abs(np.clip(i_a, -1, +1))   # 提取大小
            # a_s = np.sign(i_a)                   # 提取符号
            # self.obj_set_velocity(i_oh, self.joints_max_velocity * a_s)
            # self.obj_set_force(i_oh, self.power * a_f)
            self.obj_set_velocity(i_oh, i_velocity * i_a)
            self.obj_set_force(i_oh, i_power)
        # self._make_observation()
        # q = self.observation[0:6]
        # for i_oh, i_a, i_q in zip(self.oh_joint, a, q):
        #     q_c = i_q + i_a * self.joint_max_position
        #     # print('target: %s' % q_c)
        #     # print('actual: %s' % i_q)
        #     self.obj_set_position_target(i_oh, q_c)



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
        target_fz = 10
        target_vy = 0.02
        fz = self.observation[-1]
        vy = self.observation[-2]
        print('vy=%s, f=%s' % (vy, fz))
        # ex = target_vx - vx
        ey = target_vy - vy
        ez = target_fz - fz
        bonus = 1
        lam1 = 10     # 精确到cm
        lam2 = 0.07
        lam3 = 1

        reward = -lam1*np.linalg.norm(ey) - lam2*np.linalg.norm(ez)

        # Early stop
        # target_x = -0.1889 + 0.06        # 初始位置为0.1702
        target_z = 1.14              # 初始高度为1.0650

        x = self.observation[12]
        y = self.observation[13]
        # print(y)
        z = self.observation[14]
        target_y = 0.7650 + 0.06
        #print(z)
        #print(y)

        # y的位置： + 0.7285
        # z的位置： + 0.4600
        x_lim = [-0.1889-0.03, -0.1889+0.03]
        y_lim = [0.7650-0.01, 0.7650+0.05]
        z_lim = [0.5559-0.05, 0.605]
        done1 = (y > target_y) and (ez < 2)
        done2 = (z < z_lim[0]) or (z > z_lim[1]) or (x < x_lim[0]) or (x > x_lim[1]) or (y < y_lim[0]) # 只能沿着x轴的方向走
        # done2 = (z < z_lim[0]) or (z > z_lim[1])
        done = False
        if done1:
            done = done1
            # print('OK')

        if done2:
            # if (z < z_lim[0]) or (z > z_lim[1]):
            #     print('z方向结束！')
            # else:
            #     print('x方向结束！')
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

        # 读取状态之前，需要走一步，不然没办法读取力传感器
        for i in range(1):
            self._make_action(np.zeros(6, ))
            self.step_simulation()
        self._make_observation()
        # print(self.observation[-1])
        print('Initial Start!!!')

        return self.observation

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


def main(args):
    env = SixLink2VrepEnv()

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





