import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

import random

tf.disable_v2_behavior()
# import gym


class Advantage_Actor_Critic(object):
    def __init__(self, n_features, n_actions, actor_lr=0.01, critic_lr=0.01, reward_decay=0.9):
        self.n_features = n_features
        self.n_actions = n_actions
        # 奖励衰减值
        self.gamma = reward_decay
        # actor网络学习率
        self.actor_lr = actor_lr
        # critic网络学习率
        self.critic_lr = critic_lr

        self._build_net()  # 建立网络

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.choose_action_random_probability = 0.7

    def _build_net(self):
        tf.reset_default_graph()  # 清空计算图

        # 网络输入当前状态
        self.s = tf.placeholder(tf.float32, [1, self.n_features], "state")
        # 当前选择的动作值，只要Critic使用
        self.a = tf.placeholder(tf.int32, None, "act")
        # Critic产生的td_error，用于更新Actor
        self.actor_td_error = tf.placeholder(tf.float32, None, "td_error")
        # 下一步状态st+1的价值，用于更新Critic网络
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        # 执行动作获得的奖励，由环境反馈
        self.r = tf.placeholder(tf.float32, None, 'r')

        # 权重初始化方式
        w_initializer = tf.random_normal_initializer(0.0, 0.1)
        b_initializer = tf.constant_initializer(0.1)
        n_l1 = 20  # n_l1为network隐藏层神经元的个数

        # 此处Actor和Critic共享权重
        with tf.variable_scope("share-l1"):
            w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer)
            b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer)
            l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

        # 此处是属于Actor
        with tf.variable_scope("Actor"):
            w21 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer)
            b21 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer)
            # Actor的输出兀()
            self.acts_prob = tf.nn.softmax(tf.matmul(l1, w21) + b21)

        # 创建Critic。输出当前状态s动作a的价值
        with tf.variable_scope("Critic"):
            w22 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer)
            b22 = tf.get_variable('b2', [1], initializer=b_initializer)
            # Actor的输出兀()
            self.v = tf.matmul(l1, w22) + b22

        # Critic的loss计算
        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + self.gamma * self.v_ - self.v
            self.critic_loss = tf.square(self.td_error)

        with tf.variable_scope('critic_train_op'):
            self.critic_train_op = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)

        # Actor的loss计算
        with tf.variable_scope('exp_v'):
            # 计算更新公式中梯度log部分
            neg_log_prob = -tf.reduce_sum(tf.log(self.acts_prob) * tf.one_hot(self.a, self.n_actions), axis=1)
            # 计算损失，将之前policy gradient中的Gt和基准线b的差，改为Critic的td error即可
            self.exp_v = tf.reduce_mean(neg_log_prob * self.actor_td_error)

        with tf.variable_scope('actor_train_op'):
            self.actor_train_op = tf.train.AdamOptimizer(self.actor_lr).minimize(self.exp_v)

    def actor_learn(self, s, a, td):
        s = s[np.newaxis, :]
        # 训练并返回损失
        exp_v, _ = self.sess.run([self.exp_v, self.actor_train_op],
                                 feed_dict={self.s: s, self.a: a, self.actor_td_error: td})
        return exp_v

    def critic_learn(self, s, r, s_):
        # 当前状态和下一个状态
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        # 下一个状态st+1的价值
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.critic_train_op], {self.s: s, self.v_: v_, self.r: r})
        # 返回td_error用于更新Actor网络
        return td_error

    # 需要根据actor选动作
    def choose_action(self, s):
        s = s[np.newaxis, :]
        # 得到动作的概率
        probs = self.sess.run(self.acts_prob, {self.s: s})
        print("Choose action", probs)
        # 按照概率选择动作
        possbility = random.random()
        if possbility < self.choose_action_random_probability:
            action = np.random.choice(range(probs.shape[1]), p=probs.ravel())
        else:
            action = np.random.choice(self.n_actions)
        print(action)
        return action


def run_maze(RENDER):
    for i_episode in range(3000):
        s = env.reset()
        t = 0
        track_r = []
        while True:
            if RENDER:
                env.render()
            # 根据actor输出概率选择动作
            a = advantage_actor_critic.choose_action(s)
            # 根据选择动作，得到下一步状态、奖励、是否结束和信息
            s_, r, done, info = env.step(a)
            if done:
                r = -20
            # 记录奖励
            track_r.append(r)
            # 使用critic网络学习td error
            td_error = advantage_actor_critic.critic_learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            # 利用critic网络输出td error学习actor
            advantage_actor_critic.actor_learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            # 状态变动
            s = s_
            t += 1

            if done:
                # 当前回合获得的总奖励
                ep_rs_sum = sum(track_r)
                if 'running_reward' not in locals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True  # rendering

                print("episode:", i_episode, "  reward:", int(running_reward))
                break


if __name__ == "__main__":
    RENDER = False
    DISPLAY_REWARD_THRESHOLD = 200
    env = gym.make('CartPole-v0')
    env = env.unwrapped  # 取消限制
    env.seed(1)
    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n

    advantage_actor_critic = Advantage_Actor_Critic(n_features=N_F, n_actions=N_A)
    run_maze(RENDER)
