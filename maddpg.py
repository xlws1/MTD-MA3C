import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1. - done)
        discounted.append(r)
    return discounted[::-1]


def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2  # Polyak averaging factor
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        # Soft update target network
        expression.append(var_target.assign(polyak * var_target + (1.0 - polyak) * var))
    expression = tf.group(*expression)  # Group updates for efficiency
    return U.function([], [], updates=[expression])


# 修改后的MLP模型
def mlp_model(input, num_outputs, scope, reuse=False, num_units=64):
    # 这是原始的MLP网络实现
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = tf.layers.dense(out, num_units, activation=tf.nn.relu, name="mlp1")
        out = tf.layers.dense(out, num_units, activation=tf.nn.relu, name="mlp2")
        out = tf.layers.dense(out, num_outputs, activation=None, name="output")
        return out


# 修改后的SE注意力模块 - 直接对特征进行处理，不需要堆叠
def se_attention(input_tensor, reduction=16, scope="se_attention", reuse=None):
    """
    Squeeze-and-Excitation注意力机制，用于处理单个智能体的观察

    Args:
        input_tensor: 输入张量 [batch_size, feature_dim]
        reduction: 缩减比例，控制中间层神经元数量
        scope: 变量作用域
        reuse: 是否重用已有变量

    Returns:
        带有注意力权重的输出特征 [batch_size, feature_dim]
    """
    with tf.variable_scope(scope, reuse=reuse):
        # 获取特征数量
        input_shape = input_tensor.get_shape().as_list()
        feature_dim = input_shape[-1]

        # Squeeze - 这一步对于1D输入是不必要的，但我们保留以与SE结构一致
        # 对于每个样本，我们已经有了一个向量表示

        # Excitation - 使用两个全连接层，第一个降维，第二个恢复维度
        reduced_dim = max(1, feature_dim // reduction)
        excitation = tf.layers.dense(input_tensor, reduced_dim, activation=tf.nn.relu, name="fc1")
        excitation = tf.layers.dense(excitation, feature_dim, activation=tf.nn.sigmoid, name="fc2")

        # Scale - 对特征进行按元素乘法，重新加权
        output = input_tensor * excitation

        return output


def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None,
            local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distributions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]

        # 使用SE注意力增强当前智能体的观察
        p_input_with_attention = se_attention(p_input, reduction=4, scope="p_se_attention", reuse=reuse)

        # 使用增强后的输入生成策略
        p = p_func(p_input_with_attention, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func",
                   num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:, 0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input_with_attention, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func",
                          num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}


def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False,
            scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distributions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        # 准备Q网络的输入
        if local_q_func:
            # 对于局部Q函数，仅使用当前智能体的观察和动作
            q_input_raw = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        else:
            # 对于全局Q函数，使用所有智能体的观察和动作
            q_input_raw = tf.concat(obs_ph_n + act_ph_n, 1)

        # 使用SE注意力增强Q输入
        q_input = se_attention(q_input_raw, reduction=4, scope="q_se_attention", reuse=reuse)

        # 使用Q网络预测价值
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:, 0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss  # + 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:, 0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


class MAACAgentTrainerWithAttention(AgentTrainer):
    def __init__(self, name, p_model, q_model, obs_shape_n, obs_space_n,
                 act_space_n, agent_index, args, local_q_func=False, ADV=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.ADV = ADV

        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation" + str(i)).get())

        if self.ADV:
            optimizer = tf.train.AdamOptimizer(learning_rate=args.lr_adv)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)

        # 改用SE注意力修改后的训练函数
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=q_model,
            optimizer=optimizer,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=p_model,
            q_func=q_model,
            optimizer=optimizer,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )

        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs, add_noise=True):
        """选择动作，可添加探索噪声"""
        action = self.act(obs[None])[0]
        if add_noise and hasattr(self.args, 'noise_std') and self.args.noise_std > 0:
            action += self.args.noise_std * np.random.randn(*action.shape)
        return np.clip(action, -1.0, 1.0)  # 裁剪动作到[-1, 1]范围

    def experience(self, obs, act, rew, new_obs, done, terminal=False):
        """存储经验到回放缓冲区"""
        # 合并 done 和 terminal：如果 terminal 为 True，则认为情节结束
        effective_done = bool(done) or terminal
        self.replay_buffer.add(obs, act, rew, new_obs, float(effective_done))

    def preupdate(self):
        """在更新前进行准备工作"""
        self.replay_sample_index = None

    def update(self, agents, t):
        """更新智能体的网络"""
        if len(self.replay_buffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        # 采样经验
        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        index = self.replay_sample_index
        obs_n = []
        obs_next_n = []
        act_n = []
        for i in range(self.n):
            obs_i, act_i, rew_i, obs_next_i, done_i = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs_i)
            obs_next_n.append(obs_next_i)
            act_n.append(act_i)

        # 获取当前智能体的经验
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # 训练 Q 网络
        num_sample = 1
        target_q = 0.0
        for _ in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))

            # 确保 `rew` 和 `done` 的形状与 `target_q_next` 一致
            adjusted_rew = -rew if self.ADV else rew
            adjusted_rew = np.reshape(adjusted_rew, target_q_next.shape)
            done = np.reshape(done, target_q_next.shape)

            target_q += adjusted_rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample

        # 训练 Q 网络
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # 训练策略网络
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]