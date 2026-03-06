import numpy as np
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0.0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1.0 - done)
        discounted.append(r)
    return discounted[::-1]


def make_update_exp(vals, target_vals, tau=1e-2):
    """
    Soft target update:
        theta_target <- (1 - tau) * theta_target + tau * theta
    """
    polyak = 1.0 - tau
    expression = []
    vals = sorted(vals, key=lambda v: v.name)
    target_vals = sorted(target_vals, key=lambda v: v.name)

    assert len(vals) == len(target_vals), \
        "Current vars and target vars must have the same length."

    for var, var_target in zip(vals, target_vals):
        expression.append(var_target.assign(polyak * var_target + (1.0 - polyak) * var))

    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


def get_action_dim(act_space):
    """
    Assume continuous action space (gym.spaces.Box).
    """
    if hasattr(act_space, "shape") and act_space.shape is not None:
        return int(np.prod(act_space.shape))
    raise ValueError("FG-MAD3PG trainer assumes continuous Box action spaces.")


def mlp_model(input_tensor, num_outputs, scope, reuse=False, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        out = input_tensor
        out = tf.layers.dense(out, num_units, activation=tf.nn.relu, name="fc1")
        out = tf.layers.dense(out, num_units, activation=tf.nn.relu, name="fc2")
        out = tf.layers.dense(out, num_outputs, activation=None, name="output")
        return out


def shared_tuple_embedding(x, scope, num_units=64):
    """
    Shared embedding network for (o, a) tuples.
    Uses AUTO_REUSE so neighbors share the same embedding weights.
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        h = tf.layers.dense(x, num_units, activation=tf.nn.relu, name="fc1")
        h = tf.layers.dense(h, num_units, activation=tf.nn.relu, name="fc2")
        return h


def attention_context(obs_ph_n, act_ph_n, agent_index,
                      neighbor_indices=None,
                      scope="critic_attention",
                      emb_dim=64,
                      key_dim=32):
    """
    Compute attention-based neighbor context h_i following the paper:
        e_i = f_self(o_i, a_i)
        e_j = f_nbr(o_j, a_j)
        alpha_ij = <W_q e_i, W_k e_j> / sqrt(d_k)
        beta_ij = softmax(alpha_ij)
        h_i = sum_j beta_ij W_v e_j

    Returns:
        context: [batch_size, key_dim]
        attn_weights: [batch_size, num_neighbors]
    """
    n = len(obs_ph_n)
    if neighbor_indices is None:
        neighbor_indices = [j for j in range(n) if j != agent_index]

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        self_tuple = tf.concat([obs_ph_n[agent_index], act_ph_n[agent_index]], axis=1)
        self_emb = shared_tuple_embedding(self_tuple, scope="self_embed", num_units=emb_dim)

        query = tf.layers.dense(self_emb, key_dim, activation=None, name="query")

        score_list = []
        value_list = []

        for j in neighbor_indices:
            nbr_tuple = tf.concat([obs_ph_n[j], act_ph_n[j]], axis=1)
            nbr_emb = shared_tuple_embedding(nbr_tuple, scope="neighbor_embed", num_units=emb_dim)

            key = tf.layers.dense(nbr_emb, key_dim, activation=None, name="key")
            value = tf.layers.dense(nbr_emb, key_dim, activation=None, name="value")

            score = tf.reduce_sum(query * key, axis=1, keepdims=True) / np.sqrt(float(key_dim))
            score_list.append(score)
            value_list.append(value)

        if len(score_list) == 0:
            batch_size = tf.shape(obs_ph_n[agent_index])[0]
            context = tf.zeros([batch_size, key_dim], dtype=tf.float32)
            attn_weights = tf.zeros([batch_size, 0], dtype=tf.float32)
        else:
            scores = tf.concat(score_list, axis=1)                       # [B, K]
            attn_weights = tf.nn.softmax(scores, axis=1)                # [B, K]
            values = tf.stack(value_list, axis=1)                       # [B, K, d_k]
            context = tf.reduce_sum(tf.expand_dims(attn_weights, -1) * values, axis=1)

        return context, attn_weights


def build_critic_input(obs_ph_n, act_ph_n, q_index,
                       neighbor_indices=None,
                       scope="critic_attention",
                       emb_dim=64,
                       key_dim=32):
    """
    Critic input in the paper:
        [o_i, a_i, h_i]
    """
    local_pair = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], axis=1)
    context, attn_weights = attention_context(
        obs_ph_n=obs_ph_n,
        act_ph_n=act_ph_n,
        agent_index=q_index,
        neighbor_indices=neighbor_indices,
        scope=scope,
        emb_dim=emb_dim,
        key_dim=key_dim
    )
    critic_input = tf.concat([local_pair, context], axis=1)
    return critic_input, attn_weights


def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer,
            grad_norm_clipping=None, num_units=64, tau=1e-2,
            neighbor_indices=None, attn_emb_dim=64, attn_key_dim=32,
            scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        obs_ph_n = make_obs_ph_n
        act_dim_n = [get_action_dim(act_space) for act_space in act_space_n]
        act_ph_n = [
            tf.placeholder(tf.float32, [None, act_dim_n[i]], name="action{}".format(i))
            for i in range(len(act_space_n))
        ]

        # Actor: a_i = mu_i(o_i)
        p_input = obs_ph_n[p_index]
        p_raw = p_func(
            p_input,
            act_dim_n[p_index],
            scope="p_func",
            num_units=num_units
        )
        act_sample = tf.tanh(p_raw)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # Small regularization on action magnitude
        p_reg = tf.reduce_mean(tf.square(act_sample))

        # Build policy gradient through critic Q_i(o_i, a_i, h_i)
        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_sample

        q_input, attn_weights = build_critic_input(
            obs_ph_n=obs_ph_n,
            act_ph_n=act_input_n,
            q_index=p_index,
            neighbor_indices=neighbor_indices,
            scope="critic_attention",
            emb_dim=attn_emb_dim,
            key_dim=attn_key_dim
        )

        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:, 0]
        pg_loss = -tf.reduce_mean(q)
        loss = pg_loss + 1e-3 * p_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], act_sample)

        # Target actor
        target_p_raw = p_func(
            obs_ph_n[p_index],
            act_dim_n[p_index],
            scope="target_p_func",
            num_units=num_units
        )
        target_act_sample = tf.tanh(target_p_raw)

        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars, tau=tau)

        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {
            'p_values': p_values,
            'target_act': target_act,
            'attn_weights': U.function(inputs=obs_ph_n + act_ph_n, outputs=attn_weights)
        }


def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer,
            grad_norm_clipping=None, num_units=64, tau=1e-2,
            neighbor_indices=None, attn_emb_dim=64, attn_key_dim=32,
            scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        obs_ph_n = make_obs_ph_n
        act_dim_n = [get_action_dim(act_space) for act_space in act_space_n]
        act_ph_n = [
            tf.placeholder(tf.float32, [None, act_dim_n[i]], name="action{}".format(i))
            for i in range(len(act_space_n))
        ]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        # Current critic input: [o_i, a_i, h_i]
        q_input, attn_weights = build_critic_input(
            obs_ph_n=obs_ph_n,
            act_ph_n=act_ph_n,
            q_index=q_index,
            neighbor_indices=neighbor_indices,
            scope="critic_attention",
            emb_dim=attn_emb_dim,
            key_dim=attn_key_dim
        )

        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:, 0]

        critic_attn_vars = U.scope_vars(U.absolute_scope_name("critic_attention"))
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        critic_vars = critic_attn_vars + q_func_vars

        q_loss = tf.reduce_mean(tf.square(q - target_ph))
        loss = q_loss

        optimize_expr = U.minimize_and_clip(optimizer, loss, critic_vars, grad_norm_clipping)

        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # Target critic
        target_q_input, target_attn_weights = build_critic_input(
            obs_ph_n=obs_ph_n,
            act_ph_n=act_ph_n,
            q_index=q_index,
            neighbor_indices=neighbor_indices,
            scope="target_critic_attention",
            emb_dim=attn_emb_dim,
            key_dim=attn_key_dim
        )

        target_q = q_func(target_q_input, 1, scope="target_q_func", num_units=num_units)[:, 0]

        target_critic_attn_vars = U.scope_vars(U.absolute_scope_name("target_critic_attention"))
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        target_critic_vars = target_critic_attn_vars + target_q_func_vars

        update_target_q = make_update_exp(critic_vars, target_critic_vars, tau=tau)
        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {
            'q_values': q_values,
            'target_q_values': target_q_values,
            'attn_weights': U.function(inputs=obs_ph_n + act_ph_n, outputs=attn_weights),
            'target_attn_weights': U.function(inputs=obs_ph_n + act_ph_n, outputs=target_attn_weights)
        }


class FGMAD3PGAgentTrainer(AgentTrainer):
    """
    FlipIt Game-based Multi-Agent Attention Distributed DDPG trainer.
    This trainer implements the learning component of FG-MAD3PG.

    Notes:
    - FlipIt-style reward shaping should be implemented in the environment.
    - LIME explanation should be implemented offline after training.
    """
    def __init__(self, name, p_model, q_model, obs_shape_n, obs_space_n,
                 act_space_n, agent_index, args,
                 neighbor_indices_n=None,
                 ADV=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.ADV = ADV

        # Neighborhood N_i in the paper
        if neighbor_indices_n is None:
            self.neighbor_indices = [j for j in range(self.n) if j != self.agent_index]
        else:
            self.neighbor_indices = neighbor_indices_n[self.agent_index]

        obs_ph_n = [
            U.BatchInput(obs_shape_n[i], name="observation{}".format(i)).get()
            for i in range(self.n)
        ]

        lr = getattr(args, "lr_adv", args.lr) if self.ADV else args.lr
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        tau = getattr(args, "tau", 1e-2)
        attn_emb_dim = getattr(args, "attn_emb_dim", args.num_units)
        attn_key_dim = getattr(args, "attn_key_dim", max(16, args.num_units // 2))

        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=q_model,
            optimizer=optimizer,
            grad_norm_clipping=0.5,
            num_units=args.num_units,
            tau=tau,
            neighbor_indices=self.neighbor_indices,
            attn_emb_dim=attn_emb_dim,
            attn_key_dim=attn_key_dim
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
            num_units=args.num_units,
            tau=tau,
            neighbor_indices=self.neighbor_indices,
            attn_emb_dim=attn_emb_dim,
            attn_key_dim=attn_key_dim
        )

        self.replay_buffer = ReplayBuffer(int(1e6))

        # Algorithm 1 in paper: only start updating when buffer is sufficiently populated
        self.min_replay_buffer_len = int(
            getattr(args, "buffer_min_size", args.batch_size * args.max_episode_len)
        )
        self.update_interval = int(getattr(args, "update_interval", 100))
        self.replay_sample_index = None

    def action(self, obs, add_noise=True):
        """
        Deterministic actor output + Gaussian exploration noise
        """
        action = self.act(obs[None])[0]

        noise_std = getattr(self.args, 'noise_std', 0.0)
        if add_noise and noise_std > 0.0:
            action = action + noise_std * np.random.randn(*action.shape)

        return np.clip(action, -1.0, 1.0)

    def experience(self, obs, act, rew, new_obs, done, terminal=False):
        effective_done = bool(done) or bool(terminal)
        self.replay_buffer.add(obs, act, rew, new_obs, float(effective_done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.min_replay_buffer_len:
            return
        if t % self.update_interval != 0:
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        index = self.replay_sample_index

        obs_n, obs_next_n, act_n = [], [], []

        for i in range(self.n):
            obs_i, act_i, rew_i, obs_next_i, done_i = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs_i)
            obs_next_n.append(obs_next_i)
            act_n.append(act_i)

        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # Target actions from target actors
        target_act_next_n = [
            agents[i].p_debug['target_act'](obs_next_n[i])
            for i in range(self.n)
        ]

        # Target critic Q'(o', a')
        target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))

        adjusted_rew = -rew if self.ADV else rew
        adjusted_rew = np.reshape(adjusted_rew, target_q_next.shape)
        done = np.reshape(done, target_q_next.shape)

        target_q = adjusted_rew + self.args.gamma * (1.0 - done) * target_q_next

        # Update critic
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # Update actor
        p_loss = self.p_train(*(obs_n + act_n))

        # Soft update target networks
        self.p_update()
        self.q_update()

        return [
            q_loss,
            p_loss,
            np.mean(target_q),
            np.mean(rew),
            np.mean(target_q_next),
            np.std(target_q)
        ]
