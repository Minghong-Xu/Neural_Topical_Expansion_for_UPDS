# coding=utf-8
# 在GRUCell的基础上实现自己的cell
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
import math
import numpy as np


# # TODO(ebrevdo): Remove once _linear is fully deprecated.
# Linear = tf.contrib.rnn.python.ops.core_rnn_cell_impl._Linear
# Linear = rnn_cell_impl._Linear  # pylint: disable=protected-access,invalid-name
from tensorflow.python.util import nest
from tensorflow.python.ops import nn_ops
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
class Linear(object):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of weight variable.
    dtype: data type for variables.
    build_bias: boolean, whether to build a bias variable.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.

  Raises:
    ValueError: if inputs_shape is wrong.
  """

  def __init__(self,
               args,
               output_size,
               build_bias,
               bias_initializer=None,
               kernel_initializer=None):
    self._build_bias = build_bias

    if args is None or (nest.is_sequence(args) and not args):
      raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
      args = [args]
      self._is_sequence = False
    else:
      self._is_sequence = True

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
      if shape.ndims != 2:
        raise ValueError("linear is expecting 2D arguments: %s" % shapes)
      if shape[1].value is None:
        raise ValueError("linear expects shape[1] to be provided for shape %s, "
                         "but saw %s" % (shape, shape[1]))
      else:
        total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
      self._weights = vs.get_variable(
          _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
          dtype=dtype,
          initializer=kernel_initializer)
      if build_bias:
        with vs.variable_scope(outer_scope) as inner_scope:
          inner_scope.set_partitioner(None)
          if bias_initializer is None:
            bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
          self._biases = vs.get_variable(
              _BIAS_VARIABLE_NAME, [output_size],
              dtype=dtype,
              initializer=bias_initializer)

  def __call__(self, args):
    if not self._is_sequence:
      args = [args]

    if len(args) == 1:
      res = math_ops.matmul(args[0], self._weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), self._weights)
    if self._build_bias:
      res = nn_ops.bias_add(res, self._biases)
    return res


def random_weight(dim_in, dim_out, name=None):
    return tf.get_variable(dtype=tf.float32, name=name, shape=[dim_in, dim_out], initializer=tf.contrib.layers.xavier_initializer(uniform=False))


def random_bias(dim, name=None):
    # return tf.get_variable(dtype=tf.float32, name=name, shape=[dim], initializer=tf.constant_initializer(1.0))
    return tf.get_variable(dtype=tf.float32, name=name, shape=[dim], initializer=tf.truncated_normal_initializer())


def mat_weight_mul(mat, weight):
    # [batch_size, n, m] * [m, p] = [batch_size, n, p]
    # 用.as_list()得到size
    mat_shape = mat.get_shape().as_list()
    weight_shape = weight.get_shape().as_list()
    # mat_shape的最后一列数和weight_shape的第一行数相同（判断可运算）
    assert (mat_shape[-1] == weight_shape[0])
    mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]])  # [batch_size * n, m]
    mul = tf.matmul(mat_reshape, weight)  # [batch_size * n, p]
    return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])


class MyCell1(tf.contrib.rnn.RNNCell):
    # 对所有context 做attention
    # 对两种persona做相互影响的multi-hop attention
    def __init__(self,
                 num_units,
                 persona_attention_state,
                 att_persona_mask_inf,
                 message_attention_state,
                 att_message_mask_inf,
                 encoder_vector,
                 explored_persona_attention_state,
                 vocab_size=None,
                 outputs_size=None,
                 args=None,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(MyCell1, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self.persona_attention_state = persona_attention_state
        self.att_persona_mask_inf = att_persona_mask_inf
        self.message_attention_state = message_attention_state
        self.att_message_mask_inf = att_message_mask_inf
        self.encoder_vector = encoder_vector
        self.explored_persona_attention_state = explored_persona_attention_state
        self._activation = activation or math_ops.tanh
        self.vocab_size = vocab_size
        self.outputs_size = outputs_size
        self.args = args
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

        #---------------------------------------
        # 参数设置
        # --------------------------------------
        # ---------------encoder_vector-------------
        self.encoder_vector_size = self.encoder_vector.shape[1].value

        # ----------------persona-------------------
        self.persona_length = self.persona_attention_state.shape[1].value
        self.persona_size = self.persona_attention_state.shape[2].value
        self.explored_persona_length = self.explored_persona_attention_state.shape[1].value
        self.explored_persona_size = self.explored_persona_attention_state.shape[2].value

        self.persona_memory_size = 2*num_units

        self.W_p_key = random_weight(self.persona_size, self.persona_memory_size, name='W_p_key')
        self.W_p_value = random_weight(self.persona_size, self.persona_memory_size, name='W_p_value')
        self.W_ep_key = random_weight(self.explored_persona_size, self.persona_memory_size, name='W_ep_key')
        self.W_ep_value = random_weight(self.explored_persona_size, self.persona_memory_size, name='W_ep_value')

        self.persona_key = mat_weight_mul(self.persona_attention_state, self.W_p_key)
        self.persona_value = mat_weight_mul(self.persona_attention_state, self.W_p_value)
        self.explored_persona_key = mat_weight_mul(self.explored_persona_attention_state, self.W_ep_key)
        self.explored_persona_value = mat_weight_mul(self.explored_persona_attention_state, self.W_ep_value)

        self.W_q_p = random_weight(num_units + self.encoder_vector_size, self.persona_memory_size, name='W_q_persona')

        # -------------message_attention-------------
        self.message_attention_length = self.message_attention_state.shape[1].value
        # attention_state的最后一维同时作为attention_size
        self.message_attention_size = self.message_attention_state.shape[2].value

        self.W_q_m = random_weight(num_units + self.encoder_vector_size, self.message_attention_size, name='W_q_message')
        self.W_a_m = random_weight(self.message_attention_size, self.message_attention_size, name='W_a_message')
        self.B_v_m = tf.get_variable(dtype=tf.float32, name='B_v_message', shape=[self.message_attention_size, 1], initializer=tf.truncated_normal_initializer())

    @property
    def state_size(self):
        return self._num_units

    @property
    # 自己定义输出的size
    def output_size(self):
        if self.outputs_size is None:
            self.outputs_size = self._num_units + 2 * self._num_units + self.message_attention_size
        return self.outputs_size

    def call(self, inputs, state_):
        """自己实现，包含attention的处理，以及两个预测分布相加"""
        state, state = self.gru_call(inputs, state_)
        with vs.variable_scope("message_attention", reuse=tf.AUTO_REUSE):
            att_message = self.Message_attention(state)
            # print("att_message:", att_message)
        with vs.variable_scope("persona_attention", reuse=tf.AUTO_REUSE):
            att_persona = self.persona_attention(state)
            # print("att_persona:", att_persona)

        output = tf.concat([state, att_persona, att_message], 1)
        return output, state

    def gru_call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates", reuse=tf.AUTO_REUSE):  # Reset gate and update gate.
                self._gate_linear = Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        new_h = u * state + (1 - u) * c
        return new_h, new_h

    def Message_attention(self, query):
        # query 和 att_question 一起做attention
        query_ = tf.concat([query, self.encoder_vector], 1)
        Wq_Q = tf.matmul(query_, self.W_q_m)     # [b, h]
        Wa_A = mat_weight_mul(self.message_attention_state, self.W_a_m)       # [b, len, h]
        tanh = tf.tanh(tf.expand_dims(Wq_Q, 1) + Wa_A)      # [b, len, h]
        # print("tanh:", tanh)
        s = tf.squeeze(mat_weight_mul(tanh, self.B_v_m))   # [b, len]
        # print("s:", s)
        s = s + self.att_message_mask_inf
        a_t = tf.nn.softmax(s)
        # print("a_t:", a_t)
        # a_t_extend = tf.concat([tf.reshape(a_t, [-1, self.message_attention_length, 1])] * self.message_attention_size, 2)
        state = tf.reduce_sum(tf.multiply(tf.expand_dims(a_t, 2), self.message_attention_state), 1)
        # print("state:", state)  # [batch_size, hide_size]
        return state

    def persona_attention(self, query_):
        # query 和 att_question 一起做attention
        query_ = tf.concat([query_, self.encoder_vector], 1)
        query = tf.matmul(query_, self.W_q_p)     # [b, h]
        for i in range(5):
            # -----对persona---------------
            s_p = tf.reduce_sum(tf.multiply(tf.expand_dims(query, 1), self.persona_key), 2)  # [batch_size, len]
            s_p = s_p + self.att_persona_mask_inf
            a_t_p = tf.nn.softmax(s_p)
            # print("a_t:", a_t)
            # a_t_extend = tf.concat([tf.expand_dims(a_t, -1)] * (self._num_units * 2), 2)
            v_P = tf.reduce_sum(tf.multiply(tf.expand_dims(a_t_p, 2), self.persona_value), 1)
            # print("v_P:", v_P)  # [batch_size, hiden_size]

            # -----对explored persona---------------
            s_ep = tf.reduce_sum(tf.multiply(tf.expand_dims(query, 1), self.explored_persona_key), 2)  # [batch_size, len]
            # mask 部分将s设置为-inf，所以softmax的时候为0
            a_t_ep = tf.nn.softmax(s_ep)
            # print("a_t:", a_t)
            v_EP = tf.reduce_sum(tf.multiply(tf.expand_dims(a_t_ep, 2), self.explored_persona_value), 1)
            # print("v_EP:", v_EP)  # [batch_size, hiden_size]

            # 下一步的query
            query = query + v_P + v_EP
        # state = tf.concat([v_P, v_EP], -1)
        # return v_P + v_EP
        # return state
        return query


class MyCell2(tf.contrib.rnn.RNNCell):
    # 对所有context 做attention
    # 对pre-defined persona做multi-hop attention
    def __init__(self,
                 num_units,
                 persona_attention_state,
                 att_persona_mask_inf,
                 message_attention_state,
                 att_message_mask_inf,
                 encoder_vector,
                 explored_persona_attention_state,
                 vocab_size=None,
                 outputs_size=None,
                 args=None,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(MyCell2, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self.persona_attention_state = persona_attention_state
        self.att_persona_mask_inf = att_persona_mask_inf
        self.message_attention_state = message_attention_state
        self.att_message_mask_inf = att_message_mask_inf
        self.encoder_vector = encoder_vector
        self.explored_persona_attention_state = explored_persona_attention_state
        self._activation = activation or math_ops.tanh
        self.vocab_size = vocab_size
        self.outputs_size = outputs_size
        self.args = args
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

        #---------------------------------------
        # 参数设置
        # --------------------------------------
        # ---------------encoder_vector-------------
        self.encoder_vector_size = self.encoder_vector.shape[1].value

        # ----------------persona-------------------
        self.persona_length = self.persona_attention_state.shape[1].value
        self.persona_size = self.persona_attention_state.shape[2].value
        self.explored_persona_length = self.explored_persona_attention_state.shape[1].value
        self.explored_persona_size = self.explored_persona_attention_state.shape[2].value

        self.persona_memory_size = 2*num_units

        self.W_p_key = random_weight(self.persona_size, self.persona_memory_size, name='W_p_key')
        self.W_p_value = random_weight(self.persona_size, self.persona_memory_size, name='W_p_value')
        self.W_ep_key = random_weight(self.explored_persona_size, self.persona_memory_size, name='W_ep_key')
        self.W_ep_value = random_weight(self.explored_persona_size, self.persona_memory_size, name='W_ep_value')

        self.persona_key = mat_weight_mul(self.persona_attention_state, self.W_p_key)
        self.persona_value = mat_weight_mul(self.persona_attention_state, self.W_p_value)
        self.explored_persona_key = mat_weight_mul(self.explored_persona_attention_state, self.W_ep_key)
        self.explored_persona_value = mat_weight_mul(self.explored_persona_attention_state, self.W_ep_value)

        self.W_q_p = random_weight(num_units + self.encoder_vector_size, self.persona_memory_size, name='W_q_persona')

        # -------------message_attention-------------
        self.message_attention_length = self.message_attention_state.shape[1].value
        # attention_state的最后一维同时作为attention_size
        self.message_attention_size = self.message_attention_state.shape[2].value

        self.W_q_m = random_weight(num_units + self.encoder_vector_size, self.message_attention_size, name='W_q_message')
        self.W_a_m = random_weight(self.message_attention_size, self.message_attention_size, name='W_a_message')
        self.B_v_m = tf.get_variable(dtype=tf.float32, name='B_v_message', shape=[self.message_attention_size, 1], initializer=tf.truncated_normal_initializer())

    @property
    def state_size(self):
        return self._num_units

    @property
    # 自己定义输出的size
    def output_size(self):
        if self.outputs_size is None:
            self.outputs_size = self._num_units + 2 * self._num_units + self.message_attention_size
        return self.outputs_size

    def call(self, inputs, state_):
        """自己实现，包含attention的处理，以及两个预测分布相加"""
        state, state = self.gru_call(inputs, state_)
        with vs.variable_scope("message_attention", reuse=tf.AUTO_REUSE):
            att_message = self.Message_attention(state)
            # print("att_message:", att_message)
        with vs.variable_scope("persona_attention", reuse=tf.AUTO_REUSE):
            att_persona = self.persona_attention(state)
            # print("att_persona:", att_persona)

        output = tf.concat([state, att_persona, att_message], 1)
        return output, state

    def gru_call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates", reuse=tf.AUTO_REUSE):  # Reset gate and update gate.
                self._gate_linear = Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        new_h = u * state + (1 - u) * c
        return new_h, new_h

    def Message_attention(self, query):
        # query 和 att_question 一起做attention
        query_ = tf.concat([query, self.encoder_vector], 1)
        Wq_Q = tf.matmul(query_, self.W_q_m)     # [b, h]
        Wa_A = mat_weight_mul(self.message_attention_state, self.W_a_m)       # [b, len, h]
        tanh = tf.tanh(tf.expand_dims(Wq_Q, 1) + Wa_A)      # [b, len, h]
        # print("tanh:", tanh)
        s = tf.squeeze(mat_weight_mul(tanh, self.B_v_m))   # [b, len]
        # print("s:", s)
        s = s + self.att_message_mask_inf
        a_t = tf.nn.softmax(s)
        # print("a_t:", a_t)
        # a_t_extend = tf.concat([tf.reshape(a_t, [-1, self.message_attention_length, 1])] * self.message_attention_size, 2)
        state = tf.reduce_sum(tf.multiply(tf.expand_dims(a_t, 2), self.message_attention_state), 1)
        # print("state:", state)  # [batch_size, hide_size]
        return state

    def persona_attention(self, query_):
        # query 和 att_question 一起做attention
        query_ = tf.concat([query_, self.encoder_vector], 1)
        query = tf.matmul(query_, self.W_q_p)     # [b, h]
        v_P = 0
        for i in range(3):
            # -----对persona---------------
            s_p = tf.reduce_sum(tf.multiply(tf.expand_dims(query, 1), self.persona_key), 2)  # [batch_size, len]
            s_p = s_p + self.att_persona_mask_inf
            a_t_p = tf.nn.softmax(s_p)
            # print("a_t:", a_t)
            # a_t_extend = tf.concat([tf.expand_dims(a_t, -1)] * (self._num_units * 2), 2)
            v_P = tf.reduce_sum(tf.multiply(tf.expand_dims(a_t_p, 2), self.persona_value), 1)
            # print("v_P:", v_P)  # [batch_size, hiden_size]

            # 下一步的query
            query = query + v_P
        return v_P

