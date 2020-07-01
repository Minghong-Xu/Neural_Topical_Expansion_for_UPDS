# 2019-08-15
# AAAI 尝试新模型
# context 对persona一次做信息检索，用memory；decoder中对两种persona的attention有相互作用


from args_8 import args
import tensorflow as tf
# from Mycell_8 import MyCell1 as MyCell
from Mycell_8 import MyCell2 as MyCell      # 不加扩展词
import math
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import embedding_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.rnn import dynamic_rnn, static_rnn
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import h5py
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#默认为0：输出所有log信息
#设置为1：进一步屏蔽INFO信息
#设置为2：进一步屏蔽WARNING信息
#设置为3：进一步屏蔽ERROR信息

eps = 1e-30
tf.set_random_seed(1234)


class Model1:
    def __init__(self, type, training_steps_per_epoch, vocabSize):
        self.type = type
        self.args = args(type)

        print("build model1")
        print("vocabSize", vocabSize)

        # ==============================================================================
        #                               设置必要参数
        # ==============================================================================
        if(type == "train"):
            self.global_step = tf.Variable(0, trainable=False)
            # 设置学习速率
            self.learning_rate = tf.maximum(
                # 如果staircase=True，那就表明每training_steps_per_epoch次计算学习速率变化
                tf.train.exponential_decay(
                    self.args.learning_rate,
                    self.global_step,
                    training_steps_per_epoch,
                    self.args.learning_rate_decay_factor,
                    staircase=True),
                self.args.min_learning_rate)

        START = tf.constant(value=[self.args.GO] * self.args.batch_size)
        # ==============================================================================
        #                               define placeholder
        # ==============================================================================
        with tf.name_scope("placeholder"):
            self.personas_ph = tf.placeholder(tf.int32,
                                              shape=[self.args.batch_size, self.args.max_num_persona,
                                                     self.args.max_num_personalength], name="personas")
            self.personas_len_ph = tf.placeholder(tf.int32,
                                                  shape=[self.args.batch_size, self.args.max_num_persona],
                                                  name="persona_lengths")
            self.persona_turn = tf.placeholder(tf.int32,
                                                  shape=[self.args.batch_size], name="persona_turn")

            self.historys_ph = tf.placeholder(tf.int32,
                                              shape=[self.args.batch_size, 2*self.args.max_num_history_turns+1,
                                                     self.args.max_num_Qlength], name="historys")
            self.historys_len_ph = tf.placeholder(tf.int32,
                                           shape=[self.args.batch_size, 2*self.args.max_num_history_turns+1],
                                                  name="history_lengths")
            self.historys_turn = tf.placeholder(tf.int32,
                                                  shape=[self.args.batch_size], name="history_turn")

            # self.questions_ph = tf.placeholder(tf.int32,
            #                                    shape=[self.args.batch_size,self.args.max_num_Qlength], name="questions")
            # self.question_len_ph = tf.placeholder(tf.int32, shape=[self.args.batch_size], name="question_lengths")

            self.answers_ph = tf.placeholder(tf.int32,
                                             shape=[self.args.batch_size, self.args.max_num_Alength], name="answers")
            self.answer_len_ph = tf.placeholder(tf.int32, shape=[self.args.batch_size], name="answer_lengths")
            self.answer_targets_ph = tf.placeholder(tf.int32,
                                                shape=[self.args.batch_size, self.args.max_num_Alength + 1],
                                                name="answer_targets")

        personas_turn_mask = tf.sequence_mask(self.persona_turn, self.args.max_num_persona)
        self.att_persona_sentence_mask = tf.cast(personas_turn_mask, dtype=tf.float32)
        print("self.att_persona_sentence_mask :", self.att_persona_sentence_mask)
        personas_len_mask = tf.sequence_mask(self.personas_len_ph, self.args.max_num_personalength)
        personas_len_mask = tf.reshape(personas_len_mask, [self.args.batch_size, -1])
        self.att_persona_mask = tf.cast(personas_len_mask, dtype=tf.float32)
        print("self.att_persona_mask:", self.att_persona_mask)
        historys_len_mask = tf.sequence_mask(self.historys_len_ph, self.args.max_num_Qlength)
        historys_len_mask = tf.reshape(historys_len_mask, [self.args.batch_size, -1])
        self.att_message_mask = tf.cast(historys_len_mask, dtype=tf.float32)
        print("self.att_message_mask:", self.att_message_mask)  # 输出测试正确
        # question_len_mask = tf.sequence_mask(self.question_len_ph, self.args.max_num_Qlength)
        # question_len_mask = tf.reshape(question_len_mask, [self.args.batch_size, -1])
        # self.att_message_mask = tf.concat([historys_len_mask, question_len_mask], 1)      # 不理解当时为什么queation加上，不是history中有question吗？
        # self.att_message_mask = tf.cast(self.att_message_mask, dtype=tf.float32)
        # print("self.att_message_mask:", self.att_message_mask)  # 输出测试正确

        # 因为结尾加入EOS,长度都加1
        self.answer_len_ph_ = self.answer_len_ph + 1

        # ---------------------------------------------------
        # 加入DB 多引入的输入
        # ---------------------------------------------------
        # self.personas_emb_ph = tf.placeholder(tf.float32, shape=[self.args.batch_size, self.args.max_num_persona,
        #                                                          self.args.num_topics], name="personas_emb")
        # self.historys_emb_ph = tf.placeholder(tf.float32,
        #                                       shape=[self.args.batch_size, 2 * self.args.max_num_history_turns + 1,
        #                                              self.args.num_topics], name="historys_emb")
        # self.questions_emb_ph = tf.placeholder(tf.float32, shape=[self.args.batch_size,
        #                                                           self.args.num_topics], name="questions_emb")

        self.topic_words_emb_ph = tf.placeholder(tf.float32, shape=[self.args.batch_size, self.args.num_topic_words,
                                                                    self.args.num_topics], name="topic_words_emb")
        # self.topic_words_index = tf.placeholder(tf.int32, shape=[self.args.batch_size, self.args.num_topic_words], name="topic_words_index")
        # self.batch_OOV_num = tf.placeholder(tf.int32, shape=[], name="batch_OOV_num")

        # ---------------------------------------------------
        # 加入bow-loss  多引入的输入
        # ---------------------------------------------------
        self.answers_in_persona_label = tf.placeholder(tf.int32, shape=[self.args.batch_size, vocabSize],
                                                       name="answers_in_persona_label")  # 0/1 标签
        answers_in_persona_label = tf.cast(self.answers_in_persona_label, tf.float32)

        # ---------------------------------------------------
        # 加上 persona attention的label 多引入的输入
        # ---------------------------------------------------
        self.answer_attention_ph = tf.placeholder(tf.float32,
                                                  shape=[self.args.batch_size, self.args.max_num_persona],
                                                  name="answer_attention")
        # self.answer_attention_ph = tf.placeholder(tf.int32,
        #                                           shape=[self.args.batch_size, self.args.max_num_persona],
        #                                           name="answer_attention")

        # ==============================================================================
        # Embedding (share) and other variable
        # ==============================================================================
        with ops.device("/cpu:0"):
            if variable_scope.get_variable_scope().initializer:
                # 变量已存在
                initializer = variable_scope.get_variable_scope().initializer
            else:
                # Default initializer for embeddings should have variance=1. 初始化
                # sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
                # # 生成均匀分布的随机数
                # initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)
                File = h5py.File("../Data/glove_train.h5", 'r')
                initializer = np.array(File["embedding"])

            # embedding = variable_scope.get_variable(name="embedding", shape=[vocabSize, self.args.embedSize], initializer=initializer, dtype=tf.float32)
            embedding = variable_scope.get_variable(name="embedding", initializer=initializer, dtype=tf.float32)

            # Weights
            # self.W_eQ = self.random_weight(self.args.rnnHiddenSize * 2, self.args.rnnHiddenSize * 2,
            #                                name='W_eQ')
            # self.W_eP = self.random_weight(self.args.rnnHiddenSize * 2, self.args.rnnHiddenSize * 2,
            #                                name='W_eP')

            self.W_p_key = self.random_weight(self.args.rnnHiddenSize * 2, self.args.rnnHiddenSize * 2,
                                              name="W_p_key")
            self.W_p_value = self.random_weight(self.args.rnnHiddenSize * 2, self.args.rnnHiddenSize * 2,
                                                name="W_p_value")

            # Biases
            # self.B_v_P = self.random_bias(self.args.rnnHiddenSize * 2, name='B_v_P')

            # Lambde loss 比例参数
            # self.lam = tf.Variable(tf.random_normal(1), name="lam")

        START_EMB = embedding_ops.embedding_lookup(embedding, START)

        # ==============================================================================
        # split placeholders and embed
        # ==============================================================================
        personas = embedding_ops.embedding_lookup(embedding, self.personas_ph)
        personas = tf.transpose(personas, [1, 0, 2, 3])
        personas_lengths = tf.transpose(self.personas_len_ph, [1, 0])
        historys = embedding_ops.embedding_lookup(embedding, self.historys_ph)
        historys = tf.transpose(historys, [1, 0, 2, 3])
        historys_lengths = tf.transpose(self.historys_len_ph, [1, 0])
        # questions = embedding_ops.embedding_lookup(embedding, self.questions_ph)
        answers = embedding_ops.embedding_lookup(embedding, self.answers_ph)

        # ==============================================================================
        # make RNN cell
        # ==============================================================================
        def single_cell(hidden_size, in_keep_prob):
            if self.args.use_lstm:
                # forget_bias就是LSTM们的忘记系数，如果等于1，就是不会忘记任何信息。
                cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
            else:
                cell = tf.contrib.rnn.GRUCell(hidden_size)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=in_keep_prob)
            return cell

        def make_cell(hidden_size, in_keep_prob):
            if self.args.rnnLayers > 1:
                return tf.contrib.rnn.MultiRNNCell(
                    [single_cell(hidden_size, in_keep_prob) for _ in range(hidden_size)])
            else:
                return single_cell(hidden_size, in_keep_prob)

        fw_encoder_cell_persona = make_cell(self.args.rnnHiddenSize, self.args.keep_prob)
        bw_encoder_cell_persona = make_cell(self.args.rnnHiddenSize, self.args.keep_prob)
        fw_encoder_cell_history_1 = make_cell(self.args.rnnHiddenSize, self.args.keep_prob)
        bw_encoder_cell_history_1 = make_cell(self.args.rnnHiddenSize, self.args.keep_prob)
        fw_encoder_cell_history_2 = make_cell(self.args.rnnHiddenSize, self.args.keep_prob)
        bw_encoder_cell_history_2 = make_cell(self.args.rnnHiddenSize, self.args.keep_prob)
        # fw_encoder_cell_question = make_cell(self.args.rnnHiddenSize, self.args.keep_prob)
        # bw_encoder_cell_question = make_cell(self.args.rnnHiddenSize, self.args.keep_prob)

        # 串联所有word level 的 hidden-state
        persona_word_enc = []
        message_word_enc = []

        # ==============================================================================
        # encode persona (直接串联)
        # ==============================================================================
        print("encode personas...")
        personas_enc = []
        for i in range(self.args.max_num_persona):
            with tf.variable_scope('persona_sentence_EncoderRNN', reuse=tf.AUTO_REUSE) as varscope:
                persona_sentence_Output, persona_sentence_State = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=fw_encoder_cell_persona, cell_bw=bw_encoder_cell_persona,
                    inputs=personas[i], sequence_length=personas_lengths[i],
                    dtype=tf.float32, scope=varscope)  # [batch_size, encoder_cell.state_size]
                persona_sentence_Output = tf.concat([persona_sentence_Output[0], persona_sentence_Output[1]], -1)
                persona_sentence_State = tf.concat([persona_sentence_State[0], persona_sentence_State[1]], -1)
            # print("persona_sentence_State.h:",persona_sentence_State.h)
            # 构建成下一个rnn 的输入形式
            persona_sentence_State = tf.reshape(persona_sentence_State, [self.args.batch_size, 1, -1])
            if i == 0:
                personas_enc = persona_sentence_State
                persona_word_enc = persona_sentence_Output
            else:
                personas_enc = tf.concat([personas_enc, persona_sentence_State], 1)  # sentenses memory
                persona_word_enc = tf.concat([persona_word_enc, persona_sentence_Output], 1)  # words memory

        print("personas_enc:", personas_enc)  # [batch_size, max_num_persona, hiddensize]
        print("persona_word_enc:", persona_word_enc)  # [batch_size, max_num_persona*persona_length, hiddensize]

        # ==============================================================================
        # encode history (HRED)
        # ==============================================================================
        print("encode history...")
        historys_sentence_enc = []
        for i in range(2 * self.args.max_num_history_turns + 1):
            with tf.variable_scope('history_sentence_EncoderRNN', reuse=tf.AUTO_REUSE) as varscope:
                history_sentence_Output, history_sentence_State = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=fw_encoder_cell_history_1, cell_bw=bw_encoder_cell_history_1,
                    inputs=historys[i], sequence_length=historys_lengths[i],
                    dtype=tf.float32, scope=varscope)  # [batch_size, encoder_cell.state_size]
                history_sentence_Output = tf.concat([history_sentence_Output[0], history_sentence_Output[1]], -1)
                history_sentence_State = tf.concat([history_sentence_State[0], history_sentence_State[1]], -1)
            # print("history_sentence_State:",history_sentence_State)
            # 构建成下一个rnn 的输入形式
            history_sentence_State = tf.reshape(history_sentence_State, [self.args.batch_size, 1, -1])
            if i == 0:
                historys_sentence_enc = history_sentence_State
                message_word_enc = history_sentence_Output
            else:
                historys_sentence_enc = tf.concat([historys_sentence_enc, history_sentence_State], 1)
                message_word_enc = tf.concat([message_word_enc, history_sentence_Output], 1)
        print("historys_sentence_enc:", historys_sentence_enc)  # [batch_size, h_turn, hidden_size*2]
        print("message_word_enc:", message_word_enc)
        with tf.variable_scope('history_sequence_EncoderRNN', reuse=tf.AUTO_REUSE) as varscope:
            history_sequence_Output, history_sequence_State = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_encoder_cell_history_2, cell_bw=bw_encoder_cell_history_2,
                inputs=historys_sentence_enc, sequence_length=self.historys_turn,
                dtype=tf.float32, scope=varscope)  # [batch_size, encoder_cell.state_size]
            history_sequence_Output = tf.concat([history_sequence_Output[0], history_sequence_Output[1]], -1)
            history_sequence_State = tf.concat([history_sequence_State[0], history_sequence_State[1]], -1)
        print("history_sequence_output:", history_sequence_Output)  # [batch_size, h_turn, hidden_size]
        history_sequence_Output = tf.transpose(history_sequence_Output, [1, 0, 2])
        print("history_sequence_output:", history_sequence_Output)  # [h_turn, batch_size, hidden_size]
        print("history_sequence_State:", history_sequence_State)  # [batch_size, hidden_size]

        # ====================================================
        # context 检索 persona sentense memory
        # ====================================================
        print("query -->  persona_swntensen-memory")
        personas_enc_key = tf.matmul(personas_enc, self.W_p_key)
        personas_enc_value = tf.matmul(personas_enc, self.W_p_value)
        persona_memory_enc = []
        persona_a_t_all = []

        query = 0
        for i in range(2 * self.args.max_num_history_turns + 1):
            query = query + history_sequence_Output[i]
            s = tf.reduce_sum(tf.multiply(tf.expand_dims(query, 1), personas_enc_key), 2)  # [batch_size, len]
            a_t = tf.nn.softmax(s)
            # print("a_t:", a_t)
            persona_a_t_all.append(a_t)
            v_P = tf.reduce_sum(tf.multiply(tf.expand_dims(a_t, -1), personas_enc_value), 1)
            # print("v_P:", v_P)  # [batch_size, hiden_size]
            # 下一步的query
            query = v_P
            # query = query + v_P
            persona_memory_enc.append(query)  # [len, batch, hiden_size]

        # 根据context长度选择最后memory结果
        a = tf.range(self.args.batch_size)
        b = self.historys_turn - 1
        index = tf.concat([tf.expand_dims(b, 1), tf.expand_dims(a, 1)], 1)
        print("index:", index)
        persona_memory = tf.gather_nd(persona_memory_enc, index)
        print("persona_memory:", persona_memory)
        self.persona_a_t = tf.gather_nd(persona_a_t_all, index)

        # ====================================================
        # 合并信息，得到s0
        # ====================================================
        encoder_State = tf.concat(values=[history_sequence_State, persona_memory],
                                  axis=1)  # [batch_size, (2*hidden_size)*2]
        encoder_State = tf.layers.dense(encoder_State, self.args.rnnHiddenSize)
        print("encoder_State:", encoder_State)

        # attention到句子
        persona_sentence_attention_State = personas_enc
        print("persona_sentence_attention_State:", persona_sentence_attention_State)
        # attention 到词
        persona_attention_State = persona_word_enc
        message_attention_State = message_word_enc
        print("persona_attention_State:", persona_attention_State)
        print("message_attention_State:", message_attention_State)

        # ==============================================================================
        # decode
        # ==============================================================================
        print("decode ...")
        with tf.variable_scope('DecoderRNN'):
            # att_persona_sentence_mask = self.att_persona_sentence_mask
            att_persona_mask = self.att_persona_mask
            att_message_mask = self.att_message_mask
            topic_words_emb_ph = self.topic_words_emb_ph
            encoder_State_s0 = encoder_State
            #     # self.topic_words_index 转成onehot， 为了扩展Pk的维度， 加到Pv上
            if (self.type != "train") and self.args.beam_search:
                # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
                print("use beamsearch decoding..  num_BeamSearch=", self.args.num_BeamSearch)
                # persona_sentence_attention_State = tf.contrib.seq2seq.tile_batch(persona_sentence_attention_State,
                #                                                                  multiplier=self.args.num_BeamSearch)
                # att_persona_sentence_mask = tf.contrib.seq2seq.tile_batch(self.att_persona_sentence_mask,
                #                                                           multiplier=self.args.num_BeamSearch)
                persona_attention_State = tf.contrib.seq2seq.tile_batch(persona_attention_State,
                                                                        multiplier=self.args.num_BeamSearch)
                att_persona_mask = tf.contrib.seq2seq.tile_batch(self.att_persona_mask,
                                                                 multiplier=self.args.num_BeamSearch)
                message_attention_State = tf.contrib.seq2seq.tile_batch(message_attention_State,
                                                                        multiplier=self.args.num_BeamSearch)
                att_message_mask = tf.contrib.seq2seq.tile_batch(self.att_message_mask,
                                                                 multiplier=self.args.num_BeamSearch)
                topic_words_emb_ph = tf.contrib.seq2seq.tile_batch(self.topic_words_emb_ph,
                                                                   multiplier=self.args.num_BeamSearch)
                encoder_State_s0 = tf.contrib.seq2seq.tile_batch(encoder_State, multiplier=self.args.num_BeamSearch)
                encoder_State = nest.map_structure(
                    lambda s: tf.contrib.seq2seq.tile_batch(s, self.args.num_BeamSearch),
                    encoder_State)
                # 如果使用beam_seach则batch_size = self.batch_size * self.beam_size。因为之前已经复制过一次

            # mask 部分将s设置为-inf，所以softmax的时候为0
            message_mask_inf = 1 - att_message_mask
            mask = np.zeros(att_message_mask.shape)
            for i in range(att_message_mask.shape[0]):
                for j in range(att_message_mask.shape[1]):
                    if message_mask_inf[i][j] == 1:
                        mask[i][j] = -np.inf
            att_message_mask_inf = message_mask_inf * mask

            persona_mask_inf = 1 - att_persona_mask
            mask = np.zeros(att_persona_mask.shape)
            for i in range(att_persona_mask.shape[0]):
                for j in range(att_persona_mask.shape[1]):
                    if persona_mask_inf[i][j] == 1:
                        mask[i][j] = -np.inf
            att_persona_mask_inf = persona_mask_inf * mask

            self.decoder_cell_ = MyCell(self.args.rnnHiddenSize, persona_attention_State, att_persona_mask_inf,
                                        message_attention_State, att_message_mask_inf, encoder_State_s0,
                                        topic_words_emb_ph)

            # ------------- dropout ------------------
            self.decoder_cell = tf.contrib.rnn.DropoutWrapper(self.decoder_cell_,
                                                              input_keep_prob=self.args.keep_prob)

            # train 和 test用的decoder不一样， 为了变量名对应
            if (self.type == "train"):
                output_layer = tf.compat.v1.layers.Dense(vocabSize,
                                                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                                            stddev=0.1),
                                                         name='decoder/dense')
            else:
                output_layer = tf.compat.v1.layers.Dense(vocabSize,
                                                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                                            stddev=0.1))

            if (self.type == "train"):
                # 定义decoder阶段的输入，其实就是在decoder的seq_outputs开始处添加一个<start>
                # seq_outputs的shape为[batch_size, decoder_targets_length, embedding_size]
                # seq_outputs 分解为 max_num_Alength 个tensor（batch_size, emb_size）
                answers = [tf.squeeze(input=word, axis=1) for word in
                           tf.split(value=answers, num_or_size_splits=self.args.max_num_Alength, axis=1)]
                # print("answer:",answer)
                answers = [START_EMB] + answers
                # answers = tf.transpose(answers, [1, 0, 2])  # [batch_size, A_length+1, embedding_size]
                print("answers:", answers)

                # 用dynamic_rnn进行decoder
                decoder_Outputs, decoder_State = static_rnn(cell=self.decoder_cell, inputs=answers,
                                                            initial_state=encoder_State,
                                                            sequence_length=self.answer_len_ph_,
                                                            dtype=tf.float32, scope="decoder")
                decoder_Outputs = tf.stack(decoder_Outputs, 1)
                print("decoder_Outputs:", decoder_Outputs)

                # ------------输出 persona_sentence_a_t----------------------
                # self.persona_sentence_a_t = decoder_Outputs[:, :, -self.args.max_num_persona:]
                # print("self.persona_sentence_a_t:", self.persona_sentence_a_t)
                # decoder_Outputs = decoder_Outputs[:, :, :-self.args.max_num_persona]
                # print("decoder_Outputs:", decoder_Outputs)

                # ------------输出 topic_a_t----------------------
                # self.topic_a_t = decoder_Outputs[:, :, -self.args.num_topic_words:]
                # print("self.topic_a_t:", self.topic_a_t)
                # decoder_Outputs = decoder_Outputs[:, :, :-self.args.num_topic_words]
                # print("decoder_Outputs:", decoder_Outputs)

                self.decoder_logits_train = output_layer(decoder_Outputs)  # [batch_size, A_len, vocab]
                print("self.decoder_logits_train:", self.decoder_logits_train)

                # 结果
                self.answers_predict = tf.argmax(self.decoder_logits_train, axis=-1, name='answers_predict')
                print("self.answers_predict:", self.answers_predict)

                # 根据输出计算loss和梯度，并定义进行更新的AdamOptimizer和train_op
                # mask标记是否为pad部分（1/0）
                mask = tf.cast(x=tf.not_equal(x=self.answer_targets_ph, y=self.args.PAD),
                               dtype=tf.float32)  # [batch_size, Alength+1]
                # 使用sequence_loss计算loss，这里需要传入之前定义的mask标志
                # 这里可能报错，是因为dynamic_decode的输出大小是动态的，可能造成数据shape不匹配
                self.loss1 = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                              targets=self.answer_targets_ph, weights=mask)

                self.ppl = tf.reduce_mean(tf.exp(self.loss1))

                # ----------------------------------------------------------------------------------------------
                #  2:loss2: label是answer中的所有词(加重persona中词的权重)  sigmoid
                # -----------------------------------------------------------------------------------------------
                print("bow-loss-sigmoid-weight")
                print("lamba_loss1:", self.args.lamba_loss1)
                bow_state = tf.reduce_sum(self.decoder_logits_train, 1)  # [batch_size, vocab]
                self.bow_prediction = tf.nn.sigmoid(bow_state)
                print("self.bow_prediction:", self.bow_prediction)  # [batch_size, vocab]

                target_one_hot_bow = tf.one_hot(indices=self.answer_targets_ph, depth=vocabSize,
                                                dtype=tf.float32)  # [batch_size, Alength+1, vocab]
                # target_bow = tf.reduce_sum(input_tensor=target_one_hot_bow, axis=1)     # [batch_size, vocab]   出现过累计计数，未出现计数0
                target_bow = tf.reduce_max(input_tensor=target_one_hot_bow,
                                           axis=1)  # [batch_size, vocab]   出现过计数1，未出现计数0
                # mask2 是为了去掉 pad、eos等
                m1 = [1.0 for _ in range(vocabSize - 4)]
                m2 = [0.0 for _ in range(4)]
                m3 = tf.reshape(tf.concat([m2, m1], 0), [1, -1])
                self.mask2 = tf.concat([m3] * self.args.batch_size, axis=0)
                print("mask2:", self.mask2)
                self.target_bow = target_bow * self.mask2 + answers_in_persona_label * self.args.lamba_persona_weight
                print("self.target_bow:", self.target_bow)
                # sigmoid loss ylogy+(1-y)log(1-y)
                self.loss2 = -tf.reduce_mean(input_tensor=self.target_bow * tf.log(self.bow_prediction + eps) +
                                                          (1 - self.target_bow) * tf.log(
                    (1 - self.bow_prediction) + eps), axis=1)

                # ----------------------------------------------------------------------------------------------
                # loss3: 监督对persona的attention (encoder)
                # -----------------------------------------------------------------------------------------------
                print("lamba_loss2:", self.args.lamba_loss2)
                persona_a_t = tf.log(self.persona_a_t + eps)
                print("answer_attention_ph:", self.answer_attention_ph)
                print("persona_sentence_a_t:", persona_a_t)
                self.loss3 = -tf.reduce_sum(input_tensor=persona_a_t * self.answer_attention_ph,
                                            axis=1)  # [batch_size]
                print("self.loss3:", self.loss3)

                # 总loss
                self.loss1 = tf.reduce_mean(self.loss1)
                self.loss2 = tf.reduce_mean(self.loss2)
                self.loss3 = tf.reduce_mean(self.loss3)

                self.loss = self.loss1 + self.args.lamba_loss1 * self.loss2 + self.args.lamba_loss2 * self.loss3

                # self.loss = tf.reduce_mean(self.loss1)

                # -----------tersonborad ------------------
                # tf.summary.scalar('loss', self.loss)

                # 计算ppl时，注释掉
                params = tf.trainable_variables()
                # 求导
                gradients = tf.gradients(self.loss, params)
                # 如果 clip_norm < global_norm ，就进行截取 (缩小梯度）
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.args.max_gradient_norm)
                self.opt_op = tf.compat.v1.train.AdamOptimizer(self.learning_rate).apply_gradients(
                    zip(clipped_gradients, params),
                    global_step=self.global_step)
            else:
                start_tokens = tf.ones([self.args.batch_size, ], tf.int32) * self.args.GO
                end_token = self.args.EOS
                # decoder阶段根据是否使用beam_search决定不同的组合，
                # 如果使用则直接调用BeamSearchDecoder（里面已经实现了helper类）
                # 如果不使用则调用GreedyEmbeddingHelper+BasicDecoder的组合进行贪婪式解码

                # 因为存在oov词语，在预测时embedding中查不到，故扩展embedding 大小为vocabSize + self.batch_OOV_num
                # oov 词语的embedding 用<UNK>代替
                # self.batch_OOV_num 是tf.Tensor，会报错，  直接扩大成vocabSize + self.args.
                # OOV_embedding = tf.concat([tf.expand_dims(embedding[3], 0)]*self.args.num_topic_words, 0)
                # # print("OOV_embedding:", OOV_embedding)
                # embedding = tf.concat([embedding, OOV_embedding], 0)
                # print("embedding:", embedding)
                if self.args.beam_search:
                    print("decoder_cell:", self.decoder_cell)
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=self.decoder_cell,
                                                                             embedding=embedding,
                                                                             start_tokens=start_tokens,
                                                                             end_token=end_token,
                                                                             initial_state=encoder_State,
                                                                             beam_width=self.args.num_BeamSearch,
                                                                             output_layer=output_layer,
                                                                             length_penalty_weight=0.5)
                else:
                    decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                                               start_tokens=start_tokens,
                                                                               end_token=end_token)
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                                        helper=decoding_helper,
                                                                        initial_state=encoder_State,
                                                                        output_layer=output_layer)
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                          maximum_iterations=self.args.max_num_Alength + 1,
                                                                          scope="decoder")
                # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，
                # 对于不使用beam_search的时候，它里面包含两项(rnn_outputs, sample_id)
                # rnn_output: [batch_size, decoder_targets_length, vocab_size]
                # sample_id: [batch_size, decoder_targets_length], tf.int32

                # 对于使用beam_search的时候，它里面包含两项(predicted_ids, beam_search_decoder_output)
                # predicted_ids: [batch_size, decoder_targets_length, beam_size],保存输出结果
                # beam_search_decoder_output: BeamSearchDecoderOutput instance namedtuple(scores, predicted_ids, parent_ids)
                # 所以对应只需要返回predicted_ids或者sample_id即可翻译成最终的结果
                if self.args.beam_search:
                    # self.decoder_predict_decode = decoder_outputs.predicted_ids
                    self.decoder_predict_decode = decoder_outputs.predicted_ids
                else:
                    self.decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)
                print("self.decoder_predict_decode:", self.decoder_predict_decode)

                # 取第一个结果
                self.answers_predict = self.decoder_predict_decode[:, :, 0]
                print("answers_predict:", self.answers_predict)

        variable = [v for v in tf.trainable_variables()]
        for v in variable:
            print(v)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=99999999)
        print("build model finish")

    def random_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        # 从截断的正态分布中输出随机值
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))),
                           name=name)

    def random_bias(self, dim, name=None):
        return tf.Variable(tf.truncated_normal([dim]), name=name)

    def mat_weight_mul(self, mat, weight):
        # [batch_size, n, m] * [m, p] = [batch_size, n, p]
        # 用.as_list()得到size
        mat_shape = mat.get_shape().as_list()
        weight_shape = weight.get_shape().as_list()
        # mat_shape的最后一列数和weight_shape的第一行数相同（判断可运算）
        assert (mat_shape[-1] == weight_shape[0])
        mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]])  # [batch_size * n, m]
        mul = tf.matmul(mat_reshape, weight)  # [batch_size * n, p]
        return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])

    def add_layer(self, inputs, in_size, out_size, activation_function=None):
        # add one more layer and return the output of this layer
        Weights = self.random_weight(in_size, out_size, name="MLP_Weight")
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name="MLP_bais")
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs


class Model1_2:
    def __init__(self, type, training_steps_per_epoch, vocabSize):
        # context 检索 persona 的时候， 直接attention
        self.type = type
        self.args = args(type)

        print("build model1")
        print("vocabSize", vocabSize)

        # ==============================================================================
        #                               设置必要参数
        # ==============================================================================
        if(type == "train"):
            self.global_step = tf.Variable(0, trainable=False)
            # 设置学习速率
            self.learning_rate = tf.maximum(
                # 如果staircase=True，那就表明每training_steps_per_epoch次计算学习速率变化
                tf.train.exponential_decay(
                    self.args.learning_rate,
                    self.global_step,
                    training_steps_per_epoch,
                    self.args.learning_rate_decay_factor,
                    staircase=True),
                self.args.min_learning_rate)

        START = tf.constant(value=[self.args.GO] * self.args.batch_size)
        # ==============================================================================
        #                               define placeholder
        # ==============================================================================
        with tf.name_scope("placeholder"):
            self.personas_ph = tf.placeholder(tf.int32,
                                              shape=[self.args.batch_size, self.args.max_num_persona,
                                                     self.args.max_num_personalength], name="personas")
            self.personas_len_ph = tf.placeholder(tf.int32,
                                                  shape=[self.args.batch_size, self.args.max_num_persona],
                                                  name="persona_lengths")
            self.persona_turn = tf.placeholder(tf.int32,
                                                  shape=[self.args.batch_size], name="persona_turn")

            self.historys_ph = tf.placeholder(tf.int32,
                                              shape=[self.args.batch_size, 2*self.args.max_num_history_turns+1,
                                                     self.args.max_num_Qlength], name="historys")
            self.historys_len_ph = tf.placeholder(tf.int32,
                                           shape=[self.args.batch_size, 2*self.args.max_num_history_turns+1],
                                                  name="history_lengths")
            self.historys_turn = tf.placeholder(tf.int32,
                                                  shape=[self.args.batch_size], name="history_turn")

            # self.questions_ph = tf.placeholder(tf.int32,
            #                                    shape=[self.args.batch_size,self.args.max_num_Qlength], name="questions")
            # self.question_len_ph = tf.placeholder(tf.int32, shape=[self.args.batch_size], name="question_lengths")

            self.answers_ph = tf.placeholder(tf.int32,
                                             shape=[self.args.batch_size, self.args.max_num_Alength], name="answers")
            self.answer_len_ph = tf.placeholder(tf.int32, shape=[self.args.batch_size], name="answer_lengths")
            self.answer_targets_ph = tf.placeholder(tf.int32,
                                                shape=[self.args.batch_size, self.args.max_num_Alength + 1],
                                                name="answer_targets")

            personas_turn_mask = tf.sequence_mask(self.persona_turn, self.args.max_num_persona)
            self.att_persona_sentence_mask = tf.cast(personas_turn_mask, dtype=tf.float32)
            print("self.att_persona_sentence_mask :", self.att_persona_sentence_mask)
            personas_len_mask = tf.sequence_mask(self.personas_len_ph, self.args.max_num_personalength)
            personas_len_mask = tf.reshape(personas_len_mask, [self.args.batch_size, -1])
            self.att_persona_mask = tf.cast(personas_len_mask, dtype=tf.float32)
            print("self.att_persona_mask:", self.att_persona_mask)
            historys_len_mask = tf.sequence_mask(self.historys_len_ph, self.args.max_num_Qlength)
            historys_len_mask = tf.reshape(historys_len_mask, [self.args.batch_size, -1])
            self.att_message_mask = tf.cast(historys_len_mask, dtype=tf.float32)
            print("self.att_message_mask:", self.att_message_mask)  # 输出测试正确
            # question_len_mask = tf.sequence_mask(self.question_len_ph, self.args.max_num_Qlength)
            # question_len_mask = tf.reshape(question_len_mask, [self.args.batch_size, -1])
            # self.att_message_mask = tf.concat([historys_len_mask, question_len_mask], 1)      # 不理解当时为什么queation加上，不是history中有question吗？
            # self.att_message_mask = tf.cast(self.att_message_mask, dtype=tf.float32)
            # print("self.att_message_mask:", self.att_message_mask)  # 输出测试正确

            # 因为结尾加入EOS,长度都加1
            self.answer_len_ph_ = self.answer_len_ph + 1

            # ---------------------------------------------------
            # 加入DB 多引入的输入
            # ---------------------------------------------------
            # self.personas_emb_ph = tf.placeholder(tf.float32, shape=[self.args.batch_size, self.args.max_num_persona,
            #                                                          self.args.num_topics], name="personas_emb")
            # self.historys_emb_ph = tf.placeholder(tf.float32,
            #                                       shape=[self.args.batch_size, 2 * self.args.max_num_history_turns + 1,
            #                                              self.args.num_topics], name="historys_emb")
            # self.questions_emb_ph = tf.placeholder(tf.float32, shape=[self.args.batch_size,
            #                                                           self.args.num_topics], name="questions_emb")

            self.topic_words_emb_ph = tf.placeholder(tf.float32, shape=[self.args.batch_size, self.args.num_topic_words,
                                                                      self.args.num_topics], name="topic_words_emb")
            # self.topic_words_index = tf.placeholder(tf.int32, shape=[self.args.batch_size, self.args.num_topic_words], name="topic_words_index")
            # self.batch_OOV_num = tf.placeholder(tf.int32, shape=[], name="batch_OOV_num")

            # ---------------------------------------------------
            # 加入bow-loss  多引入的输入
            # ---------------------------------------------------
            self.answers_in_persona_label = tf.placeholder(tf.int32, shape=[self.args.batch_size, vocabSize],
                                                           name="answers_in_persona_label")     # 0/1 标签
            answers_in_persona_label = tf.cast(self.answers_in_persona_label, tf.float32)

            # ---------------------------------------------------
            # 加上 persona attention的label 多引入的输入
            # ---------------------------------------------------
            self.answer_attention_ph = tf.placeholder(tf.float32,
                                                      shape=[self.args.batch_size, self.args.max_num_persona],
                                                      name="answer_attention")
            # self.answer_attention_ph = tf.placeholder(tf.int32,
            #                                           shape=[self.args.batch_size, self.args.max_num_persona],
            #                                           name="answer_attention")

            # ==============================================================================
            # Embedding (share) and other variable
            # ==============================================================================
            with ops.device("/cpu:0"):
                if variable_scope.get_variable_scope().initializer:
                    # 变量已存在
                    initializer = variable_scope.get_variable_scope().initializer
                else:
                    # Default initializer for embeddings should have variance=1. 初始化
                    # sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
                    # # 生成均匀分布的随机数
                    # initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)
                    File = h5py.File("../Data/glove_train.h5", 'r')
                    initializer = np.array(File["embedding"])

                # embedding = variable_scope.get_variable(name="embedding", shape=[vocabSize, self.args.embedSize], initializer=initializer, dtype=tf.float32)
                embedding = variable_scope.get_variable(name="embedding", initializer=initializer, dtype=tf.float32)

                # Weights
                # self.W_eQ = self.random_weight(self.args.rnnHiddenSize * 2, self.args.rnnHiddenSize * 2,
                #                                name='W_eQ')
                # self.W_eP = self.random_weight(self.args.rnnHiddenSize * 2, self.args.rnnHiddenSize * 2,
                #                                name='W_eP')

                self.W_p_key = self.random_weight(self.args.rnnHiddenSize * 2, self.args.rnnHiddenSize * 2,
                                                  name="W_p_key")
                self.W_p_value = self.random_weight(self.args.rnnHiddenSize * 2, self.args.rnnHiddenSize * 2,
                                                  name="W_p_value")

                # Biases
                # self.B_v_P = self.random_bias(self.args.rnnHiddenSize * 2, name='B_v_P')

                # Lambde loss 比例参数
                # self.lam = tf.Variable(tf.random_normal(1), name="lam")

            START_EMB = embedding_ops.embedding_lookup(embedding, START)

            # ==============================================================================
            # split placeholders and embed
            # ==============================================================================
            personas = embedding_ops.embedding_lookup(embedding, self.personas_ph)
            personas = tf.transpose(personas, [1, 0, 2, 3])
            personas_lengths = tf.transpose(self.personas_len_ph, [1, 0])
            historys = embedding_ops.embedding_lookup(embedding, self.historys_ph)
            historys = tf.transpose(historys, [1, 0, 2, 3])
            historys_lengths = tf.transpose(self.historys_len_ph, [1, 0])
            # questions = embedding_ops.embedding_lookup(embedding, self.questions_ph)
            answers = embedding_ops.embedding_lookup(embedding, self.answers_ph)

            # ==============================================================================
            # make RNN cell
            # ==============================================================================
            def single_cell(hidden_size, in_keep_prob):
                if self.args.use_lstm:
                    # forget_bias就是LSTM们的忘记系数，如果等于1，就是不会忘记任何信息。
                    cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
                else:
                    cell = tf.contrib.rnn.GRUCell(hidden_size)
                cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=in_keep_prob)
                return cell

            def make_cell(hidden_size, in_keep_prob):
                if self.args.rnnLayers > 1:
                    return tf.contrib.rnn.MultiRNNCell(
                        [single_cell(hidden_size, in_keep_prob) for _ in range(hidden_size)])
                else:
                    return single_cell(hidden_size, in_keep_prob)

            fw_encoder_cell_persona = make_cell(self.args.rnnHiddenSize, self.args.keep_prob)
            bw_encoder_cell_persona = make_cell(self.args.rnnHiddenSize, self.args.keep_prob)
            fw_encoder_cell_history_1 = make_cell(self.args.rnnHiddenSize, self.args.keep_prob)
            bw_encoder_cell_history_1 = make_cell(self.args.rnnHiddenSize, self.args.keep_prob)
            fw_encoder_cell_history_2 = make_cell(self.args.rnnHiddenSize, self.args.keep_prob)
            bw_encoder_cell_history_2 = make_cell(self.args.rnnHiddenSize, self.args.keep_prob)
            # fw_encoder_cell_question = make_cell(self.args.rnnHiddenSize, self.args.keep_prob)
            # bw_encoder_cell_question = make_cell(self.args.rnnHiddenSize, self.args.keep_prob)

            # 串联所有word level 的 hidden-state
            persona_word_enc = []
            message_word_enc = []

            # ==============================================================================
            # encode persona (直接串联)
            # ==============================================================================
            print("encode personas...")
            personas_enc = []
            for i in range(self.args.max_num_persona):
                with tf.variable_scope('persona_sentence_EncoderRNN', reuse=tf.AUTO_REUSE) as varscope:
                    persona_sentence_Output, persona_sentence_State = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=fw_encoder_cell_persona, cell_bw=bw_encoder_cell_persona,
                        inputs=personas[i], sequence_length=personas_lengths[i],
                        dtype=tf.float32, scope=varscope)  # [batch_size, encoder_cell.state_size]
                    persona_sentence_Output = tf.concat([persona_sentence_Output[0], persona_sentence_Output[1]], -1)
                    persona_sentence_State = tf.concat([persona_sentence_State[0], persona_sentence_State[1]], -1)
                # print("persona_sentence_State.h:",persona_sentence_State.h)
                # 构建成下一个rnn 的输入形式
                persona_sentence_State = tf.reshape(persona_sentence_State, [self.args.batch_size, 1, -1])
                if i == 0:
                    personas_enc = persona_sentence_State
                    persona_word_enc = persona_sentence_Output
                else:
                    personas_enc = tf.concat([personas_enc, persona_sentence_State], 1)     # sentenses memory
                    persona_word_enc = tf.concat([persona_word_enc, persona_sentence_Output], 1)      # words memory

            print("personas_enc:", personas_enc)  # [batch_size, max_num_persona, hiddensize]
            print("persona_word_enc:", persona_word_enc)  # [batch_size, max_num_persona*persona_length, hiddensize]

            # ==============================================================================
            # encode history (HRED)
            # ==============================================================================
            print("encode history...")
            historys_sentence_enc = []
            for i in range(2 * self.args.max_num_history_turns + 1):
                with tf.variable_scope('history_sentence_EncoderRNN', reuse=tf.AUTO_REUSE) as varscope:
                    history_sentence_Output, history_sentence_State = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=fw_encoder_cell_history_1, cell_bw=bw_encoder_cell_history_1,
                        inputs=historys[i], sequence_length=historys_lengths[i],
                        dtype=tf.float32, scope=varscope)  # [batch_size, encoder_cell.state_size]
                    history_sentence_Output = tf.concat([history_sentence_Output[0], history_sentence_Output[1]], -1)
                    history_sentence_State = tf.concat([history_sentence_State[0], history_sentence_State[1]], -1)
                # print("history_sentence_State:",history_sentence_State)
                # 构建成下一个rnn 的输入形式
                history_sentence_State = tf.reshape(history_sentence_State, [self.args.batch_size, 1, -1])
                if i == 0:
                    historys_sentence_enc = history_sentence_State
                    message_word_enc = history_sentence_Output
                else:
                    historys_sentence_enc = tf.concat([historys_sentence_enc, history_sentence_State], 1)
                    message_word_enc = tf.concat([message_word_enc, history_sentence_Output], 1)
            print("historys_sentence_enc:", historys_sentence_enc)  # [batch_size, h_turn, hidden_size*2]
            print("message_word_enc:", message_word_enc)
            with tf.variable_scope('history_sequence_EncoderRNN', reuse=tf.AUTO_REUSE) as varscope:
                history_sequence_Output, history_sequence_State = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=fw_encoder_cell_history_2, cell_bw=bw_encoder_cell_history_2,
                    inputs=historys_sentence_enc, sequence_length=self.historys_turn,
                    dtype=tf.float32, scope=varscope)  # [batch_size, encoder_cell.state_size]
                history_sequence_Output = tf.concat([history_sequence_Output[0], history_sequence_Output[1]], -1)
                history_sequence_State = tf.concat([history_sequence_State[0], history_sequence_State[1]], -1)
            print("history_sequence_output:", history_sequence_Output)  # [batch_size, h_turn, hidden_size]
            history_sequence_Output = tf.transpose(history_sequence_Output, [1, 0, 2])
            print("history_sequence_output:", history_sequence_Output)  # [h_turn, batch_size, hidden_size]
            print("history_sequence_State:", history_sequence_State)    # [batch_size, hidden_size]

            # ====================================================
            # context 检索 persona sentense memory
            # ====================================================
            print("query -->  persona_swntensen-memory")
            personas_enc_key = tf.matmul(personas_enc, self.W_p_key)
            personas_enc_value = tf.matmul(personas_enc, self.W_p_value)

            # query = 0
            # for i in range(2 * self.args.max_num_history_turns + 1):
            #     query = query + history_sequence_Output[i]
            #     s = tf.reduce_sum(tf.multiply(tf.expand_dims(query, 1), personas_enc_key), 2)     # [batch_size, len]
            #     a_t = tf.nn.softmax(s)
            #     # print("a_t:", a_t)
            #     persona_a_t_all.append(a_t)
            #     v_P = tf.reduce_sum(tf.multiply(tf.expand_dims(a_t, -1), personas_enc_value), 1)
            #     # print("v_P:", v_P)  # [batch_size, hiden_size]
            #     # 下一步的query
            #     query = v_P
            #     # query = query + v_P
            #     persona_memory_enc.append(query)        # [len, batch, hiden_size]
            #
            # # 根据context长度选择最后memory结果
            # a = tf.range(self.args.batch_size)
            # b = self.historys_turn - 1
            # index = tf.concat([tf.expand_dims(b, 1), tf.expand_dims(a, 1)], 1)
            # print("index:", index)
            # persona_memory = tf.gather_nd(persona_memory_enc, index)
            # print("persona_memory:", persona_memory)
            # self.persona_a_t = tf.gather_nd(persona_a_t_all, index)

            s = tf.reduce_sum(tf.multiply(tf.expand_dims(history_sequence_State, 1), personas_enc_key), 2)  # [batch_size, len]
            a_t = tf.nn.softmax(s)
            # print("a_t:", a_t)
            v_P = tf.reduce_sum(tf.multiply(tf.expand_dims(a_t, -1), personas_enc_value), 1)
            self.persona_a_t = a_t

            # ====================================================
            # 合并信息，得到s0
            # ====================================================
            encoder_State = tf.concat(values=[history_sequence_State, v_P], axis=1)  # [batch_size, (2*hidden_size)*2]
            encoder_State = tf.layers.dense(encoder_State, self.args.rnnHiddenSize)
            print("encoder_State:", encoder_State)

            # attention到句子
            persona_sentence_attention_State = personas_enc
            print("persona_sentence_attention_State:", persona_sentence_attention_State)
            # attention 到词
            persona_attention_State = persona_word_enc
            message_attention_State = message_word_enc
            print("persona_attention_State:", persona_attention_State)
            print("message_attention_State:", message_attention_State)

            # ==============================================================================
            # decode
            # ==============================================================================
            print("decode ...")
            with tf.variable_scope('DecoderRNN'):
                # att_persona_sentence_mask = self.att_persona_sentence_mask
                att_persona_mask = self.att_persona_mask
                att_message_mask = self.att_message_mask
                topic_words_emb_ph = self.topic_words_emb_ph
                encoder_State_s0 = encoder_State
            #     # self.topic_words_index 转成onehot， 为了扩展Pk的维度， 加到Pv上
                if (self.type != "train") and self.args.beam_search:
                    # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
                    print("use beamsearch decoding..  num_BeamSearch=", self.args.num_BeamSearch)
                    # persona_sentence_attention_State = tf.contrib.seq2seq.tile_batch(persona_sentence_attention_State,
                    #                                                                  multiplier=self.args.num_BeamSearch)
                    # att_persona_sentence_mask = tf.contrib.seq2seq.tile_batch(self.att_persona_sentence_mask,
                    #                                                           multiplier=self.args.num_BeamSearch)
                    persona_attention_State = tf.contrib.seq2seq.tile_batch(persona_attention_State,
                                                                            multiplier=self.args.num_BeamSearch)
                    att_persona_mask = tf.contrib.seq2seq.tile_batch(self.att_persona_mask,
                                                                     multiplier=self.args.num_BeamSearch)
                    message_attention_State = tf.contrib.seq2seq.tile_batch(message_attention_State,
                                                                            multiplier=self.args.num_BeamSearch)
                    att_message_mask = tf.contrib.seq2seq.tile_batch(self.att_message_mask,
                                                                     multiplier=self.args.num_BeamSearch)
                    topic_words_emb_ph = tf.contrib.seq2seq.tile_batch(self.topic_words_emb_ph,
                                                                       multiplier=self.args.num_BeamSearch)
                    encoder_State_s0 = tf.contrib.seq2seq.tile_batch(encoder_State, multiplier=self.args.num_BeamSearch)
                    encoder_State = nest.map_structure(
                        lambda s: tf.contrib.seq2seq.tile_batch(s, self.args.num_BeamSearch),
                        encoder_State)
                    # 如果使用beam_seach则batch_size = self.batch_size * self.beam_size。因为之前已经复制过一次


                # mask 部分将s设置为-inf，所以softmax的时候为0
                message_mask_inf = 1 - att_message_mask
                mask = np.zeros(att_message_mask.shape)
                for i in range(att_message_mask.shape[0]):
                    for j in range(att_message_mask.shape[1]):
                        if message_mask_inf[i][j] == 1:
                            mask[i][j] = -np.inf
                att_message_mask_inf = message_mask_inf * mask

                persona_mask_inf = 1 - att_persona_mask
                mask = np.zeros(att_persona_mask.shape)
                for i in range(att_persona_mask.shape[0]):
                    for j in range(att_persona_mask.shape[1]):
                        if persona_mask_inf[i][j] == 1:
                            mask[i][j] = -np.inf
                att_persona_mask_inf = persona_mask_inf * mask

                self.decoder_cell_ = MyCell(self.args.rnnHiddenSize, persona_attention_State, att_persona_mask_inf,
                                            message_attention_State, att_message_mask_inf, encoder_State_s0,
                                            topic_words_emb_ph)

                # ------------- dropout ------------------
                self.decoder_cell = tf.contrib.rnn.DropoutWrapper(self.decoder_cell_,
                                                                  input_keep_prob=self.args.keep_prob)

                # train 和 test用的decoder不一样， 为了变量名对应
                if (self.type == "train"):
                    output_layer = tf.compat.v1.layers.Dense(vocabSize,
                                                   kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                                      stddev=0.1),
                                                   name='decoder/dense')
                else:
                    output_layer = tf.compat.v1.layers.Dense(vocabSize,
                                                   kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                                      stddev=0.1))

                if (self.type == "train"):
                    # 定义decoder阶段的输入，其实就是在decoder的seq_outputs开始处添加一个<start>
                    # seq_outputs的shape为[batch_size, decoder_targets_length, embedding_size]
                    # seq_outputs 分解为 max_num_Alength 个tensor（batch_size, emb_size）
                    answers = [tf.squeeze(input=word, axis=1) for word in
                               tf.split(value=answers, num_or_size_splits=self.args.max_num_Alength, axis=1)]
                    # print("answer:",answer)
                    answers = [START_EMB] + answers
                    # answers = tf.transpose(answers, [1, 0, 2])  # [batch_size, A_length+1, embedding_size]
                    print("answers:", answers)

                    # 用dynamic_rnn进行decoder
                    decoder_Outputs, decoder_State = static_rnn(cell=self.decoder_cell, inputs=answers,
                                                                initial_state=encoder_State,
                                                                sequence_length=self.answer_len_ph_,
                                                                dtype=tf.float32, scope="decoder")
                    decoder_Outputs = tf.stack(decoder_Outputs, 1)
                    print("decoder_Outputs:", decoder_Outputs)

                    # ------------输出 persona_sentence_a_t----------------------
                    # self.persona_sentence_a_t = decoder_Outputs[:, :, -self.args.max_num_persona:]
                    # print("self.persona_sentence_a_t:", self.persona_sentence_a_t)
                    # decoder_Outputs = decoder_Outputs[:, :, :-self.args.max_num_persona]
                    # print("decoder_Outputs:", decoder_Outputs)

                    # ------------输出 topic_a_t----------------------
                    # self.topic_a_t = decoder_Outputs[:, :, -self.args.num_topic_words:]
                    # print("self.topic_a_t:", self.topic_a_t)
                    # decoder_Outputs = decoder_Outputs[:, :, :-self.args.num_topic_words]
                    # print("decoder_Outputs:", decoder_Outputs)

                    self.decoder_logits_train = output_layer(decoder_Outputs)  # [batch_size, A_len, vocab]
                    print("self.decoder_logits_train:", self.decoder_logits_train)

                    # 结果
                    self.answers_predict = tf.argmax(self.decoder_logits_train, axis=-1, name='answers_predict')
                    print("self.answers_predict:", self.answers_predict)

                    # 根据输出计算loss和梯度，并定义进行更新的AdamOptimizer和train_op
                    # mask标记是否为pad部分（1/0）
                    mask = tf.cast(x=tf.not_equal(x=self.answer_targets_ph, y=self.args.PAD),
                                   dtype=tf.float32)  # [batch_size, Alength+1]
                    # 使用sequence_loss计算loss，这里需要传入之前定义的mask标志
                    # 这里可能报错，是因为dynamic_decode的输出大小是动态的，可能造成数据shape不匹配
                    self.loss1 = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                                  targets=self.answer_targets_ph, weights=mask)

                    self.ppl = tf.reduce_mean(tf.exp(self.loss1))

                    # ----------------------------------------------------------------------------------------------
                    #  2:loss2: label是answer中的所有词(加重persona中词的权重)  sigmoid
                    # -----------------------------------------------------------------------------------------------
                    print("bow-loss-sigmoid-weight")
                    print("lamba_loss1:", self.args.lamba_loss1)
                    bow_state = tf.reduce_sum(self.decoder_logits_train, 1)  # [batch_size, vocab]
                    self.bow_prediction = tf.nn.sigmoid(bow_state)
                    print("self.bow_prediction:", self.bow_prediction)  # [batch_size, vocab]

                    target_one_hot_bow = tf.one_hot(indices=self.answer_targets_ph, depth=vocabSize,
                                                    dtype=tf.float32)  # [batch_size, Alength+1, vocab]
                    # target_bow = tf.reduce_sum(input_tensor=target_one_hot_bow, axis=1)     # [batch_size, vocab]   出现过累计计数，未出现计数0
                    target_bow = tf.reduce_max(input_tensor=target_one_hot_bow,
                                               axis=1)  # [batch_size, vocab]   出现过计数1，未出现计数0
                    # mask2 是为了去掉 pad、eos等
                    m1 = [1.0 for _ in range(vocabSize - 4)]
                    m2 = [0.0 for _ in range(4)]
                    m3 = tf.reshape(tf.concat([m2, m1], 0), [1, -1])
                    self.mask2 = tf.concat([m3] * self.args.batch_size, axis=0)
                    print("mask2:", self.mask2)
                    self.target_bow = target_bow * self.mask2 + answers_in_persona_label * self.args.lamba_persona_weight
                    print("self.target_bow:", self.target_bow)
                    # sigmoid loss ylogy+(1-y)log(1-y)
                    self.loss2 = -tf.reduce_mean(input_tensor=self.target_bow * tf.log(self.bow_prediction + eps) +
                                                              (1 - self.target_bow) * tf.log(
                        (1 - self.bow_prediction) + eps), axis=1)

                    # ----------------------------------------------------------------------------------------------
                    # loss3: 监督对persona的attention (encoder)
                    # -----------------------------------------------------------------------------------------------
                    print("lamba_loss2:", self.args.lamba_loss2)
                    persona_a_t = tf.log(self.persona_a_t + eps)
                    print("answer_attention_ph:", self.answer_attention_ph)
                    print("persona_sentence_a_t:", persona_a_t)
                    self.loss3 = -tf.reduce_sum(input_tensor=persona_a_t * self.answer_attention_ph,
                                                axis=1)  # [batch_size]
                    print("self.loss3:", self.loss3)

                    # 总loss
                    self.loss1 = tf.reduce_mean(self.loss1)
                    self.loss2 = tf.reduce_mean(self.loss2)
                    self.loss3 = tf.reduce_mean(self.loss3)

                    self.loss = self.loss1 + self.args.lamba_loss1 * self.loss2 + self.args.lamba_loss2 * self.loss3

                    # self.loss = tf.reduce_mean(self.loss1)

                    # -----------tersonborad ------------------
                    # tf.summary.scalar('loss', self.loss)

                    # 计算ppl时，注释掉
                    params = tf.trainable_variables()
                    # 求导
                    gradients = tf.gradients(self.loss, params)
                    # 如果 clip_norm < global_norm ，就进行截取 (缩小梯度）
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.args.max_gradient_norm)
                    self.opt_op = tf.compat.v1.train.AdamOptimizer(self.learning_rate).apply_gradients(
                        zip(clipped_gradients, params),
                        global_step=self.global_step)
                else:
                    start_tokens = tf.ones([self.args.batch_size, ], tf.int32) * self.args.GO
                    end_token = self.args.EOS
                    # decoder阶段根据是否使用beam_search决定不同的组合，
                    # 如果使用则直接调用BeamSearchDecoder（里面已经实现了helper类）
                    # 如果不使用则调用GreedyEmbeddingHelper+BasicDecoder的组合进行贪婪式解码

                    # 因为存在oov词语，在预测时embedding中查不到，故扩展embedding 大小为vocabSize + self.batch_OOV_num
                    # oov 词语的embedding 用<UNK>代替
                    # self.batch_OOV_num 是tf.Tensor，会报错，  直接扩大成vocabSize + self.args.
                    # OOV_embedding = tf.concat([tf.expand_dims(embedding[3], 0)]*self.args.num_topic_words, 0)
                    # # print("OOV_embedding:", OOV_embedding)
                    # embedding = tf.concat([embedding, OOV_embedding], 0)
                    # print("embedding:", embedding)
                    if self.args.beam_search:
                        print("decoder_cell:", self.decoder_cell)
                        inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=self.decoder_cell,
                                                                                 embedding=embedding,
                                                                                 start_tokens=start_tokens,
                                                                                 end_token=end_token,
                                                                                 initial_state=encoder_State,
                                                                                 beam_width=self.args.num_BeamSearch,
                                                                                 output_layer=output_layer,
                                                                                 length_penalty_weight=0.5)
                    else:
                        decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                                                   start_tokens=start_tokens,
                                                                                   end_token=end_token)
                        inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                                            helper=decoding_helper,
                                                                            initial_state=encoder_State,
                                                                            output_layer=output_layer)
                    decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                              maximum_iterations=self.args.max_num_Alength + 1,
                                                                              scope="decoder")
                    # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，
                    # 对于不使用beam_search的时候，它里面包含两项(rnn_outputs, sample_id)
                    # rnn_output: [batch_size, decoder_targets_length, vocab_size]
                    # sample_id: [batch_size, decoder_targets_length], tf.int32

                    # 对于使用beam_search的时候，它里面包含两项(predicted_ids, beam_search_decoder_output)
                    # predicted_ids: [batch_size, decoder_targets_length, beam_size],保存输出结果
                    # beam_search_decoder_output: BeamSearchDecoderOutput instance namedtuple(scores, predicted_ids, parent_ids)
                    # 所以对应只需要返回predicted_ids或者sample_id即可翻译成最终的结果
                    if self.args.beam_search:
                        # self.decoder_predict_decode = decoder_outputs.predicted_ids
                        self.decoder_predict_decode = decoder_outputs.predicted_ids
                    else:
                        self.decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)
                    print("self.decoder_predict_decode:", self.decoder_predict_decode)

                    # 取第一个结果
                    self.answers_predict = self.decoder_predict_decode[:, :, 0]
                    print("answers_predict:", self.answers_predict)

            variable = [v for v in tf.trainable_variables()]
            for v in variable:
                print(v)

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=99999999)
            print("build model finish")

    def random_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        # 从截断的正态分布中输出随机值
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))),
                           name=name)

    def random_bias(self, dim, name=None):
        return tf.Variable(tf.truncated_normal([dim]), name=name)

    def mat_weight_mul(self, mat, weight):
        # [batch_size, n, m] * [m, p] = [batch_size, n, p]
        # 用.as_list()得到size
        mat_shape = mat.get_shape().as_list()
        weight_shape = weight.get_shape().as_list()
        # mat_shape的最后一列数和weight_shape的第一行数相同（判断可运算）
        assert (mat_shape[-1] == weight_shape[0])
        mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]])  # [batch_size * n, m]
        mul = tf.matmul(mat_reshape, weight)  # [batch_size * n, p]
        return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])

    def add_layer(self, inputs, in_size, out_size, activation_function=None):
        # add one more layer and return the output of this layer
        Weights = self.random_weight(in_size, out_size, name="MLP_Weight")
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name="MLP_bais")
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs


if __name__ == '__main__':
    model = Model1("train", 100, 1000)