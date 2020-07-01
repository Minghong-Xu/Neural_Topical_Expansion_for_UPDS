# 训练不同model都可使用
# 需要修改import和input_feed信息
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("..")
import math
import random
import tensorflow as tf
import numpy as np
# from nltk.tokenize import word_tokenize
import datetime
from data_util_8 import data_preprocess3
# from Mymodel_20190815.model1 import Model1 as Model
from Mymodel_20190815.model1 import Model1_2 as Model
from args_8 import args


# ==============================================================================
#                               Loading dataset
# ==============================================================================
args = args("train")
# args = args("test")
data_preprocess = data_preprocess3(args)
num_sample = data_preprocess.num_sample
print("num_sample:", num_sample)

if not os.path.exists(args.savePath):
  os.makedirs(args.savePath)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# gpu_options = tf.GPUOptions(allow_growth=True)

# Iterations per epoch
numIterPerEpoch = int(math.ceil(num_sample/args.batch_size))
print('%d iter per epoch.\n' %numIterPerEpoch)

# ==============================================================================
#                           Build Graph
# ==============================================================================

model = Model(type="train", training_steps_per_epoch=numIterPerEpoch,
              vocabSize=data_preprocess.data.vocabSize)


# ==============================================================================
#                         other method
# ==============================================================================
def update_batch(batch_topic_words):
    batch_topic_words_emb = []
    for i in range(args.batch_size):
        topic_words = batch_topic_words[i]
        topic_words_emb = []
        if len(topic_words) == 0:
            # 未扩充words
            topic_words_emb = np.zeros(args.num_topic_words)
            batch_topic_words_emb.append(topic_words_emb)
            continue

        for j in range(args.num_topic_words):
            if j >= len(topic_words):
                # 筛选长度超过args.num_topic_words的，先填充前边的（不超过10条数据，暂时先这样跑）
                print("填充")
                w = topic_words[j-len(topic_words)]
            else:
                w = topic_words[j]

            # topic_words 转化为embedding形式
            emb_index = data_preprocess.dic["token2id"][w]
            topic_words_emb.append(data_preprocess.embedding[emb_index])

        batch_topic_words_emb.append(topic_words_emb)
    return batch_topic_words_emb


def update_batch2(batch_topic_words):
    batch_topic_words_emb = []
    for i in range(args.batch_size):
        topic_words_id = batch_topic_words[i]
        topic_words_emb = []

        for j in range(args.num_topic_words):
            w = topic_words_id[j]

            # topic_words 转化为embedding形式
            # emb_index = data_preprocess.dic["token2id"][w]
            topic_words_emb.append(data_preprocess.embedding[w])

        batch_topic_words_emb.append(topic_words_emb)
    return batch_topic_words_emb


with tf.Session(config=config) as sess:
    ckpt = tf.train.get_checkpoint_state(args.savePath)

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        # 用保存的模型训练
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        step = int(ckpt.model_checkpoint_path.split('-')[1])

        # checkpoint_path = os.path.join(args.savePath, "visdial-%d" % 6156)
        # model.saver.restore(sess, checkpoint_path)
        # step = 6156

    else:
        # 重新训练
        sess.run(tf.global_variables_initializer())
        step = 0

    # ----------tensorboard 可视化 -------------------
    # merged = tf.summary.merge_all()
    # # 这里的graph是将前面定义的框架信息收集起来，然后放在logs/目录下面
    # writer = tf.summary.FileWriter("logs/", sess.graph)

    # 输出所有变量
    # variable_names = [v.name for v in tf.trainable_variables()]
    # values = sess.run(variable_names)
    # for k, v in zip(variable_names, values):
    #     print("Variable: ", k, "  Shape: ", v.shape)

    idxs = [i for i in range(num_sample)]
    for epoch in range(1, args.num_Epochs + 1):
        # 打乱顺序
        random.shuffle(idxs)
        # 时间是用来计算进度的
        tic = datetime.datetime.now()
        # 使用enumerate函数迭代
        for batch_id, (start, end) in enumerate(zip(range(0, num_sample, args.batch_size),
                                                    range(args.batch_size, num_sample+1, args.batch_size))):
            batch_persona, batch_persona_len, batch_persona_turn, batch_history, batch_history_len, batch_history_turn, \
            batch_question, batch_question_len, batch_answer, batch_answer_len, batch_answer_target, batch_answer_str, \
            batch_answers_in_persona = data_preprocess.get_batch(idxs[start:end])
            # print("batch_answer_str:", batch_answer_str[0])
            # print("batch_answers:", id2word(batch_answer[0]))
            batch_personas_emb, batch_historys_emb, batch_questions_emb, batch_topic_words, batch_topic_words_weigth = data_preprocess.get_batch_topic_info(
                idxs[start:end])

            batch_topic_words_emb = update_batch(batch_topic_words)   # words
            # batch_topic_words_emb = update_batch2(batch_topic_words)    # id
            batch_answer_attention = data_preprocess.get_batch_attention_label(idxs[start:end])

            input_feed = {model.personas_ph: batch_persona,
                          model.personas_len_ph: batch_persona_len,
                          model.persona_turn: batch_persona_turn,
                          model.historys_ph: batch_history,
                          model.historys_len_ph: batch_history_len,
                          model.historys_turn: batch_history_turn,
                          model.answers_ph: batch_answer,
                          model.answer_len_ph: batch_answer_len,
                          model.answer_targets_ph: batch_answer_target,
                          model.topic_words_emb_ph: batch_topic_words_emb,
                          model.answers_in_persona_label: batch_answers_in_persona,
                          model.answer_attention_ph: batch_answer_attention}

            # output_feed = [model.loss, model.opt_op, model.loss1, model.loss2, model.answers_predict]
            output_feed = [model.loss, model.opt_op, model.answers_predict, model.loss1, model.loss2, model.loss3]

            outputs = sess.run(output_feed, input_feed)

            # print("model.Pgen:", outputs[4])
            # for i in range(args.batch_size):
            #     # print("batch_answer:", batch_answer_str[i])
            #     print("batch_answer_target:", id2word(batch_answer_target[i]))
            #     print("model.decoder_predict_train:", id2word(outputs[2][i]))
            step += 1
            # if (step % 1) == 0:
            if (step % 20) == 0:
                # print('%5d:\tepoch=%d\tbatch_id=%d\tloss=%f' % (step, epoch, batch_id + 1, outputs[0]))
                print('%5d:\tepoch=%d\tbatch_id=%d\tloss=%f\tloss1=%f\tloss2=%f\tloss3=%f' %
                      (step, epoch, batch_id + 1, outputs[0], outputs[3], outputs[4], outputs[5]))

                # ----------tensorboard 可视化 -------------------
                # result = sess.run(merged, input_feed)
                # writer.add_summary(result, step)

            # 刷新输出
            sys.stdout.flush()

        toc = datetime.datetime.now()
        print("Epoch finished in {}".format(toc - tic))
        checkpoint_path = os.path.join(args.savePath, "visdial")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        # model.saver.save(sess, checkpoint_path, global_step=int(step/numIterPerEpoch))
        print("Model saved in ", args.savePath)
