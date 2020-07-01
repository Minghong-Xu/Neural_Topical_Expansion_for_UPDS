import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("..")
import math
import random
import tensorflow as tf
from collections import Counter
import numpy as np
from util import metrics
from data_util_8 import data_preprocess3
from sklearn.metrics import f1_score
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from Mymodel_20190815.model1 import Model1 as Model
from Mymodel_20190815.model1 import Model1_2 as Model
from args_8 import args
import nltk
import json
from nlgeval import NLGEval

nlgeval = NLGEval()  # loads the models
eps = 1e-10
smoothie = SmoothingFunction(epsilon=1e-12).method1     # 和ParlAI用的一样

# ==============================================================================
#                               Loading dataset
# ==============================================================================
args = args("test")
data_preprocess = data_preprocess3(args)
num_sample = data_preprocess.num_sample
print("num_sample:", num_sample)
num_sample_train = 65607
num_IterPerEpoch_train = int(math.ceil(num_sample_train/args.batch_size))-1
print("num_IterPerEpoch_train:", num_IterPerEpoch_train)

# ==============================================================================
#                               Build Graph
# ==============================================================================
# Iterations per epoch
# numIterPerEpoch = int(math.ceil(data_preprocess.num_sample/args.batch_size))
# model = Model(type="train", training_steps_per_epoch=numIterPerEpoch,
#               vocabSize=data_preprocess.data.vocabSize)
model = Model(type="test", training_steps_per_epoch=None,
              vocabSize=data_preprocess.data.vocabSize)


config = tf.ConfigProto()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
config.gpu_options.allow_growth = True


# ==============================================================================
#                               测试方法
# ==============================================================================
def id2word(data):
    ind2word = data_preprocess.data.ind2word
    new_data = ' '.join([ind2word[str(m)] for m in data if m != -1])
    # for datai in data:
    #     data = [ind2word[str(m)] for m in datai]
    #     new_data.append(data)
    return new_data


def blue_parlAI(p_answers, answers_str, bleu_n=1):
    # 调用parlAI平台评测指标
    bleu = []
    for k in range(len(p_answers)):
        p_a = p_answers[k]
        y_true = answers_str[k]

        # 判断p_answer结束位置
        j = 0
        for ii in range(len(p_a)):
            p_ai = p_a[ii]
            if p_ai == args.EOS:
                break
            j += 1
        y_pred = id2word(p_a[:j])
        # y_pred = " ".join(y_pred)
        # print("y_pred:", y_pred)
        # print("y_true:", y_true)

        bleu.append(metrics._bleu(y_pred, [y_true], bleu_n))
    return bleu


def F1_parlAI(p_answers, answers_str):
    # 调用parlAI平台评测指标
    F1_score = []
    for k in range(len(p_answers)):
        p_a = p_answers[k]
        y_true = answers_str[k]

        # 判断p_answer结束位置
        j = 0
        for ii in range(len(p_a)):
            p_ai = p_a[ii]
            if p_ai == args.EOS:
                break
            j += 1
        y_pred = id2word(p_a[:j])
        # y_pred = " ".join(y_pred)
        # print("y_pred:", y_pred)
        # print("y_true:", y_true)

        F1_score.append(metrics._f1_score(y_pred, [y_true]))
    return F1_score


def distinctEval(all_paths):
    # distinct evaluation
    # all_paths 是 所有 answer 的 ID  [N, A_len]
    response_ugm = set([])
    response_bgm = set([])
    response_tgm = set([])
    response_len = sum([len(p) for p in all_paths])

    for path in all_paths:
        for u in path:
            response_ugm.add(u)
        for b in list(nltk.bigrams(path)):
            response_bgm.add(b)
        for t in list(nltk.trigrams(path)):
            response_tgm.add(t)

    # print("total length of response:", response_len)
    # print("distinct unigrams:", len(response_ugm)/response_len)
    # print("distinct bigrams:", len(response_bgm)/response_len)
    dist1 = len(response_ugm)/response_len
    dist2 = len(response_bgm)/response_len
    dist3 = len(response_tgm) / response_len
    return dist1, dist2, dist3


# ==============================================================================
#                         other method
# ==============================================================================
def update_batch(batch_topic_words):
    batch_topic_words_emb = []
    for i in range(args.batch_size):
        topic_words = batch_topic_words[i]
        topic_words_emb = []
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


Bleu_total_all = []
Bleu_total_1_all = []
Bleu_total_2_all = []
Bleu_total_3_all = []
Bleu_total_4_all = []
F1_score_all = []
PPL_total_all = []
dict1_all = []
dict2_all = []
dict3_all = []
for bbb in range(num_IterPerEpoch_train, num_IterPerEpoch_train*30+1, num_IterPerEpoch_train):
# for bbb in range(8200, 27000, 200):
    with tf.Session(config=config) as sess:
        idxs = [i for i in range(num_sample)]
        print("model restore from savePath:", args.savePath)
        checkpoint_path = os.path.join(args.savePath, "visdial-%d" % bbb)
        if not os.path.exists(checkpoint_path + '.index'):
            exit(0)

        model.saver.restore(sess, checkpoint_path)

        Bleu_total = []
        Bleu_total_1 = []
        Bleu_total_2 = []
        Bleu_total_3 = []
        Bleu_total_4 = []
        F1_score = []
        PPL_total = []
        model_answers_id = []
        true_answers = []
        model_answers = []

        # 使用enumerate函数迭代
        for batch_id, (start, end) in enumerate(zip(range(0, data_preprocess.num_sample, args.batch_size),
                                                    range(args.batch_size, data_preprocess.num_sample, args.batch_size))):
            # print("idxs[start:end]:", idxs[start:end])
            batch_persona, batch_persona_len, batch_persona_turn, batch_history, batch_history_len, batch_history_turn, \
            batch_question, batch_question_len, batch_answer, batch_answer_len, batch_answer_target, batch_answer_str, \
            batch_answers_in_persona = data_preprocess.get_batch(idxs[start:end])
            # print("batch_answer_str:", batch_answer_str[0])
            # print("batch_answers:", id2word(batch_answer[0]))
            batch_personas_emb, batch_historys_emb, batch_questions_emb, batch_topic_words, batch_topic_words_weigth = data_preprocess.get_batch_topic_info(
                idxs[start:end])

            batch_topic_words_emb = update_batch(batch_topic_words)   # words
            # batch_topic_words_emb = update_batch2(batch_topic_words)  # id

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
                          model.answers_in_persona_label: batch_answers_in_persona}

            # outputs = [ model.answers_predict, model.decoder_outputs]
            # [answers_predict, decoder_outputs] = sess.run(outputs, input_feed)
            # print("batch_id:", batch_id, "decoder_outputs:", decoder_outputs)

            # # 计算ppl要用 train的model计算
            # output_feed = [model.loss]
            # outputs = sess.run(output_feed, input_feed)
            # # print("batch_id:", batch_id, "batch_loss:", outputs[1])
            # ppl = 2**outputs[0]
            # # print("batch_id:", batch_id, "\tbatch_loss:", outputs[0], "\tbatch_ppl:", ppl)
            # PPL_total.append(ppl)

            # 计算bleu 是用valid的model计算

            output_feed = [model.answers_predict]
            outputs = sess.run(output_feed, input_feed)

            # for i in range(args.batch_size):
            #     print("batch_question:", id2word(batch_question[i]))
            #     # print("batch_answer:", id2word(batch_answer[i]))
            #     print("batch_answer_str:", batch_answer_str[i])
            #     print("answers_predict:", id2word(outputs[0][i]))

            bleus = blue_parlAI(outputs[0], batch_answer_str, bleu_n=0)
            Bleu_total.extend(bleus)

            bleus1 = blue_parlAI(outputs[0], batch_answer_str, bleu_n=1)
            Bleu_total_1.extend(bleus1)

            bleus2 = blue_parlAI(outputs[0], batch_answer_str, bleu_n=2)
            Bleu_total_2.extend(bleus2)

            bleus3 = blue_parlAI(outputs[0], batch_answer_str, bleu_n=3)
            Bleu_total_3.extend(bleus3)

            bleus4 = blue_parlAI(outputs[0], batch_answer_str, bleu_n=4)
            Bleu_total_4.extend(bleus4)

            f1 = F1_parlAI(outputs[0], batch_answer_str)
            F1_score.extend(f1)

            for i in range(args.batch_size):
                if batch_id == 100 and i < 10:
                    # print("batch_question:", id2word(batch_question[i]))
                    print("batch_answer_str:", batch_answer_str[i])  # real answer
                    print("model_answer:", id2word(outputs[0][i]))

                # true_answers.append(batch_answer_str[i])
                # model1_answers.append(id2word(outputs[0][i]))
                model_answers_id.append([m for m in outputs[0][i] if m > 2])

                true_answers.append(batch_answer_str[i])
                model_answers.append(id2word(outputs[0][i]).replace("<EOS>", ""))

        # answers_save_path = os.path.join(args.savePath, "answer_save_beam2_23.json")
        # with open(answers_save_path, "w", encoding='UTF-8') as file:
        #     data = {"true_answers": true_answers,
        #             "model1_answers": model_answers}
        #     json.dump(data, file, ensure_ascii=False)
        #     print("save in ", answers_save_path)

        print("num_batch:", bbb, "beam_wide=", args.num_BeamSearch)
        model_n_b_beam2_metrics_dict = nlgeval.compute_metrics([true_answers], model_answers)
        print("model_n_b_beam2:\n", model_n_b_beam2_metrics_dict)
        print("Bleu_total:", np.mean(Bleu_total))
        print("Bleu_total_1:", np.mean(Bleu_total_1))
        print("Bleu_total_2:", np.mean(Bleu_total_2))
        print("Bleu_total_3:", np.mean(Bleu_total_3))
        print("Bleu_total_4:", np.mean(Bleu_total_4))
        Bleu_total_all.append(np.mean(Bleu_total))
        Bleu_total_1_all.append(np.mean(Bleu_total_1))
        Bleu_total_2_all.append(np.mean(Bleu_total_2))
        Bleu_total_3_all.append(np.mean(Bleu_total_3))
        Bleu_total_4_all.append(np.mean(Bleu_total_4))

        print("F1_score:", np.mean(F1_score))
        F1_score_all.append(np.mean(F1_score))

        dist1, dist2, dist3 = distinctEval(model_answers_id)
        print("dist1:", dist1)
        print("dist2:", dist2)
        print("dist3:", dist3)
        dict1_all.append(dist1)
        dict2_all.append(dist2)
        dict3_all.append(dist3)

        # print("PPL_total:", np.mean(PPL_total))
        # PPL_total_all.append(np.mean(PPL_total))

print("Bleu_total_all:", Bleu_total_all)
print("Bleu_total_1_all:", Bleu_total_1_all)
print("Bleu_total_2_all:", Bleu_total_2_all)
print("Bleu_total_3_all:", Bleu_total_3_all)
print("Bleu_total_4_all:", Bleu_total_4_all)
print("F1_score_all:", F1_score_all)
print("dict1_all:", dict1_all)
print("dict2_all:", dict2_all)
print("dict3_all:", dict3_all)
# print("PPL_total_all:",PPL_total_all)

