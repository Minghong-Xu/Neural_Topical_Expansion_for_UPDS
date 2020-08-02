import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("..")
import math
import random
import tensorflow as tf
import numpy as np
from util import metrics
from data_util import data_preprocess3
from model import Model
from args_8 import args
import nltk
import json
from nlgeval import NLGEval

nlgeval = NLGEval()  # loads the models
eps = 1e-10

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
model = Model(type="test", training_steps_per_epoch=None,
              vocabSize=data_preprocess.data.vocabSize)


config = tf.ConfigProto()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config.gpu_options.allow_growth = True


# ==============================================================================
#                               测试方法
# ==============================================================================
def id2word(data):
    ind2word = data_preprocess.data.ind2word
    new_data = ' '.join([ind2word[str(m)] for m in data if m != -1])
    return new_data


def F1_parlAI(p_answers, answers_str):
    # parlAI
    F1_score = []
    for k in range(len(p_answers)):
        p_a = p_answers[k]
        y_true = answers_str[k]

        # Determine the end position of p_answer
        j = 0
        for ii in range(len(p_a)):
            p_ai = p_a[ii]
            if p_ai == args.EOS:
                break
            j += 1
        y_pred = id2word(p_a[:j])
        F1_score.append(metrics._f1_score(y_pred, [y_true]))
    return F1_score


def distinctEval(all_paths):
    # distinct evaluation
    # all_paths is all answers' ID  [N, A_len]
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
                print("padding")
                w = topic_words[j-len(topic_words)]
            else:
                w = topic_words[j]

            # topic_words to embedding
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

            # topic_words to embedding
            # emb_index = data_preprocess.dic["token2id"][w]
            topic_words_emb.append(data_preprocess.embedding[w])

        batch_topic_words_emb.append(topic_words_emb)
    return batch_topic_words_emb


F1_score_all = []
dict1_all = []
dict2_all = []
dict3_all = []
for bbb in range(num_IterPerEpoch_train, num_IterPerEpoch_train*30+1, num_IterPerEpoch_train):
    with tf.Session(config=config) as sess:
        idxs = [i for i in range(num_sample)]
        print("model restore from savePath:", args.savePath)
        checkpoint_path = os.path.join(args.savePath, "visdial-%d" % bbb)
        if not os.path.exists(checkpoint_path + '.index'):
            exit(0)

        model.saver.restore(sess, checkpoint_path)

        F1_score = []
        model_answers_id = []
        true_answers = []
        model_answers = []

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

            output_feed = [model.answers_predict]
            outputs = sess.run(output_feed, input_feed)

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

        print("num_batch:", bbb, "beam_wide=", args.num_BeamSearch)
        model_metrics_dict = nlgeval.compute_metrics([true_answers], model_answers)
        print("model:\n", model_metrics_dict)

        print("F1_score:", np.mean(F1_score))
        F1_score_all.append(np.mean(F1_score))

        dist1, dist2, dist3 = distinctEval(model_answers_id)
        print("dist1:", dist1)
        print("dist2:", dist2)
        print("dist3:", dist3)
        dict1_all.append(dist1)
        dict2_all.append(dist2)
        dict3_all.append(dist3)

print("F1_score_all:", F1_score_all)
print("dict1_all:", dict1_all)
print("dict2_all:", dict2_all)
print("dict3_all:", dict3_all)

