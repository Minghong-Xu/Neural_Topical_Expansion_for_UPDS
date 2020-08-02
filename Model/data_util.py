# coding=utf-8
# Processing Database：persona-chat
import json
import os
import h5py
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from args_8 import args
from nltk.corpus import stopwords
import string


def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


stop = set(stopwords.words('english'))
exclude = set(string.punctuation + string.digits +
              r"""！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.""")


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class data_preprocess():
    # txt is processed into formatted json file, the file is self.data
    def __init__(self, args):
        self.args = args
        self.data = self.load_data()

    def load_data(self):
        if not os.path.exists(self.args.data_save_path1):
            print("[loading dialog data:" + self.args.data_load_path + "]")
            data = []
            x = {}
            your_persona = []
            dialogues = []
            reward = 0
            print("Processing initial information...")
            with open(self.args.data_load_path) as read:
                last_conv_id = None
                for line in read:
                    line = line.strip().replace('\\n', '\n')
                    if len(line) == 0:
                        # empty response
                        continue
                    # first, get conversation index -- '1' means start of episode
                    space_idx = line.find(' ')

                    if space_idx == -1:
                        # empty line, both individuals are saying whitespace
                        conv_id = int(line)
                    else:
                        conv_id = int(line[:space_idx])

                    # split line into constituent parts, if available:
                    split = line[space_idx + 1:].split('\t')

                    # remove empty items and strip each one
                    for i in range(len(split)):
                        word = split[i].strip()
                        if len(word) == 0:
                            split[i] = ''
                        else:
                            split[i] = word

                    # now check if we're at a new episode
                    if last_conv_id is not None and conv_id <= last_conv_id:
                        x["your_persona"] = your_persona
                        x["dialogues"] = dialogues
                        data.append(x)
                        x = {}
                        your_persona = []
                        dialogues = []
                        reward = 0

                    if len(split) > 1 and split[1]:
                        # Process dialogues QA-pairs
                        dialogue = {}
                        # only generate an example if we have a y
                        dialogue["Q"] = split[0]
                        # split labels
                        # dialogue["A"] = split[1].split('|')
                        dialogue["A"] = split[1]

                        # Empty reward string same as None
                        if len(split) > 2 and split[2] == '':
                            split[2] = None
                        # calculate reward
                        if len(split) > 2 and split[2]:
                            reward += float(split[2])
                        dialogue["reward"] = reward
                        if len(split) > 3:
                            # split label_candidates
                            dialogue["candidates"] = split[3].split('|')
                        dialogues.append(dialogue)
                    else:
                        # process profile
                        start_idx = split[0].find(':')
                        if start_idx == -1:
                            # empty line, both individuals are saying whitespace
                            persona = ""
                        else:
                            persona = split[0][start_idx + 1:]

                        your_persona.append(persona)

                    last_conv_id = conv_id

                x["your_persona"] = your_persona
                x["dialogues"] = dialogues
                data.append(x)
            print("The processing information of the first step is completed and saved")
            self.write(data)
        else:
            print("[loading dialog data:" + self.args.data_save_path1 + "]")
            fdata = open(self.args.data_save_path1, encoding='UTF-8')
            data = json.load(fdata)
        return data

    def write(self, data):
        with open(self.args.data_save_path1, 'w') as json_file:
            # json_file.write(json_str)
            # json.dump(data, json_file)
            json.dump(data, json_file, ensure_ascii=False)

    def ans_in_persoan(self):
        answers_in_persona = []
        for d in self.data:
            persona = []
            for p in d["your_persona"]:
                p_tokens = word_tokenize(p)
                persona.extend(p_tokens)
            persona = self.clean(persona)
            # print(persona)

            answer_turn = []
            dias = d["dialogues"]
            for dia in dias:
                a_tokens = self.clean(word_tokenize(dia["A"]))
                # words appear in answer and persona at the same time
                a_tokens = [a for a in a_tokens if a in persona]

                print(a_tokens)
                answer_turn.append(a_tokens)
            answers_in_persona.append(answer_turn)
        return answers_in_persona

    def clean(self, doc):  # Remove punctuation, stop words
        stop_free = " ".join([i for i in doc if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        punc_free = word_tokenize(punc_free)
        return punc_free


class data_preprocess2():
    # Divide data into person, history, question, etc. and store them in .h5 files
    def __init__(self, args):
        self.args = args
        self.personas, self.num_personas, self.personas_len, self.questions, \
        self.questions_len, self.answers, self.answers_len, self.dialogs_len \
            = self.load_data()

    def load_data(self):
        if not os.path.exists(self.args.data_save_path2):
            if not os.path.exists(self.args.data_save_path1):
                data_preprocess(self.args)
            print("[loading dialog data:" + self.args.data_save_path1 + "]")
            personas = []
            num_personas = []
            personas_len = []
            questions = []
            questions_len = []
            answers = []
            answers_len = []
            dialogs_len = []
            word_counter = Counter()
            with open(self.args.data_save_path1, encoding='UTF-8') as f:
                data = json.load(f)
                # word_tokenize and build word_counter
                for d in data:
                    # persona
                    persona = []
                    persona_len = []
                    for p in d["your_persona"]:
                        p_tokens = word_tokenize(p)
                        for k in p_tokens:
                            word_counter[k] +=1
                        persona.append(p_tokens)
                        persona_len.append(min(len(p_tokens), self.args.max_num_personalength))
                    personas.append(persona)
                    personas_len.append(persona_len)
                    num_personas.append(len(persona))

                    # question and answer
                    question = []
                    question_len = []
                    answer = []
                    answer_len = []
                    dialogs_len.append(len(d["dialogues"]))     # dialogue turn
                    for j in range(len(d["dialogues"])):
                        # question [i, j, k]
                        q_tokens = word_tokenize(d["dialogues"][j]["Q"])
                        for k in q_tokens:
                            word_counter[k] += 1
                        question.append(q_tokens)
                        question_len.append(min(len(q_tokens), self.args.max_num_Qlength))

                        # answer
                        a_tokens = word_tokenize(d["dialogues"][j]["A"])
                        for k in a_tokens:
                            word_counter[k] += 1
                        answer.append(a_tokens)
                        answer_len.append(min(len(a_tokens), self.args.max_num_Alength))
                    questions.append(question)
                    questions_len.append(question_len)
                    answers.append(answer)
                    answers_len.append(answer_len)

            # Get vocabulary
            if self.args.type == "train":
                # train: build vocabulary
                self.get_dict(word_counter)
            else:
                # test：use vocabulary
                out = json.load(open(self.args.save_vocab, 'r'))
                self.word2ind = out["word2int"]
                self.ind2word = out["int2word"]
                self.vocabSize = len(self.word2ind)

            # word to index
            personas = self.word2int(personas, self.args.max_num_persona, self.args.max_num_personalength)
            questions = self.word2int(questions, self.args.max_num_turns, self.args.max_num_Qlength)
            answers = np.array(self.word2int(answers, self.args.max_num_turns, self.args.max_num_Alength))

            # padding
            personas_len = self.x_len_padding(personas_len, self.args.max_num_persona)
            questions_len = self.x_len_padding(questions_len, self.args.max_num_turns)
            answers_len = self.x_len_padding(answers_len, self.args.max_num_turns )

            print('Saving hdf5...')
            fsave = h5py.File(self.args.data_save_path2, 'w')
            fsave.create_dataset('personas', dtype='int', data=personas)
            fsave.create_dataset('num_personas', dtype='int', data=num_personas)
            fsave.create_dataset('personas_len', dtype='int', data=personas_len)
            fsave.create_dataset('questions', dtype='int', data=questions)
            fsave.create_dataset('questions_len', dtype='int', data=questions_len)
            fsave.create_dataset('answers', dtype='int', data=answers)
            fsave.create_dataset('answers_len', dtype='int', data=answers_len)
            fsave.create_dataset('dialogs_len', dtype='int', data=dialogs_len)

            out = {}
            out["word2int"] = self.word2ind
            out["int2word"] = self.ind2word
            json.dump(out, open(self.args.save_vocab, 'w'))

        else:
            print("[loading dialog data:" + self.args.data_save_path2 + "]")
            File = h5py.File(self.args.data_save_path2, 'r')
            personas = np.array(File["personas"])
            num_personas = np.array(File["num_personas"])
            personas_len = np.array(File["personas_len"])
            questions = np.array(File["questions"])
            questions_len = np.array(File["questions_len"])
            answers = np.array(File["answers"])
            answers_len = np.array(File["answers_len"])
            dialogs_len = np.array(File["dialogs_len"])

            out = json.load(open(self.args.save_vocab, 'r'))
            self.word2ind = out["word2int"]
            self.ind2word = out["int2word"]
            self.vocabSize = len(self.word2ind)

        return personas, num_personas, personas_len, questions, questions_len, answers, answers_len, dialogs_len

    def get_dict(self, word_counter):
        print('Building vocabulary...')
        _PAD = "<PAD>"
        _GO = "<GO>"
        _EOS = "<EOS>"
        _UNK = "<UNK>"
        _START_VOCAB = [_PAD, _GO, _EOS, _UNK]

        vocab = [word for word in word_counter if word_counter[word] >= self.args.word_count_threshold]
        vocab = _START_VOCAB + vocab
        print('Words: %d' % len(vocab))
        self.word2ind = {word: word_ind for word_ind, word in enumerate(vocab)}
        self.ind2word = {word_ind: word for word, word_ind in self.word2ind.items()}
        self.vocabSize = len(self.word2ind)

        # glove： Filter related words
        word2ind = self.word2ind
        word2vec_dict = {}
        with open("../Data/glove.840B.300d.txt", 'r', encoding='utf-8') as fh:
            for line in fh:
                array = line.lstrip().rstrip().split(" ")
                word = array[0]  # 词语
                vector = list(map(float, array[1:]))
                if word in word2ind.keys():
                    word2vec_dict[word] = vector
                if word.lower() in word2ind.keys():
                    word2vec_dict[word.lower()] = vector

        print(word2vec_dict["pet"])
        print(word2ind["pet"])  # 1366
        print("word2vec_dict_len", len(word2vec_dict))
        print("word2ind_len", len(word2ind))

        embedding = np.zeros([len(word2ind), 300])
        for w in word2ind:
            id = int(word2ind[w])
            if w in word2vec_dict.keys():
                emb = word2vec_dict[w]
            else:
                print(w)
                emb = np.random.uniform(-1, 1, (300))
            embedding[id] = emb

        print('Saving hdf5...')
        fsave = h5py.File("../Data/glove_train.h5", 'w')
        fsave.create_dataset('embedding', dtype='float32', data=embedding)

    def word2int(self, data, num1, num2):
        data_int = np.zeros([self.args.num_dialogues, num1, num2], int)
        for i in range(min(self.args.num_dialogues, len(data))):
            for j in range(min(num1,len(data[i]))):
                for k in range(min(num2,len(data[i][j]))):
                    if data[i][j][k] in self.word2ind:
                        data_int[i][j][k] = self.word2ind[data[i][j][k]]
                    else:
                        data_int[i][j][k] = self.word2ind["<UNK>"]
        return data_int

    def x_len_padding(self, data, num):
        data_int = np.zeros([self.args.num_dialogues, num], int)
        for i in range(min(self.args.num_dialogues, len(data))):
            for j in range(min(num, len(data[i]))):
                data_int[i][j] = data[i][j]
        return data_int


class data_preprocess3():
    # After the data is merged by history, the batch is obtained
    def __init__(self, args):
        self.args = args
        self.data = data_preprocess2(self.args)

        # add <END> to the end of answer_target
        answer_targets = np.zeros((self.args.num_dialogues, self.args.max_num_turns, self.args.max_num_Alength + 1), dtype=np.int)
        for d in range(self.args.num_dialogues):
            for r in range(self.args.max_num_turns):
                answer_targets[d][r] = np.insert(self.data.answers[d][r], self.data.answers_len[d][r], self.data.word2ind['<EOS>'])
        answer_targets = answer_targets

        # string
        answer_str, answers_in_persona = self.get_answer_info()

        self.personas, self.personas_len, self.personas_turn, self.historys, self.historys_len, self.historys_turn, \
        self.questions, self.questions_len, self.answers, self.answers_len, self.answers_target, self.answers_str, \
        self.answers_in_persona = self.merage(answer_targets, answer_str, answers_in_persona)
        self.num_sample = len(self.questions)

        # 取answer对persona做attention的标签
        print("[loading dialog data:", self.args.answer2persona_attention_label, "]")
        File = h5py.File(self.args.answer2persona_attention_label, 'r')
        self.attention_label = np.array(File["answer2persona_attention_label"])     # (65607, 5)

        if self.args.topic_model == "lda":
            print("[loading dialog data:" + self.args.save_LDA_result + "]")
            fdata = open(self.args.save_LDA_result, encoding='UTF-8')
            data = json.load(fdata)
            self.embedding = np.array(data["emb"])
            self.dic = data["dic"]
            self.personas_emb, self.historys_emb, self.questions_emb, self.answers_emb, self.topic_words, self.topic_words_weigth = self.get_topic_embedding()
        elif self.args.topic_model == "nueral":
            embedding = np.load(args.nueral_topic_model_embedding_path)
            self.embedding = embedding["beta"].T
            # print(self.embedding[:2])
            f = open(self.args.nueral_topic_model_vocab_path, encoding='UTF-8')
            vocab = json.load(f)
            token2id = {word: word_ind for word_ind, word in enumerate(vocab)}
            id2token = {str(word_ind): word for word, word_ind in token2id.items()}
            self.dic = {"token2id": token2id,
                        "id2token": id2token}
            self.personas_emb, self.historys_emb, self.questions_emb, self.answers_emb, self.topic_words, self.topic_words_weigth = self.get_topic_embedding()

    def get_answer_info(self):
        # answers_str： Get string representations of answers
        # answers_in_persona: Get the words that appear in persona and the answer (remove stopwords)
        if not os.path.exists(self.args.data_save_path1):
            data_preprocess(self.args)
        print("[loading dialog data:" + self.args.data_save_path1 + "]")

        with open(self.args.data_save_path1, encoding='UTF-8') as f:
            data = json.load(f)
            answers_str = []
            answers_in_persona = []
            for d in data:
                # Calculate answer_in_persona(uesd in bow-loss-label)
                persona = []
                for p in d["your_persona"]:
                    p_tokens = word_tokenize(p)
                    persona.extend(p_tokens)
                persona = self.clean(persona)
                # print(persona)

                answer_str = []
                answer_turn = []
                dias = d["dialogues"]
                for dia in dias:
                    answer_str.append(dia["A"])
                    a_tokens = self.clean(word_tokenize(dia["A"]))
                    bow_label = np.zeros(self.data.vocabSize)
                    for a in a_tokens:
                        if a in persona and a in self.data.word2ind.keys():
                            a_id = self.data.word2ind[a]
                            bow_label[a_id] = 1
                    answer_turn.append(bow_label)
                answers_str.append(answer_str)      # [num_dialogue, num_turn] string
                answers_in_persona.append(answer_turn)      # [num_dialogue, num_turn, vocabSize]
        return answers_str, answers_in_persona

    def clean(self, doc):
        stop_free = " ".join([i for i in doc if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        # punc_free = word_tokenize(punc_free)
        punc_free = punc_free.split()
        return punc_free

    def merage(self, answer_targets, answer_str, answer_in_persona):
        personas, personas_len, personas_turn, historys, historys_len, historys_turn = [], [], [], [], [], []
        questions, questions_len, answers, answers_len, answers_target, answers_str = [], [], [], [], [], []
        answers_in_persona = []
        for i in range(len(self.data.personas)):
            for j in range(min(self.args.max_num_turns, self.data.dialogs_len[i])):
                personas.append(self.data.personas[i])
                personas_len.append(self.data.personas_len[i])
                personas_turn.append((self.data.num_personas[i]))

                history = []
                history_len = []
                history_turn = 1
                # Historical information up to 3 turns
                for k in range(max(0, j - self.args.max_num_history_turns), j):
                    history.append(self.data.questions[i][k])
                    history_len.append(self.data.questions_len[i][k])
                    history.append(self.data.answers[i][k])
                    history_len.append(self.data.answers_len[i][k])
                    history_turn += 2
                history.append(self.data.questions[i][j])       # history contains query
                history_len.append(self.data.questions_len[i][j])
                # padding to 7
                while len(history) < (2 * self.args.max_num_history_turns + 1):
                    history.append(np.zeros([self.args.max_num_Qlength], int))
                    history_len.append(0)

                historys.append(history)
                historys_len.append(history_len)
                historys_turn.append(history_turn)

                questions.append(self.data.questions[i][j])
                questions_len.append(self.data.questions_len[i][j])
                answers.append(self.data.answers[i][j])
                answers_len.append(self.data.answers_len[i][j])
                answers_target.append(answer_targets[i][j])
                answers_str.append(answer_str[i][j])
                answers_in_persona.append(answer_in_persona[i][j])

        return personas, personas_len, personas_turn, historys, historys_len, historys_turn, \
               questions, questions_len, answers, answers_len, answers_target, answers_str, answers_in_persona

    def get_topic_embedding(self):
        print("[loading dialog data:" + self.args.personachat_embedding_path + "]")
        File = h5py.File(self.args.personachat_embedding_path, 'r')
        personas_topic_embedding = np.array(File["personas_topic_embedding"])
        questions_topic_embedding = np.array(File["questions_topic_embedding"])
        answers_topic_embedding = np.array(File["answers_topic_embedding"])

        personas_topicwords, personas_topicwords_weigth = self.get_topic_words()      # Get words and weights
        # personas_topicwords, personas_topicwords_weigth = self.get_topic_words2()        # Get word ids and weights
        # print("personas_topicwords:")
        # print(personas_topicwords[:10])

        personas_emb, historys_emb, questions_emb, answers_emb = [], [], [], []
        personas_topic_words, personas_topic_words_weight = [], []       # topic words in persona
        for i in range(len(self.data.personas)):
            for j in range(min(self.args.max_num_turns, self.data.dialogs_len[i])):
                personas_emb.append(personas_topic_embedding[i])
                personas_topic_words.append(personas_topicwords[i])
                personas_topic_words_weight.append(personas_topicwords_weigth[i])

                history_emb = []
                for k in range(max(0, j - self.args.max_num_history_turns), j):
                    history_emb.append(questions_topic_embedding[i][k])
                    history_emb.append(answers_topic_embedding[i][k])
                history_emb.append(questions_topic_embedding[i][j])

                while len(history_emb) < (2 * self.args.max_num_history_turns + 1):
                    history_emb.append(np.zeros([self.args.num_topics], int))

                historys_emb.append(history_emb)
                questions_emb.append(questions_topic_embedding[i][j])
                answers_emb.append(answers_topic_embedding[i][j])

        return personas_emb, historys_emb, questions_emb, answers_emb, personas_topic_words, personas_topic_words_weight

    def get_batch(self, batch_idx):
        batch_persona, batch_persona_len, batch_persona_turn, batch_history, batch_history_len, batch_history_turn = [], [], [], [], [], []
        batch_question, batch_question_len, batch_answer, batch_answer_len, batch_answer_target, batch_answers_str = [], [], [], [], [], []
        batch_answers_in_persona = []
        for idx in batch_idx:
            batch_persona.append(self.personas[idx])
            batch_persona_len.append(self.personas_len[idx])
            batch_persona_turn.append(self.personas_turn[idx])
            batch_history.append(self.historys[idx])
            batch_history_len.append(self.historys_len[idx])
            batch_history_turn.append(self.historys_turn[idx])
            batch_question.append(self.questions[idx])
            batch_question_len.append(self.questions_len[idx])
            batch_answer.append(self.answers[idx])
            batch_answer_len.append(self.answers_len[idx])
            batch_answer_target.append(self.answers_target[idx])
            batch_answers_str.append(self.answers_str[idx])
            batch_answers_in_persona.append(self.answers_in_persona[idx])

        return batch_persona, batch_persona_len, batch_persona_turn, batch_history, batch_history_len, batch_history_turn, \
               batch_question, batch_question_len, batch_answer, batch_answer_len, batch_answer_target, batch_answers_str, \
               batch_answers_in_persona

    def get_batch_topic_emb(self, batch_idx):
        batch_personas_emb, batch_historys_emb, batch_questions_emb = [], [], []
        for idx in batch_idx:
            batch_personas_emb.append(self.personas_emb[idx])
            batch_historys_emb.append(self.historys_emb[idx])
            batch_questions_emb.append(self.questions_emb[idx])
            # batch_answers_emb.append(self.answers_emb[idx])
        return batch_personas_emb, batch_historys_emb, batch_questions_emb

    def get_batch_topic_info(self, batch_idx):
        batch_personas_emb, batch_historys_emb, batch_questions_emb = [], [], []
        batch_topic_words, batch_topic_words_weigth = [], []
        for idx in batch_idx:
            batch_personas_emb.append(self.personas_emb[idx])
            batch_historys_emb.append(self.historys_emb[idx])
            batch_questions_emb.append(self.questions_emb[idx])
            # batch_answers_emb.append(self.answers_emb[idx])
            batch_topic_words.append(self.topic_words[idx])
            batch_topic_words_weigth.append(self.topic_words_weigth[idx])
        return batch_personas_emb, batch_historys_emb, batch_questions_emb, batch_topic_words, batch_topic_words_weigth

    def get_batch_attention_label(self, batch_idx):
        batch_attention_label = []
        for idx in batch_idx:
            batch_attention_label.append(self.attention_label[idx])
        return batch_attention_label

    def get_topic_words(self):
        if not os.path.exists(self.args.extend_topic_words):
            print("[loading dialog data:" + self.args.personachat_save_path + "]")
            fdata = open(self.args.personachat_save_path, encoding='UTF-8')
            personas_topic_words = json.load(fdata)["personas_topicwords"]
            # Use topic words in persona to expand related topic words
            topic_words = []
            topic_weight = []
            for i in range(len(personas_topic_words)):  # num of dialogue
                topic_words_dict = Counter()
                persona_topic_words = [y for x in personas_topic_words[i] for y in x if y in self.dic["token2id"].keys()]
                persona_topic_words = list(set(persona_topic_words))  # Deduplication
                # print("persona_topic_words:", persona_topic_words)
                if len(persona_topic_words) == 0:
                    words = []
                    weight = []
                    print("All words are not in vocab")
                    print("personas_topic_words[i]:", personas_topic_words[i])
                else:
                    k = int(self.args.num_topic_words / len(persona_topic_words)) + 10  # how many words are selected, dynamically determined
                    while len(topic_words_dict) < self.args.num_topic_words:  # k Increase to select num_topic_words words
                        # print("k:", k)
                        for word in persona_topic_words:
                            # every persona words extend k topic words
                            dict = self.get_one_topic_words(word, k)
                            # Take the union, and take the maximum weight at this time
                            topic_words_dict = Counter(dict) | topic_words_dict
                        k = k + 5
                    if len(topic_words_dict) < self.args.num_topic_words:
                        print("Not enough length:", len(topic_words_dict))

                    # Take the most weighted num_topic_words words
                    topic_words_dict = topic_words_dict.most_common(self.args.num_topic_words)
                    words = []
                    weight = []
                    for dict in topic_words_dict:
                        words.append(dict[0])
                        weight.append(dict[1])
                if i % 100 == 0:
                    print(i)
                # print("persona_topic_words:", persona_topic_words)
                # print("words:", words)
                topic_words.append(words)
                topic_weight.append(weight)
            with open(self.args.extend_topic_words, "w", encoding='UTF-8') as file:
                data = {"topic_words": topic_words,
                        "topic_weight": topic_weight}
                # json.dump(data, file, ensure_ascii=False)
                json.dump(data, file, ensure_ascii=False, cls=MyEncoder)
                print("save in ", self.args.extend_topic_words)
        else:
            with open(self.args.extend_topic_words, encoding='UTF-8') as file:
                print("[loading dialog data:" + self.args.extend_topic_words + "]")
                data = json.load(file)
                topic_words = data["topic_words"]
                topic_weight = data["topic_weight"]
        # print(topic_words[:3])
        return topic_words, topic_weight        # [N_dialogue, num_topic_words]

    def get_one_topic_words(self, topic_word, k):
        topic_words_dict = {}
        topic_vector = self.embedding[self.dic["token2id"][topic_word]]
        # print("topic_vector:", topic_vector)

        topic_vector = topic_vector.reshape(1, -1)
        att_weight = cosine_similarity(self.embedding, topic_vector)
        att_weight = np.squeeze(att_weight)
        # print("att_weight2", att_weight2[:10])

        att_weight_index = np.argsort(att_weight)[-k:]
        # print(att_weight_index)
        for index in att_weight_index:
            word = self.dic["id2token"][str(index)]
            # word = self.dic["id2token"][index]
            if att_weight[index] < 1.0:
                # Remove the word itself, part of the low frequency will also have the same dimension
                topic_words_dict[word] = att_weight[index]
        # print("topic_words_dict:", topic_words_dict)
        return topic_words_dict

    def cos_sim(self, vector_a, vector_b):
        """
        :param vector_a: vector a
        :param vector_b: vector b
        :return: sim
        """
        vector_a = np.mat(vector_a)
        vector_b = np.mat(vector_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        sim = num / denom
        return sim


def get_glove_embedding(args):
    if not os.path.exists("../Data/glove_train.h5"):
        word2ind = data_preprocess2(args).word2ind
        word2vec_dict = {}
        with open("../Data/glove.840B.300d.txt", 'r', encoding='utf-8') as fh:
            for line in fh:
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in word2ind.keys():
                    word2vec_dict[word] = vector
                if word.lower() in word2ind.keys():
                    word2vec_dict[word.lower()] = vector

        print(word2vec_dict["pet"])
        print(word2ind["pet"])      # 1366
        print("word2vec_dict_len", len(word2vec_dict))
        print("word2ind_len", len(word2ind))

        embedding = np.zeros([len(word2ind), 300])
        for w in word2ind:
            id = int(word2ind[w])
            if w in word2vec_dict.keys():
                emb = word2vec_dict[w]
            else:
                print(w)
                emb = np.random.uniform(-1, 1, (300))
            embedding[id] = emb

        print('Saving hdf5...')
        fsave = h5py.File("../Data/glove_train.h5", 'w')
        fsave.create_dataset('embedding', dtype='float32', data=embedding)
    else:
        File = h5py.File("../Data/glove_train.h5", 'r')
        embedding = np.array(File["embedding"])
    return embedding


if __name__ == '__main__':
    args = args("train")
    # args = args("test")
    pre = data_preprocess(args)
    # pre.ans_in_persoan()
    data = pre.data
    for d in data:
        ps = d["your_persona"]
        persona = []
        for p in ps:
            print("persona:", p)
        dias = d["dialogues"]
        for dia in dias:
            print("Q:", dia["Q"])
            print("A:", dia["A"])



