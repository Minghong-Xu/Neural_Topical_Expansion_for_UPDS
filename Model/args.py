class args:
    def __init__(self, type):
        # 处理数据集 Person-chat
        self.type = type
        # data args
        # self.file = "train_self_original"
        self.file = self.type + "_self_original"
        self.data_load_path = "../Data/personachat/" + self.file + ".txt"
        self.data_save_path1 = "../Data/personachat_afterProcessing/" + self.file + ".json"
        self.data_save_path2 = "../Data/personachat_afterProcessing/" + self.file + ".h5"           # self.max_num_history_turns = 3 的记录(4_turn)
        # self.data_save_path2 = "../Data/personachat_afterProcessing/" + self.file + "_10turn.h5"
        self.save_vocab = "../Data/personachat_afterProcessing/train_self_original_vocab.json"
        self.personachat_save_path = "../Data/personachat_afterProcessing/topic_words/" + self.file + ".json"      # persona 中的topic words

        self.answer2persona_attention_label1 = "../Data/personachat_afterProcessing/" + self.file + "_answer2persona_attention_label.h5"
        self.answer2persona_attention_label2 = "../Data/personachat_afterProcessing/" + self.file + "_answer2persona_attention_label2.h5"
        self.answer2persona_attention_label2_1 = "../Data/personachat_afterProcessing/" + self.file + "_answer2persona_attention_label2_1.h5"
        self.answer2persona_attention_label2_1_try = "../Data/personachat_afterProcessing/" + self.file + "_answer2persona_attention_label2_1_try.h5"  # [500,5]

        self.answer2persona_attention_label3_1 = "../Data/personachat_afterProcessing/" + self.file + "_answer2persona_attention_label3_1.h5"  # 交并比
        self.answer2persona_attention_label3_2 = "../Data/personachat_afterProcessing/" + self.file + "_answer2persona_attention_label3_2.h5"  # 交并比 >0设置为1

        self.answer2persona_attention_label = "../Data/personachat_afterProcessing/" + self.file + "_answer2persona_attention_label3_2.h5"  # 交并比 >0设置为1

        # -------nueral_topic_model----self.num_topics = 50-------10000 words ------------
        self.num_topics = 50
        self.topic_model = "nueral"
        self.nueral_topic_model_embedding_path = "../Topic_model/save_nueral_50_10000/beta.npz"  # num_topics = 50
        self.nueral_topic_model_vocab_path = "../Topic_model/save_nueral_50_10000/vocab.json"  # num_topics = 50
        self.extend_topic_words = "../Data/personachat_afterProcessing/topic_words_nueral_50_10000/" + self.file + "_100.json"  # persona扩展出的words （num_topic=50，  num_topic_words=100）
        self.personachat_embedding_path = "../Data/personachat_afterProcessing/topic_embedding_neural/" + self.file + "_50_10000.h5"

        # self.topic_model = ""
        # self.savePath = "save_model1/checkpoints5/"  # encoder memory query 不加入下一步 return v_P
        self.savePath = "save_model1/checkpoints5_1/"  # encoder memory query 不加入下一步 return v_P  Model1_2 验证encoder里的memory retrieval
        # loss1=0.1, loss2=0.1
        # self.savePath = "save_model1/checkpoints1/"  # encoder memory query 不加入下一步   return query  hops=3
        # self.savePath = "save_model1/checkpoints1_1/"  # encoder memory query 不加入下一步    return tf.concat([v_P, v_EP],-1)
        # self.savePath = "save_model1/checkpoints1_2/"  # encoder memory query 不加入下一步  return v_P + v_EP
        # self.savePath = "save_model1/checkpoints2_old/"  # encoder memory query 加入下一步
        # loss1=0, loss2=0
        # self.savePath = "save_model1/checkpoints3/"  # encoder memory query 不加入下一步    return query
        # self.savePath = "save_model1/checkpoints3_1/"  # encoder memory query 不加入下一步    return tf.concat([v_P, v_EP],-1)
        # self.savePath = "save_model1/checkpoints3_2/"  # encoder memory query 不加入下一步  return v_P + v_EP
        # self.savePath = "save_model1/checkpoints4_old/"  # encoder memory query 加入下一步

        # loss1 = 0.1, loss2 = 0
        # self.savePath = "save_model1/checkpoints6/"  # encoder memory query 不加入下一步    return query
        # loss1 = 0,   loss2 = 0.1
        # self.savePath = "save_model1/checkpoints7/"  # encoder memory query 不加入下一步    return query

        # self.savePath = "save_model1/checkpoints2/"    # encoder memory query 不加入下一步   return query loss1=0.2, loss2=0.1 很差
        # self.savePath = "save_model1/checkpoints2_1/"  # encoder memory query 不加入下一步   return query loss1=0.2, loss2=0.2
        # self.savePath = "save_model1/checkpoints2_2/"  # encoder memory query 不加入下一步   return query loss1=0.1, loss2=0.05
        # self.savePath = "save_model1/checkpoints2_3/"    # encoder memory query 不加入下一步   return query loss1=0.1, loss2=0.1  hops=3  ***

        # self.savePath = "save_model1/checkpoints8_1/"  # encoder memory query 不加入下一步   return query loss1=0.1, loss2=0.1  hops=1  ***
        # self.savePath = "save_model1/checkpoints8_2/"  # encoder memory query 不加入下一步   return query loss1=0.1, loss2=0.1  hops=2  ***
        # self.savePath = "save_model1/checkpoints8_4/"  # encoder memory query 不加入下一步   return query loss1=0.1, loss2=0.1  hops=4  ***
        # self.savePath = "save_model1/checkpoints8_5/"  # encoder memory query 不加入下一步   return query loss1=0.1, loss2=0.1  hops=5  ***

        self.lamba_loss1 = 0              # bow_loss 的比重系数
        self.lamba_loss2 = 0              # attention_loss 的比重系数
        self.lamba_persona_weight = 1     # bow_loss 中 persona 词语增加的权重

        self.num_topic_words = 100
        self.topic_emb_threshold = 1e-4
        # 数据特征设置
        self.word_count_threshold = 0  # 构建词表所需最小词频
        # train为8939；valid为1000
        if self.type == "train":
            self.num_dialogues = 8939
            self.keep_prob = 0.8
        elif self.type == "valid":
            self.num_dialogues = 1000
            self.keep_prob = 1
        else:
            self.num_dialogues = 968
            self.keep_prob = 1
        self.max_num_persona = 5            # persona 最多包含五条
        self.max_num_personalength = 15     # persona 每条长度
        self.max_num_turns = 10             # 对话轮数
        self.max_num_history_turns = 3
        # self.max_num_history_turns = 9
        self.max_num_Qlength = 20
        self.max_num_Alength = 20

        # model 参数
        self.embedSize = 300

        self.rnnHiddenSize = 512
        # 目前只能只用GRU和单层，其他输出还没弄明白
        self.use_lstm = False
        # self.use_lstm = True
        self.rnnLayers = 1
        self.num_BeamSearch = 2
        self.beam_search = True

        self.PAD = 0
        self.GO = 1
        self.EOS = 2
        self.UNK = 3

        # train 参数设置
        self.num_Epochs = 50
        self.batch_size = 64
        self.learning_rate = 0.0001
        # self.learning_rate = 0.00005
        self.learning_rate_decay_factor = 0.9
        self.min_learning_rate = 0.00005
        self.max_gradient_norm = 5.0
        self.ckptInterval = 1000           # Store checkpoint for every ckptInterval batchs
