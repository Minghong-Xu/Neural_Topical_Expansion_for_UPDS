class args:
    def __init__(self, type):
        # Processing data set Person-chat
        self.type = type
        # data args
        self.file = self.type + "_self_original"
        self.data_load_path = "../Data/personachat/" + self.file + ".txt"
        self.data_save_path1 = "../Data/personachat_afterProcessing/" + self.file + ".json"
        self.data_save_path2 = "../Data/personachat_afterProcessing/" + self.file + ".h5"
        self.save_vocab = "../Data/personachat_afterProcessing/train_self_original_vocab.json"
        self.personachat_save_path = "../Data/personachat_afterProcessing/topic_words/" + self.file + ".json"      # topic words in persona

        self.answer2persona_attention_label = "../Data/personachat_afterProcessing/" + self.file + "_answer2persona_attention_label3_2.h5"  # 交并比 >0设置为1

        # -------nueral_topic_model----self.num_topics = 50-------10000 words ------------
        self.num_topics = 50
        self.topic_model = "nueral"
        # Refer to the code of "Neural models for documents with metadata" to train topic model and get beta and vocab
        self.nueral_topic_model_embedding_path = "../Topic_model/save_nueral_50_10000/beta.npz"  # num_topics = 50
        self.nueral_topic_model_vocab_path = "../Topic_model/save_nueral_50_10000/vocab.json"  # num_topics = 50
        self.extend_topic_words = "../Data/personachat_afterProcessing/topic_words_nueral_50_10000/" + self.file + "_100.json"  # extended persona words （num_topic=50，  num_topic_words=100）
        self.personachat_embedding_path = "../Data/personachat_afterProcessing/topic_embedding_neural/" + self.file + "_50_10000.h5"

        self.savePath = "save_model/checkpoints/"

        self.lamba_loss1 = 0.1              # bow_loss trade-off parameters γ2
        self.lamba_loss2 = 0.1              # attention_loss trade-off parameters γ1
        self.lamba_persona_weight = 1       # increase the weight of bow_loss λ

        self.num_topic_words = 100
        self.topic_emb_threshold = 1e-4

        self.word_count_threshold = 0  # Minimum word frequency required to build the vocabulary
        # train=8939；valid=1000
        if self.type == "train":
            self.num_dialogues = 8939
            self.keep_prob = 0.8
        elif self.type == "valid":
            self.num_dialogues = 1000
            self.keep_prob = 1
        else:
            self.num_dialogues = 968
            self.keep_prob = 1
        self.max_num_persona = 5            # persona Contains up to five
        self.max_num_personalength = 15     # persona
        self.max_num_turns = 10             # dialogue tuens
        self.max_num_history_turns = 3
        # self.max_num_history_turns = 9
        self.max_num_Qlength = 20
        self.max_num_Alength = 20

        self.embedSize = 300
        self.rnnHiddenSize = 512
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
        self.learning_rate_decay_factor = 0.9
        self.min_learning_rate = 0.00005
        self.max_gradient_norm = 5.0
        self.ckptInterval = 1000
