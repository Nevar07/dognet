from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "swtb"
config.resume = False
config.output = "./models/swtb_1"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 32
config.lr =0.0002 #0.025
config.verbose = 2000
config.dali = False
config.save_all_states = False
config.optimizer = 'sgd'

config.rec = "./data/arc/train"
config.num_classes = 6000
config.num_image = 19500
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
