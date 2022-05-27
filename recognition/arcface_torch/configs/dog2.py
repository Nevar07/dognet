from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "vit_b"
config.resume = False
config.output = "./models/vitb_00"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 256
config.lr = 0.05
config.verbose = 2000
config.dali = False
config.save_all_states = True
config.optimizer = 'sgd'

config.rec = "./data/train"
config.num_classes = 6000
config.num_image = 20000
config.num_epoch = 40
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
