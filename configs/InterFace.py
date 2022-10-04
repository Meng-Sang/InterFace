import math

from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
#
config.dataset = "MS1MV1"
config.loss = "InterFace&dd"
if config.loss.startswith("InterFace"):
	if config.loss.endswith("cc"):
		config.mid = 0.2
	elif config.loss.endswith("dc"):
		config.mid = 0.3
	elif config.loss.endswith("dd"):
		config.mid = math.pi / 18 * 1
	else:
		raise Exception("loss type is error")
elif config.loss.startswith("arcface"):
	pass
else:
	raise Exception("loss type is error")
# config.loss = "ItiFace&cd"
# config.loss = "ItiFace&dd"3

config.alpha = 0.1
config.m = 0.5
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 512
config.lr = 0.1

if config.dataset == "MS1MV1":
	config.milestones = [20, 28, 32]
	config.verbose = 958
	config.dali = False
	config.rec = "/home/yangyang/sangmeng/dataset/faces_webface_112x112"
	config.num_classes = 10572
	config.num_image = 490623
	config.num_epoch = 34
	config.warmup_epoch = 0
	config.val_targets = ['lfw', "agedb_30", "calfw", "cplfw", 'cfp_fp']
	config.val_targets = []
elif config.dataset == "MS1MV2":
	config.milestones = [8, 14, 20, 25]
	config.verbose = 2000
	config.rec = "/home/yangyang/sangmeng/dataset/faces_emore"
	config.num_classes = 85742
	config.num_image = 5822653
	config.num_epoch = 26
	config.warmup_epoch = 0
	config.val_targets = ['lfw', "agedb_30", "calfw", "cplfw", 'cfp_fp']
else:
	raise(Exception(f"this project is not support {config.dataset} dataset"))