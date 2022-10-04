import torch

param = torch.load("/home/yangyang/sangmeng/project/face_recognization/arcface_torch/work_dirs/wf_r100_lr_02_batch#128/softmax_fc_gpu_0.pt","cpu")
print(param)