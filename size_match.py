"""
@Time:     2020/05/09 23:34
@Author:   Jinguo Zhu
@Email:    lechatelia@stu.xjtu.edu.cn
@File:     size_match.py
@Software: PyCharm

@description:
 solution from "https://github.com/yjxiong/tsn-pytorch/issues/74"
 size match error for BN pretrained model provided by author:
 RuntimeError: Error(s) in loading state_dict for BNInception:
	size mismatch for conv1_7x7_s2_bn.weight: copying a param with shape torch.Size([1, 64]) from checkpoint, the shape in current model is torch.Size([64]).
	size mismatch for conv1_7x7_s2_bn.bias: copying a param with shape torch.Size([1, 64]) from checkpoint, the shape in current model is torch.Size([64]).

first cp the bn_inception-9f5701afb96c8044.pth file to Pretrained directory
after run this script
change lines in model_zoo/BNInception/pytorch_load.py:
class BNInception(nn.Module):
def init(self, model_path='tf_model_zoo/bninception/bn_inception.yaml', num_classes=101,
weight_url = weight_fixed_path):
#weight_url='https://yjxiong.blob.core.windows.net/models/bn_inception-9f5701afb96c8044.pth'):
...
self.load_state_dict(torch.load(weight_url))
#self.load_state_dict(torch.utils.model_zoo.load_url(weight_url))
"""

import torch

# 预训练模型
# weight_path = 'Pretrained/bn_inception-9f5701afb96c8044.pth'
# weight_fixed_path = 'Pretrained/bn_inception-9f5701afb96c8044_fixed.pth'
# state_dict = torch.load(weight_path)
# for name, weights in state_dict.items():
#     if 'bn' in name:
#         print(name)
#         state_dict[name] = weights.squeeze(0)
# torch.save(state_dict, weight_fixed_path)

#用kinectics预训练的模型
weight_path = 'Pretrained/bninception_flow_kinetics_init-1410c1ccb470.pth'
weight_fixed_path = 'Pretrained/bninception_flow_kinetics_init-1410c1ccb470_fixed.pth'
state_dict = torch.load(weight_path)['state_dict']
for name, weights in state_dict.items():
    if 'bn' in name:
        print(name)
        state_dict[name] = weights.squeeze(0)
torch.save({'state_dict':state_dict}, weight_fixed_path)
