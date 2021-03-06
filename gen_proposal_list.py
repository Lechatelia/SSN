import argparse
import os
from ops.io import process_proposal_list, parse_directory
from ops.utils import get_configs

# 因为官方提供的proposal是归一化的，也就是大小都是0-1之间的，
# 与自己提供的代码提取的帧数有关，乘以这个帧数就作为真正的proposal
parser = argparse.ArgumentParser(
    description="Generate proposal list to be used for training")
parser.add_argument('--dataset', type=str, default='thumos14', choices=['activitynet1.2', 'thumos14'])
parser.add_argument('--frame_path', type=str, default='/data/DataSets/THUMOS14/frames')

args = parser.parse_args()

configs = get_configs(args.dataset)

norm_list_tmpl = 'data/{}_normalized_proposal_list.txt'
out_list_tmpl = 'data/{}_proposal_list.txt'


if args.dataset == 'activitynet1.2':
    key_func = lambda x: x[-11:]
elif args.dataset == 'thumos14':
    key_func = lambda x: x.split('/')[-1]
else:
    raise ValueError("unknown dataset {}".format(args.dataset))


# parse the folders holding the extracted frames
frame_dict = parse_directory(args.frame_path, key_func=key_func)

process_proposal_list(norm_list_tmpl.format(configs['train_list']),
                      out_list_tmpl.format(configs['train_list']), frame_dict)
process_proposal_list(norm_list_tmpl.format(configs['test_list']),
                      out_list_tmpl.format(configs['test_list']), frame_dict)

print("proposal lists for dataset {} are ready for training.".format(args.dataset))
