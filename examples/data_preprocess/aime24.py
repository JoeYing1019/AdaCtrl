# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import random
import pandas as pd


from hdfs_io import copy, makedirs
import argparse
import json
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

sys_diff = 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first assesses the difficulty (Easy or Hard) of given question, then thinks about the reasoning process in the mind, and finally provides the user with the answer. The difficulty, reasoning process, and answer are enclosed within [], <think> </think>, and <answer> </answer> tags, respectively, i.e., [difficulty here] <think> reasoning process here </think> <answer> answer here </answer>.'
sys = 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.'
sys_baseline = r"Please reason step by step, and put your final answer within \boxed{}."

instruction_following = "Answer the following Math Problem and put the answer in the format of \\boxed{answer}\n\n"

def process_prompt(row):
    """处理每行数据，修改prompt字段"""
    # 获取原始prompt内容
    original_prompt = row['prompt']
    # if not original_prompt:  # 如果prompt为空，直接返回原行
    #     return row
    
    # 提取原始用户问题
    user_content = original_prompt[0].get('content', '')
    user_content = user_content.replace("Answer the following Math Problem and put the answer in the format of \\boxed{answer}\n\n", "")
    # user_content = instruction_following + user_content
    
    # 根据reward_model.style选择system_content
    system_content = sys_baseline
    # 构造新的prompt结构
    new_prompt = [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': user_content}
    ]
    row['prompt'] = new_prompt
    old_reward = json.loads(row['reward_model']["ground_truth"])
    row['reward_model'] = {
                "style": "rule",
                "ground_truth": old_reward['reference_answer']
            }
    # row['reward_model']['style'] = "rule"
    row['data_source'] = 'AIME2024'
    return row

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='hdfs://haruna/home/byte_data_seed/ssd_hldy/user/alphaseed/data/rl_dataset/aime2024_verifier.parquet')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    df = pd.read_parquet(args.local_dir)

    # 处理每行数据
    df = df.apply(process_prompt, axis=1)
    
    # 保存处理后的数据
    df.to_parquet('/mnt/bn/hl-joe/working_dir/diff_aware_new/data/eval/aime24_baseline.parquet', index=False)

