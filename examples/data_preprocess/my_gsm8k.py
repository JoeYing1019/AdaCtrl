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

import os
import datasets

from hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


sys_diff = 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first assesses the difficulty (Easy or Hard) of given question, then thinks about the reasoning process in the mind, and finally provides the user with the answer. The difficulty, reasoning process, and answer are enclosed within [], <think> </think>, and <answer> </answer> tags, respectively, i.e., [difficulty here] <think> reasoning process here </think> <answer> answer here </answer>.'
sys = 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.'
sys_baseline = r"Please reason step by step, and put your final answer within \boxed{}."

instruction_following = "Answer the following Math Problem and put the answer in the format of \\boxed{answer}\n\n"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/mnt/bn/hl-joe/working_dir/diff_aware_new/data/eval')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    # data_source = 'aime24_nofigures'
    # print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(path='/mnt/bn/hl-joe/dataset/gsm8k/main/', data_files=['test.json'])

    train_dataset = dataset['train']
    # test_dataset = dataset['test']

    instruction_following = "Answer the following Math Problem and put the answer in the format of \\boxed{answer}\n\n"

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('question')
            # question = instruction_following + question

            answer = example.pop('answer')
            # solution = extract_solution(answer)
            solution = answer.split('####')[1].strip()
            data = {
                "data_source": "GSM8K",
                "prompt": [
                    {
                    "role": "system",
                    "content": sys_baseline
                    },
                    {
                    "role": "user",
                    "content": question
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': solution,
                    "question": question,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    # test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'gsm8k_baseline.parquet'))
    # test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
