import pandas as pd
import pyarrow.parquet as pq
import json
import random
def read_jsonl(file_path):
    """
    读取JSONL文件并返回一个包含所有JSON对象的列表。
    参数:
        file_path (str): JSONL文件的路径。
    返回:
        list: 包含所有JSON对象的列表。
    """
    json_list = []
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            json_list.append(json_obj)
    return json_list

def parquet2json(parquet_path):
    """
    将Parquet文件转换为JSON列表。

    参数:
        parquet_path (str): Parquet文件的路径。

    返回:
        list: 包含Parquet数据的JSON列表。
    """
    # 读取Parquet文件
    table = pq.read_table(parquet_path)
    
    # 将Parquet数据转换为DataFrame
    df = table.to_pandas()
    
    # 将DataFrame转换为JSON格式的字符串
    json_str = df.to_json(orient='records')
    
    # 将JSON字符串解析为Python列表
    json_list = json.loads(json_str)
    
    return json_list

# datas = parquet2json('hdfs://haruna/home/byte_data_seed/hl_lq/user/fengjiazhan/alphaseed/data/ci_test_user_prompt_v3.math_release_1.0_250227.general_testset_0226.parquet')
# datas = parquet2json("hdfs://haruna/home/byte_data_seed/ssd_hldy/user/alphaseed/data/rl_dataset/aime2024_rule_verifier.parquet")
# datas = parquet2json("hdfs://haruna/home/byte_data_seed/ssd_hldy/user/alphaseed/data/rl_dataset/aime2025_rule_verifier.parquet")
# datas = parquet2json("hdfs://haruna/home/byte_data_seed/ssd_hldy/user/alphaseed/data/rl_dataset/gpqa_diamond_rule_verifier.parquet")

# datas = read_jsonl('/mnt/bn/hl-joe/working_dir/diff_aware/data/s1.1_train_diff_0405.json')
datas = parquet2json("/mnt/bn/hl-joe/working_dir/diff_aware_new/data/train/DeepMATH_baseline.parquet")
datas_2 = parquet2json("/mnt/bn/hl-joe/working_dir/diff_aware_new/data/eval/test_baseline.parquet")
print(len(datas))

# with open('/mnt/bn/hl-joe/dataset/gsm8k/main/test.json', 'w') as f:
#     for data in datas:
#         f.write(json.dumps(data) + '\n')



# path = '/mnt/bn/hl-joe/dataset/DeepMath-103K/data'
# import os
# files = os.listdir(path)
# datas = []
# for file in files:
#     if file.endswith('.json'):
#         continue
#     path_ = os.path.join(path, file)
#     data = parquet2json(path_)
#     datas.extend(data)


# diff_datas = {}
# for data in datas:
#     if data['difficulty'] not in diff_datas.keys():
#         diff_datas[data['difficulty']] = [data]
#     else:
#         diff_datas[data['difficulty']].append(data)

# random.seed(42)
# out_datas = []
# for key, data in diff_datas.items():
#         if len(data) > 1000:
#             sampled_data = random.sample(data, 1000)
#         else:
#             sampled_data = data
#         out_datas.extend(sampled_data)

# with open('/mnt/bn/hl-joe/dataset/DeepMath-103K/data/deepmath_selected.json', 'w') as f:
#     for data in out_datas:
#         f.write(json.dumps(data) + '\n')
