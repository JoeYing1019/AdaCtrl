import json
import pandas as pd
import numpy as np
from collections import defaultdict
# 每一行格式 "{path}, {repeat_num}'
paths = """
/mnt/bn/hl-joe/working_dir/diff_aware_new/data/eval/aime24_baseline.parquet, 8
/mnt/bn/hl-joe/working_dir/diff_aware_new/data/eval/aime25_baseline.parquet, 8
/mnt/bn/hl-joe/working_dir/diff_aware_new/data/eval/gsm8k_baseline.parquet, 1
/mnt/bn/hl-joe/working_dir/diff_aware_new/data/eval/math500_baseline.parquet, 1
"""
# 把各来源数据join到一块
paths = list(map(lambda x: x.split(","), filter(lambda x: x.strip(), paths.split("\n"))))
df = pd.DataFrame()
necessary_keys = ["ability", "data_source", "prompt", "reward_model"]
for (path, repeat_num) in paths:
    repeat_num = int(repeat_num)
    cur_df = pd.read_parquet(path)
    for key in necessary_keys:
        assert key in cur_df.keys()
    # 设置index，用于后续算法的group操作
    cur_df.loc[:, "extra_info"] = np.array(
        [{"index": cur_df.iloc[i]["data_source"] + "-" + str(i)} for i in range(len(cur_df))])
    df = pd.concat([df] + [cur_df] * repeat_num, ignore_index=True)
# 打散
df = df.sample(frac=1).reset_index(drop=True)

# 设置verify_type, 有<answer><\answer>格式的设置成3，否则设置成4
data_source_verify_type_cnt = defaultdict(float)
verify_type = 3
reward_model = df["reward_model"]
new_reward_model = []
for rm, data_source in zip(reward_model, df["data_source"]):
    if "verify_type" in rm["ground_truth"]:
        if type(rm["ground_truth"]) == str:
            gt = json.loads(rm["ground_truth"])
        else:
            gt = rm["ground_truth"]
        gt["verify_type"] = verify_type
        rm["ground_truth"] = json.dumps(gt, ensure_ascii=False)
        data_source_verify_type_cnt[data_source] += 1
    new_reward_model.append(rm)
df.loc[:, "reward_model"] = new_reward_model

# 确认verify_type设置正确
verify_type_cnt = defaultdict(float)
for rm in df["reward_model"]:
    if "verify_type" in rm["ground_truth"]:
        verify_type = json.loads(rm["ground_truth"])["verify_type"]
        verify_type_cnt[verify_type] += 1
print(verify_type_cnt)

# 统计各来源数据的分布
abilitys = defaultdict(int)
data_sources = defaultdict(int)
for ability, data_source in zip(df["ability"], df["data_source"]):
    abilitys[ability] += 1
    data_sources[data_source] += 1
print(abilitys)
print(data_sources)

# 保存数据
save_path = "/mnt/bn/hl-joe/working_dir/diff_aware_new/data/eval/test_baseline.parquet"
df.to_parquet(save_path)
print(len(df))

# 确认数据保存正确，读取测试一下
df = pd.read_parquet(save_path)
print("load succeed")
