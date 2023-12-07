import json 
from tqdm import tqdm 
import random 

TASK="valloss"

json_path = "./data/nb201_finetune.json"
out_path = f"./data/nb201_finetune_{TASK}.json"

with open(json_path, 'r') as f:
    data = json.load(f)

out_dict = {}
out_dict['type'] = data['type']
out_dict['instances'] = []

for _dict in tqdm(data['instances']):
    if 'valid' in _dict['output'] and 'accuracy' not in _dict['output']:
        out_dict['instances'].append(_dict)

# save 
with open(out_path, 'w') as f:
    json.dump(out_dict, f, indent=4)

# sample 10% of the data 
out_10p_path = f"./data/nb201_finetune_{TASK}_10p.json"
out_dict_10p = {}
out_dict_10p['type'] = data['type']
out_dict_10p['instances'] = []

for _dict in tqdm(data['instances']):
    if 'valid' in _dict['output']:
        if random.random() < 0.1:
            out_dict_10p['instances'].append(_dict)

# save
with open(out_10p_path, 'w') as f:
    json.dump(out_dict_10p, f, indent=4)

