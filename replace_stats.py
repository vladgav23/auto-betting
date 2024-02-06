import os
import torch
import json

path = "E:/Data/Extracted/Processed/TrainNew_Postprocessed/"
files_to_scan = os.listdir(path)

stats_list = []
for file in files_to_scan:
    tensor_dict = torch.load(path + file)
    length = len(tensor_dict)
    market = file.rstrip(".pt")

    stats_list.append({
        "market_id": market,
        "len": length
    })

with open('E:/Data/Extracted/Processed/TrainNew_newstats.json', 'w') as fout:
    json.dump(stats_list, fout)