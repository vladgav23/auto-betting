import os
import torch

dir_to_scan = "E:/Data/Extracted/Processed/Train/"

for market in os.listdir(dir_to_scan):
    pt = torch.load(dir_to_scan + market)
    pre_len = len(pt)

    new_dict_list = []
    for i, dic in enumerate(pt):
        if not dic.get('target'):
            continue

        # Check if any traded ladders are empty
        lpts = [d['lpt'] for d in dic['price_tensor_list']]
        last_trades = [d['last_trades'] for d in dic['price_tensor_list']]

        if not (any(p is None for p in lpts) or
                any(len(d) < 4 for d in dic['target']) or
                len(lpts) < 4 or
                any(lt.nelement() == 0 for lt in last_trades)):
            new_dict_list.append(i)

    pt = [x for i, x in enumerate(pt) if i in new_dict_list]

    if len(new_dict_list) == 0:
        print("Removing market id " + market)
        os.unlink(dir_to_scan + market)
    else:
        if len(new_dict_list) < pre_len:
            print("Resaving market id " + market)
            torch.save(pt, dir_to_scan + market)

