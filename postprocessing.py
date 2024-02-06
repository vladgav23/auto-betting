import os
import torch
import json
import torch.nn.functional as F

dir_to_scan = "E:/Data/Extracted/Processed/TrainNew/"
stats_file = "E:/Data/Extracted/Processed/TrainNew.json"

dir_to_save = os.path.dirname(dir_to_scan) + "_postprocessed/"

def transform_tensor_dict(tensor_dict):
    back = tensor_dict['back']
    lay = tensor_dict['lay']
    traded = tensor_dict['traded']
    lpts = tensor_dict['lpt']
    last_trades = tensor_dict['last_trades']

    if 'target' in tensor_dict:
        target_tensor = tensor_dict['target']

        # Transform prices into ratio to LPT
        target_tensor[:, range(0, 3)] = target_tensor[:, range(0, 3)] / lpts.unsqueeze(1)

        # Log transform total volume
        target_tensor[:, 3] = torch.log(target_tensor[:, 3])

        # Concatenate target
        tensor_dict['target'] = target_tensor.view(-1)

    # Standardise prices by div to LPT
    back[:, :, 0] = torch.where(back[:, :, 0] != 0, back[:, :, 0] / lpts.unsqueeze(1), 0)
    lay[:, :, 0] = torch.where(lay[:, :, 0] != 0, lay[:, :, 0] / lpts.unsqueeze(1), 0)
    traded[:, :, 0] = torch.where(traded[:, :, 0] != 0, traded[:, :, 0] / lpts.unsqueeze(1), 0)

    # Standardise sizes by log transforming
    back[:, :, 1] = torch.where(back[:, :, 1] != 0, (torch.log(back[:, :, 1])), 0)
    lay[:, :, 1] = torch.where(lay[:, :, 1] != 0, (torch.log(lay[:, :, 1])), 0)
    traded[:, :, 1] = torch.where(traded[:, :, 1] != 0, (torch.log(traded[:, :, 1])), 0)

    # Transform last trades
    last_trades[:, :, 0] = last_trades[:, :, 0] / lpts.unsqueeze(1)
    last_trades[:, :, 1] = torch.where(last_trades[:, :, 1] != 0, torch.log(last_trades[:, :, 1]), 0)
    last_trades[:, :, 2] = last_trades[:, :, 2] / torch.max(last_trades[:, :, 2])

def process_dict(dict, track_name, race_type, max_traded_length, min_sts, max_sts, back_lay_length, last_trades_len, track_to_int, rt_to_int):
    # Normalize sts
    input_sts = torch.tensor((dict['seconds_to_start'] - min_sts) / (max_sts - min_sts))
    price_tensor_list = dict['price_tensor_list']

    # Pad and stack back ladders
    raw_back_ladders = [sel['back_ladder'] for sel in price_tensor_list]

    padded_back_ladders = []
    for ld in raw_back_ladders:
        if ld.nelement() == 0:
            padded_seq = torch.zeros([back_lay_length, 2])
        else:
            seq_length = ld.shape[0]
            padding_length = back_lay_length - seq_length

            # Check if padding is needed
            if padding_length > 0:
                # Padding size (left, right, top, bottom)
                pad_size = (0, 0, 0, padding_length)  # Assuming you want to pad at the bottom
                padded_seq = F.pad(ld, pad_size, "constant", 0)  # Padding with zeros
            else:
                padded_seq = ld

        padded_back_ladders.append(padded_seq)

    input_back_ladders = torch.stack(padded_back_ladders)

    # Pad and stack lay ladders
    raw_lay_ladders = [sel['lay_ladder'] for sel in price_tensor_list]

    padded_lay_ladders = []
    for ld in raw_lay_ladders:
        if ld.nelement() == 0:
            padded_seq = torch.zeros([back_lay_length, 2])
        else:
            seq_length = ld.shape[0]
            padding_length = back_lay_length - seq_length

            # Check if padding is needed
            if padding_length > 0:
                # Padding size (left, right, top, bottom)
                pad_size = (0, 0, 0, padding_length)  # Assuming you want to pad at the bottom
                padded_seq = F.pad(ld, pad_size, "constant", 0)  # Padding with zeros
            else:
                padded_seq = ld

        padded_lay_ladders.append(padded_seq)

    input_lay_ladders = torch.stack(padded_lay_ladders)

    # Pad and stack traded ladders
    raw_traded_ladders = [sel['traded_ladder'] for sel in price_tensor_list]

    traded_ladders = []
    for ld in raw_traded_ladders:
        if ld.nelement() == 0:
            padded_seq = torch.zeros([max_traded_length, 2])
        else:
            seq_length = ld.shape[0]
            padding_length = max_traded_length - seq_length

            # Check if padding is needed
            if padding_length > 0:
                pad_size = (0, 0, 0, padding_length)  # Assuming you want to pad at the bottom
                padded_seq = F.pad(ld, pad_size, "constant", 0)  # Padding with zeros
            else:
                padded_seq = ld

        traded_ladders.append(padded_seq)

    input_traded_ladders = torch.stack(traded_ladders)

    # Pad and stack trade sequences
    raw_last_trade_seq = [sel['last_trades'] for sel in price_tensor_list]

    last_trade_sq = []
    for lt_seq in raw_last_trade_seq:
        if lt_seq.nelement() == 0:
            # Handle empty tensor
            padded_seq = torch.zeros((last_trades_len, 3))
        else:
            seq_length = lt_seq.shape[0]
            padding_length = last_trades_len - seq_length

            # Check if padding is needed
            if padding_length > 0:
                # Padding size (left, right, top, bottom)
                pad_size = (0, 0, 0, padding_length)  # Assuming you want to pad at the bottom
                padded_seq = F.pad(lt_seq, pad_size, "constant", 0)  # Padding with zeros
            else:
                padded_seq = lt_seq

        last_trade_sq.append(padded_seq[-100:])

    input_last_trade_seq = torch.stack(last_trade_sq)

    # Gather metadata
    try:
        tensor_dict = {
            'market_id': dict['market_id'],
            'track': track_to_int[track_name],
            'race_type': rt_to_int[race_type],
            'selection_ids': dict['selection_ids'],
            'lpt': torch.tensor([p['lpt'] if p['lpt'] is not None else 0 for p in dict['price_tensor_list']]),
            'mover_flag': torch.tensor([p['mover_flag'] for p in dict['price_tensor_list']]),
            'raw_sts': dict['seconds_to_start'],
            'seconds_to_start': input_sts,
            'back': input_back_ladders,
            'lay': input_lay_ladders,
            'traded': input_traded_ladders,
            'last_trades': input_last_trade_seq
        }
    except Exception as e:
        print("market_id " + dict['market_id'])
        raise (e)

    if 'target' in dict:
        tensor_dict['target'] = torch.stack(dict['target'])

    transform_tensor_dict(tensor_dict)

    return tensor_dict

def process_market(market, min_sts, max_sts, max_traded_length, track_to_int, rt_to_int, track, race_type):
    # Load file
    path = os.path.join(dir_to_scan, market)
    pt = torch.load(path)

    # Remove bad markets
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
                len(lpts) < 6 or
                any(lt.nelement() == 0 for lt in last_trades)):
            new_dict_list.append(i)

    pt_new = [x for i, x in enumerate(pt) if i in new_dict_list]

    if len(new_dict_list) == 0:
        print("Removing market id " + market)
        os.unlink(dir_to_scan + market)
        return None

    pt_new = [process_dict(x,
                          track_name=track,
                          race_type=race_type,
                          max_traded_length=max_traded_length,
                          min_sts=min_sts,
                          max_sts=max_sts,
                          back_lay_length=10,
                          last_trades_len=100,
                          track_to_int=track_to_int,
                          rt_to_int=rt_to_int) for x in pt_new]

    # Collect length
    stats = {
        'market_id': pt[0]['market_id'],
        'len': len(pt_new)
    }
    return pt_new, stats

def write_result_to_file(result):
    if result is not None:
        pt_new, stats = result

        # Save new tensor
        torch.save(pt_new, dir_to_save + stats['market_id'] + ".pt")

        # Append new len to list
        with open(os.path.dirname(dir_to_scan) + "_newstats.json", 'a') as file:
            file.write(str(stats) + '\n')

def main():
    market_data = []
    with open(stats_file, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            try:
                obj = json.loads(line)
                race_name_split = obj['race_name'].split()
                obj['race_type'] = race_name_split[2].lower() if len(race_name_split) > 2 else "unknown"
                obj['track'] = obj['track'].lower()
                market_data.append(obj)
            except json.JSONDecodeError:
                continue

    max_sts = max([x['max_seconds_to_start'] for x in market_data])
    min_sts = min([x['min_seconds_to_start'] for x in market_data])
    max_traded_length = int(max([x['max_traded_length'] for x in market_data]) * 1.1)

    track_list = set([x['track'] for x in market_data])
    race_type_list = set([x['race_type'] for x in market_data])

    track_mapping_path = os.path.dirname(dir_to_scan) + '_track_to_int.json'
    track_to_int = {track: idx for idx, track in enumerate(track_list)}
    with open(track_mapping_path, 'w') as file:
        json.dump(track_to_int, file)

    rt_mapping_path = os.path.dirname(dir_to_scan) + '_rt_to_int.json'

    rt_to_int = {rt: idx for idx, rt in enumerate(race_type_list)}
    with open(rt_mapping_path, 'w') as file:
        json.dump(rt_to_int, file)

    files_in_dir_to_scan = set(os.path.basename(x).rstrip(".pt") for x in os.listdir(dir_to_scan))
    files_already_scanned = set(os.path.basename(x).rstrip(".pt") for x in os.listdir(dir_to_save))

    market_data = [x for x in market_data if x['market_id'] in files_in_dir_to_scan and x['market_id'] not in files_already_scanned]

    for market in market_data:
        market['path'] = dir_to_scan + market['market_id'] + ".pt"

    for market in market_data:
        result = process_market(market['path'], min_sts, max_sts, max_traded_length, track_to_int, rt_to_int, market['track'],
        market['race_type'])
        write_result_to_file(result)

if __name__ == "__main__":
    main()
