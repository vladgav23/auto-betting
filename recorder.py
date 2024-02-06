import os
import logging
import time
import torch
import math
import smart_open
import json
import glob
import pandas as pd

from middleware import CalculateVolumePriceTrigger, FindTopSelections, RecordLastXTrades, RecordTargetLadders, RecordTradeDeltas
from pythonjsonlogger import jsonlogger
from unittest.mock import patch
from flumine import utils, clients, FlumineSimulation
from flumine import BaseStrategy
from concurrent import futures

logger = logging.getLogger()

custom_format = "%(asctime) %(levelname) %(message)"
log_handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(custom_format)
formatter.converter = time.gmtime
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)
logger.setLevel(logging.CRITICAL)

class PriceRecorder(BaseStrategy):

    def process_new_market(self, market, market_book):
        market.context['price_list'] = []
        market.context['max_traded_length'] = 0

    def check_market_book(self, market, market_book):
        lcase_name = market_book.market_definition.name.lower()
        if market.market_type == "WIN" and "trot" not in lcase_name and "pace" not in lcase_name:
            return True

    def process_market_book(self, market, market_book):
        if market_book.inplay == False and market_book.status == 'OPEN':
            if not market.context.get('vp_trigger_seconds'):
                return

            latest_trigger_second = min(market.context['vp_trigger_seconds'])

            # If there has been a trade and a trigger
            if 180 >= market.seconds_to_start >= 30 and latest_trigger_second == market.seconds_to_start and [x for x in market_book.streaming_update['rc'] if 'tv' in x]:
                market_update_tensor_list = []
                selection_ids = []
                for runner in market_book.runners:
                    if runner.selection_id in market.context['top_selections']:
                        last_trades_tensor_list = [x['last_trades'] for x in market.context['last_x_trades'] if
                                     x['id'] == runner.selection_id]

                        if last_trades_tensor_list:
                            last_trades_tensor = torch.tensor(last_trades_tensor_list[0])
                            last_trades_tensor[:, 2] = last_trades_tensor[:, 2] - market.seconds_to_start
                        else:
                            last_trades_tensor = torch.tensor([])

                        traded_ladder_tensor = [[d['price'], d['size']] for d in runner.ex.traded_volume]

                        if len(traded_ladder_tensor) > market.context['max_traded_length']:
                            market.context['max_traded_length'] = len(traded_ladder_tensor)

                        market_update_tensor_list.append(
                            {
                                "mover_flag": runner.selection_id in market.context['vp_trigger_selections'],
                                "lpt": runner.last_price_traded,
                                "back_ladder": torch.tensor(
                                    [[d['price'], d['size']] for d in runner.ex.available_to_back][:10]),
                                "lay_ladder": torch.tensor(
                                    [[d['price'], d['size']] for d in runner.ex.available_to_lay][:10]),
                                "traded_ladder": torch.tensor(traded_ladder_tensor),
                                "last_trades": last_trades_tensor
                            }
                        )

                        selection_ids.append(runner.selection_id)

                if market.seconds_to_start < 0:
                    mss = -1
                else:
                    mss = market.seconds_to_start

                market.context['price_list'].append(
                    {
                        "market_id": market.market_id,
                        "selection_ids": selection_ids,
                        "seconds_to_start": mss,
                        "price_tensor_list": market_update_tensor_list
                    }
                )

    def find_price_differences(self, list1, list2):
        # Convert the second list to a dictionary for faster lookup
        price_dict = {price: size for price, size in list2}

        # Calculate the differences
        differences = []
        for price, size in list1:
            corresponding_size = price_dict.get(price, 0)
            difference = size - corresponding_size

            if difference > 0:
                differences.append([price, round(difference,2)])

        return differences

    def process_closed_market(self, market, market_book):
        if market.context['price_list'] and market.context['target_ladders'] and len(market.context['top_selections']) == 6:
            # Trim target ladders to +/- 5 ticks from trigger LTP
            for ladder in market.context['target_ladders']:

                ladder_diffs = []
                for i, sel in enumerate(ladder['target']):
                    ladder_diffs.append(self.find_price_differences(sel, ladder['initial_ladder'][i]))

                trimmed_target_ladders = []
                for sel_idx in range(0, len(ladder_diffs)):
                    sorted_target = sorted(ladder_diffs[sel_idx], key=lambda x: x[0], reverse=True)
                    tot_vol = round(sum([x[1] for x in ladder_diffs[sel_idx]]), 2)
                    vwap_target = round(sum([(x[0] * x[1]) for x in ladder_diffs[sel_idx]]) / tot_vol, 2)

                    filtered_target = [x for x in sorted_target if x[1]/tot_vol >= 0.02]
                    min_max_prices = [filtered_target[0][0]] + [filtered_target[-1][0]]

                    target = torch.tensor(min_max_prices + [vwap_target] + [tot_vol])

                    trimmed_target_ladders.append(target)

                price_list_to_append_target = [x for x in market.context['price_list'] if x['seconds_to_start'] == ladder['second']]
                price_list_to_append_target[0]['target'] = trimmed_target_ladders

            sts_list = [x['seconds_to_start'] for x in market.context['price_list']]

            removed_non_target = [x for x in market.context['price_list'] if 'target' in x]

            torch.save(removed_non_target, "E:/Data/Extracted/Processed/HoldoutNew/" + market.market_id + ".pt")

            stats_dict = {
                'market_id': market.market_id,
                'track': market.market_book.market_definition.venue,
                'race_name': market.market_book.market_definition.name,
                'total_length': len(removed_non_target),
                'max_traded_length': market.context['max_traded_length'],
                'max_seconds_to_start': max(sts_list),
                'min_seconds_to_start': min(sts_list)
            }

            stats_str = json.dumps(stats_dict)

            with open("E:/Data/Extracted/Processed/HoldoutNew.json", 'a') as file:
                file.write(stats_str + '\n')

def run_process(markets):
    # Set Flumine to simulation mode
    client = clients.SimulatedClient()
    framework = FlumineSimulation(client=client)

    # Remove simulated middleware and add my own
    framework._market_middleware = []



    # Set parameters for our strategy
    strategy = PriceRecorder(
        # market_filter selects what portion of the historic data we simulate our strategy on
        # markets selects the list of betfair historic data files
        # market_types specifies the type of markets
        # listener_kwargs specifies the time period we simulate for each market
        market_filter={
            "markets": markets,
            "listener_kwargs": {"seconds_to_start": 300, "inplay": False ,"cumulative_runner_tv": True}
        }
    )

    # Run our strategy on the simulated market
    with patch("builtins.open", smart_open.open):
        framework.add_market_middleware(
            FindTopSelections()
        )
        framework.add_market_middleware(
            RecordTradeDeltas()
        )
        framework.add_market_middleware(
            RecordLastXTrades()
        )
        framework.add_market_middleware(
            CalculateVolumePriceTrigger()
        )
        framework.add_market_middleware(
            RecordTargetLadders()
        )
        framework.add_strategy(strategy)
        framework.run()


# Multi processing
if __name__ == "__main__":
    streaming_files = []

    for root, dirs, files in os.walk("E:/Data/Extracted/Raw/Holdout/"):
        for file in files:
            streaming_files.append(os.path.join(root, file))

    all_markets = [x for x in streaming_files if os.path.basename(x).startswith(('1.'))]

    bsp_files_path = "E:/Data/BSP/Aus_Thoroughbreds*"
    files_to_read = glob.glob(bsp_files_path)
    bsp_list = []
    for file in files_to_read:
        try:
            bsp_year = pd.read_csv(file, usecols=['MARKET_ID'])
            bsp_year = bsp_year.drop_duplicates()
            bsp_year['MARKET_ID'] = '1.' + bsp_year['MARKET_ID'].astype(str)
            market_ids = bsp_year['MARKET_ID'].tolist()
        except Exception as e:
            bsp_year = pd.read_csv(file, usecols=['market_id'])
            bsp_year = bsp_year.drop_duplicates()
            bsp_year['market_id'] = bsp_year['market_id'].astype(str)
            market_ids = bsp_year['market_id'].tolist()

        bsp_list = bsp_list + market_ids

    bsp_list = set(bsp_list)

    all_markets = [x for x in all_markets if os.path.basename(x).rstrip(".bz2") in bsp_list]

    # All the markets we want to simulate
    processes = os.cpu_count() - 1  # Returns the number of CPUs in the system.
    markets_per_process = 8  # 8 is optimal as it prevents data leakage.

    # run_process([x for x in all_markets if '1.131270437' in x])

    _process_jobs = []
    with futures.ProcessPoolExecutor(max_workers=processes) as p:
        # Number of chunks to split the process into depends on the number of markets we want to process and number of CPUs we have.
        chunk = min(
            markets_per_process, math.ceil(len(all_markets) / processes)
        )
        # Split all the markets we want to process into chunks to run on separate CPUs and then run them on the separate CPUs
        for m in (utils.chunks(all_markets, chunk)):
            _process_jobs.append(
                p.submit(
                    run_process,
                    markets=m,
                )
            )
        for job in futures.as_completed(_process_jobs):
            job.result()  # wait for result