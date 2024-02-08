# Import libraries
import time
import logging
import smart_open
import itertools
import os
import mysql.connector
import pandas as pd
import random
import math
import betfairlightweight
import glob

from betfairlightweight.filters import streaming_market_filter
from concurrent import futures
from unittest.mock import patch
from flumine import clients, FlumineSimulation, utils
from pythonjsonlogger import jsonlogger
from datetime import datetime
from pathlib import Path


from flumine.flumine import Flumine
from middleware import GetHistoricalCommission, RecordLastXTrades, FindTopSelections, PriceInference, CalculateVolumePriceTrigger, RecordTradeDeltas, CalculatePriceTensor
from strategies.strategy import NeuralAutoTrader
from logging_controls.logging_controls import OrderRecorder

# Logging
logger = logging.getLogger()
custom_format = "%(asctime) %(levelname) %(message)"
log_handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(custom_format)
formatter.converter = time.gmtime
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)

# Params
LOGNAME = datetime.now().strftime('%Y%m%d_%H%M')+'_trade_strat'
STAKE_UNIT = 0.1
RUN_TYPE = 'live' # or 'test'
TEST_DATA_PATH = 'E:/Data/Extracted/Raw/Holdout/2023/' # only need if RUN_TYPE is 'test'
MAX_TTJ = 300
CKPT_PATH = "E:/checkpoints/20240207_0454/price-ladder-epoch=01-val_loss=0.1158.ckpt"

if RUN_TYPE == 'live':
    logger.setLevel(logging.INFO)
else:
    logger.setLevel(logging.CRITICAL)

def run_process(run_type, markets):
    if run_type == 'live':
        trading = betfairlightweight.APIClient("vladgav", "Cosmos1324=", app_key="00Fi1NHkj2pPuCVg")
        trading.login_interactive()
        client = clients.BetfairClient(trading)
        client.min_bet_validation = False
        framework = Flumine(client=client)

        markets_and_names = trading.betting.list_market_catalogue(
            betfairlightweight.filters.market_filter(
                event_type_ids=['7'],
                market_countries=['AU'],
                market_type_codes=['WIN']
            ),max_results=250,lightweight=True)

        tb_markets = [x for x in markets_and_names if 'Trot' not in x['marketName'] and 'Pace' not in x['marketName']] # TODO: convert to worker maybe?

        market_filter = streaming_market_filter(
            market_ids=[x['marketId'] for x in tb_markets]
        )

    elif run_type == 'test':
        client = clients.SimulatedClient()
        framework = FlumineSimulation(client=client)

        market_filter = {
            "markets": markets,
            'market_types': ['WIN'],
            "listener_kwargs": {"inplay": False, "seconds_to_start": MAX_TTJ, "cumulative_runner_tv": True},
        }

        client.min_bet_validation = False
        tb_markets = None

    with patch('builtins.open', smart_open.open):
        framework.add_market_middleware(
            RecordTradeDeltas()
        )
        framework.add_market_middleware(
            RecordLastXTrades()
        )
        framework.add_market_middleware(
            FindTopSelections()
        )
        framework.add_market_middleware(
            CalculateVolumePriceTrigger()
        )
        framework.add_market_middleware(
            CalculatePriceTensor()
        )
        framework.add_market_middleware(
            PriceInference(ckpt_path=CKPT_PATH, tb_markets=tb_markets)
        )
        framework.add_strategy(
            NeuralAutoTrader(
                market_filter=market_filter,
                max_trade_count=3,
                stake_unit=STAKE_UNIT,
                max_back_price=15,
                max_selection_exposure=100,
                max_order_exposure=100,
                max_seconds_to_start=MAX_TTJ,
                run_type=run_type,
                conflate_ms=50
            )
        )

        framework.add_logging_control(
            OrderRecorder(
                logname=LOGNAME+'_'+RUN_TYPE
            )
        )
        framework.run()

# Multi processing
if __name__ == "__main__":
    # Run the API in a terminal

    if RUN_TYPE == 'live':
        run_process(RUN_TYPE, markets=None)
    elif RUN_TYPE == 'test':
        data_files = []

        for dirpath, dirnames, filenames in itertools.chain(os.walk(TEST_DATA_PATH)):
            for filename in filenames:
                data_files.append(os.path.join(dirpath, filename))

        data_files = sorted(data_files, key=os.path.basename)

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

        data_files = [x for x in data_files if os.path.basename(x).rstrip(".bz2") in bsp_list]

        # random.seed(85083)
        # data_files = random.sample(data_files, 10)

        # data_files = [x for x in data_files if os.path.basename(x).startswith("1.215694796")]

        processes = 6 #os.cpu_count() - 1  # Returns the number of CPUs in the system.
        # processes = 1  # Returns the number of CPUs in the system.
        markets_per_process = 8   # 8 is optimal as it prevents data leakage.

        # run_process(RUN_TYPE,data_files[:40])

        chunk = min(
            markets_per_process, math.ceil(len(data_files) / processes)
        )

        _process_jobs = []
        with futures.ProcessPoolExecutor(max_workers=processes) as p:
            # Number of chunks to split the process into depends on the number of markets we want to process and number of CPUs we have.

            # Split all the markets we want to process into chunks to run on separate CPUs and then run them on the separate CPUs
            for m in (utils.chunks(data_files, chunk)):
                _process_jobs.append(
                    p.submit(
                        run_process,
                        run_type=RUN_TYPE,
                        markets=m,
                    )
                )
            for job in futures.as_completed(_process_jobs):
                job.result()  # wait for result