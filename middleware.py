import logging
import json
import torch

import flumine.utils
import requests
import pandas as pd
import glob
from flumine.markets.middleware import Middleware
from statistics import mean
from postprocessing import process_dict
from itertools import groupby

logger = logging.getLogger(__name__)

class ScoredPricesMiddleware(Middleware):
    def __init__(self, extra_fields_to_get):
        self._runner_removals = []
        self.first_scratching_update = 600
        self.extra_fields_to_get = extra_fields_to_get

    def __call__(self, market) -> None:
        runner_removals = []

        for runner in market.market_book.runners:
            if runner.status == "ACTIVE":
                continue
            elif runner.status == "REMOVED":
                _removal = runner.selection_id
                if _removal not in self._runner_removals:
                    self._runner_removals.append(_removal)
                    runner_removals.append(_removal)

        for _removal in runner_removals:
            if market.seconds_to_start < self.first_scratching_update:
                self._process_runner_removal(market, _removal)

        if not market.context.get('model_data'):
            self._update_prices(market)

    def _process_runner_removal(self, market, removal_selection_id: int) -> None:
        params = {
            'market_id': market.market_id,
            'extra_fields_to_get': self.extra_fields_to_get,
            'extra_features': json.dumps({}),
            'scratching': removal_selection_id
        }

        # Send the GET request with the parameters
        response = requests.get("http://host.docker.internal:8000/process_scratching", params=params)
        market.context["model_data"] = response.json()

    def _update_prices(self, market):
        params = {
            'market_id': market.market_id,
            'extra_fields_to_get': self.extra_fields_to_get,
            'extra_features': json.dumps({})
        }

        # Send the GET request with the parameters
        response = requests.get("http://host.docker.internal:8000/get_prices", params=params)
        market.context["model_data"] = response.json()

class GetPricesFromScoredHoldout(Middleware):
    def __init__(self, holdout_scored_path):
        self.predictions = pd.read_csv(holdout_scored_path,usecols=["market_id","selection_id","seconds_to_start","predicted_wap","predicted_min_price","predicted_max_price","mover"])
        self.predictions['market_id'] = self.predictions['market_id'].astype(str)
        self.predictions['selection_id'] = self.predictions['selection_id'].astype(int,errors="ignore")
        self.predictions = self.predictions.to_dict('records')

    def __call__(self, market) -> None:
        market_id_to_get = market.market_id
        market.context['scored_data'] = [x for x in self.predictions if x['market_id'] == market_id_to_get and round(x['seconds_to_start'],2) == round(market.seconds_to_start,2)]

class GetHistoricalCommission(Middleware):
    def __init__(self):
        bsp_files_path = "E:/Data/BSP/Aus_Thoroughbreds_2023*"
        files_to_read = glob.glob(bsp_files_path)
        bsp_list = []
        for file in files_to_read:
            bsp_list.append(
                pd.read_csv(file,usecols=['MARKET_ID','STATE_CODE'])
            )

        self.bsp_df = pd.concat(bsp_list).drop_duplicates()
        self.bsp_df['MARKET_ID'] = '1.' + self.bsp_df['MARKET_ID'].astype(str)

    def __call__(self, market):
        market_id_to_get = market.market_id

        if not market.context.get('commission'):
            market_state = self.bsp_df.query("MARKET_ID == @market_id_to_get")['STATE_CODE']

            if market_state.empty:
                market.context['commission'] = 0.1
            elif market_state.item() in ('NSW','ACT'):
                market.context['commission'] = 0.1
            else:
                market.context['commission'] = 0.07

class FindTopSelections(Middleware):
    def __call__(self, market) -> None:
        if market.seconds_to_start <= 600 and not market.context.get('top_selections'):
            market.context['min_ltp'] = [{
                'selection_id': x.selection_id,
                'min_ltp': x.last_price_traded}
                for x in market.market_book.runners]

            for runner in market.market_book.runners:
                ltp = runner.last_price_traded
                runner_min_ltp = [x for x in market.context['min_ltp'] if x['selection_id'] == runner.selection_id][0]
                if ltp and runner_min_ltp['min_ltp'] and ltp < runner_min_ltp['min_ltp']:
                    runner_min_ltp['min_ltp'] = ltp

            top_list = sorted(market.context['min_ltp'], key=lambda x: float('inf') if x['min_ltp'] is None else x['min_ltp'])[:6]
            market.context['top_selections'] = [x['selection_id'] for x in top_list]

class CalculateWAPMetrics(Middleware):
    def __call__(self, market) -> None:
        market.context['last_trades_wap'] = []
        market.context['traded_ladder_wap'] = []

        if not market.context.get('last_x_trades'):
            return
        else:
            for runner in market.context['last_x_trades']:
                runner_last_trades_prev_15_seconds = [x for x in runner['last_trades'] if
                                                      market.seconds_to_start + 15 >= x[2] >= market.seconds_to_start]
                runner_total_matched_last_15 = sum(x[1] for x in runner_last_trades_prev_15_seconds)
                runner_wap_last_15 = round(
                    sum([x[0] * x[1] / runner_total_matched_last_15 for x in runner_last_trades_prev_15_seconds]), 2)

                market.context['last_trades_wap'].append(
                    {
                        'id': runner['id'],
                        'wap_last_15': runner_wap_last_15
                    }
                )

        if not market.context.get('prev_traded_ladders'):
            return
        else:
            for runner in market.context['prev_traded_ladders']:
                runner_traded_total = sum([x['size'] for x in runner['trd']])
                runner_total_wap = round(sum([x['price'] * x['size'] / runner_traded_total for x in runner['trd']]),2)

                market.context['traded_ladder_wap'].append(
                    {
                        'id': runner['id'],
                        'wap_total': runner_total_wap
                    }
                )

class CalculateVolumePriceTrigger(Middleware):
    """
    Returns True or False based on whether a sufficient amount of
    volume has suddenly entered the market (i.e. 1% of market in 1 update),
    and has shifted the price of one of the top 4 by at least 2 ticks
    """
    def __call__(self, market) -> None:
        if not market.context.get('vp_trigger_seconds'):
            market.context['vp_trigger_seconds'] = []

        if not market.context.get('top_selections'):
            market.context['vp_trigger_selections'] = []
            return

        if len(market.context['top_selections']) != 6:
            market.context['vp_trigger_selections'] = []
            return

        if not market.context.get('trade_deltas'):
            market.context['vp_trigger_selections'] = []
            return

        market_total_vol = sum([runner['total_matched'] for runner in market.market_book.runners])
        sorted_deltas = sorted(market.context['trade_deltas'], key=lambda x: x['id'])

        # Compact solution using groupby from itertools
        market.context['sum_deltas'] = [{
            'id': key,
            'size': sum(item['delta'][1] for item in items),
            'min_traded': min(item['delta'][0] for item in items),
            'max_traded': max(item['delta'][0] for item in items)
        } for key, group in groupby(sorted_deltas, key=lambda x: x['id']) for items in [list(group)]]

        volume_trigger = set(
            [x['id'] for x in market.context['sum_deltas'] if x['size'] / market_total_vol >= 0.01 and x['size'] > 50])

        if not volume_trigger:
            market.context['vp_trigger_selections'] = []
            return

        if market.seconds_to_start >= 30:
            market.context['vp_trigger_selections'] = list(set(
                [x['id'] for x in market.context['trade_deltas'] if x['id'] in volume_trigger]))
            market.context['vp_trigger_seconds'].append(round(market.seconds_to_start,2))

class RecordLastXTrades(Middleware):
    def __call__(self, market) -> None:
        if not market.context.get('trade_deltas'):
            return

        if market.seconds_to_start < 0:
            mss = -1
        else:
            mss = market.seconds_to_start

        # Initialise last_x_trades if it doesn't exist
        if not market.context.get('last_x_trades'):
            market.context['last_x_trades'] = []

        for delta in market.context['trade_deltas']:
            if delta['delta'][1] < 5:
                continue
            last_trades_for_id = [x['last_trades'] for x in market.context['last_x_trades'] if x['id'] == delta['id']]
            delta_to_append = delta['delta'] + [mss]

            if not last_trades_for_id:
                market.context['last_x_trades'].append(
                    {
                        'id': delta['id'],
                        'last_trades': [delta_to_append]
                    }
                )
                return

            last_trades_for_id[0].append(delta_to_append)

        # Make sure we're only keeping last 100 trades
        for sel in market.context['last_x_trades']:
            sel['last_trades'] = sel['last_trades'][-100:]

class RecordTradeDeltas(Middleware):
    def __call__(self, market) -> None:
        if not market.market_book.streaming_update.get('rc'):
            market.context['trade_deltas'] = []
            return

        traded_vol_update = [d for d in market.market_book.streaming_update['rc'] if
                             'tv' in d and d['id']]

        if not traded_vol_update:
            market.context['trade_deltas'] = []
            return

        current_traded_ladder = [{
            'id': r.selection_id, 'trd': r.ex.traded_volume} for r in market.market_book.runners if
            r.selection_id]

        if not market.context.get('prev_traded_ladders'):
            market.context['prev_traded_ladders'] = current_traded_ladder
            return

        trade_deltas = []
        for trade in traded_vol_update:
            previous_ladder = [x['trd'] for x in market.context['prev_traded_ladders'] if x['id'] == trade['id']][0]

            for ladder in trade['trd']:
                price = ladder[0]
                size = ladder[1]

                previous_size = [x['size'] for x in previous_ladder if x['price'] == price]

                if previous_size:
                    size = round(size - previous_size[0], 2)

                if size > 0:
                    trade_deltas.append(
                        {
                            'id': trade['id'],
                            'delta': [price, size]
                        }
                    )

        market.context['trade_deltas'] = trade_deltas
        market.context['prev_traded_ladders'] = current_traded_ladder

class RecordTargetLadders(Middleware):
    def __call__(self, market) -> None:
        # If no triggers yet, return
        if not market.context.get('vp_trigger_seconds'):
            return

        # If target ladders not initialised yet, do it
        if not market.context.get('target_ladders'):
            market.context['target_ladders'] = []

        for trig_sec in market.context['vp_trigger_seconds']:
            existing_trig_sec_target = [x for x in market.context['target_ladders'] if x['second'] == trig_sec]
            current_traded_dicts = [x.ex.traded_volume for x in market.market_book.runners if
                                               x.selection_id in market.context['top_selections']]

            current_traded_ladders = []
            for sel in current_traded_dicts:
                sel = [[d['price'],d['size']] for d in sel]
                current_traded_ladders.append(sel)

            if not existing_trig_sec_target:
                lpt_for_selections = [x.last_price_traded for x in market.market_book.runners if x.selection_id in market.context['top_selections']]
                market.context['target_ladders'].append(
                    {
                        'second': trig_sec,
                        'initial_ladder': current_traded_ladders,
                        'target': current_traded_ladders,
                        'lpts': lpt_for_selections
                    }
                )
            else:
                if (market.seconds_to_start >= trig_sec - 60) and not market.market_book.inplay and market.status == "OPEN":
                    existing_trig_sec_target[0]['target'] = current_traded_ladders

class CalculatePriceTensor(Middleware):
    def __call__(self,market) -> None:
        if not market.context.get("vp_trigger_selections"):
            market.context['price_list'] = {}
            return

        market_update_tensor_list = []
        selection_ids = []
        for runner in market.market_book.runners:
            if runner.selection_id in market.context['top_selections']:
                last_trades_tensor_list = [x['last_trades'] for x in market.context['last_x_trades'] if
                                           x['id'] == runner.selection_id]

                if last_trades_tensor_list:
                    last_trades_tensor = torch.tensor(last_trades_tensor_list[0])
                    last_trades_tensor[:, 2] = last_trades_tensor[:, 2] - market.seconds_to_start
                else:
                    last_trades_tensor = torch.tensor([])

                traded_ladder_tensor = [[d['price'], d['size']] for d in runner.ex.traded_volume]

                # if len(traded_ladder_tensor) > market.context['max_traded_length']:
                #     market.context['max_traded_length'] = len(traded_ladder_tensor)

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

        market.context['price_list'] = {
                "market_id": market.market_id,
                "selection_ids": selection_ids,
                "seconds_to_start": mss,
                "price_tensor_list": market_update_tensor_list}

class PriceInference(Middleware):
    def __init__(self, ckpt_path,track_to_int_path, rt_to_int_path,max_sts,suffix,tb_markets=None):
        from model.model import PriceLadderModel, PriceLadderDataModule

        ckpt_file = torch.load(ckpt_path, map_location="cpu")
        traded_weight = ckpt_file['state_dict']['proj_traded_ladder.weight']
        self.max_traded_length_train = int(traded_weight.shape[1] / 128)
        with open(track_to_int_path, 'r') as file:
            self.track_to_int = json.load(file)

        with open(rt_to_int_path, 'r') as file:
            self.rt_to_int = json.load(file)

        self.model = PriceLadderModel(max_traded_length=self.max_traded_length_train,
                                 track_to_int=self.track_to_int,
                                 rt_to_int=self.rt_to_int).to("cpu")

        self.model.load_state_dict(ckpt_file['state_dict'])

        del traded_weight, ckpt_file

        self.model.eval()
        self.collate_batch = PriceLadderDataModule.collate_batch

        self.tb_markets = tb_markets
        self.max_sts = max_sts
        self.suffix = suffix

    def __call__(self, market) -> None:
        if not market.context.get('price_list'):
            market.context['scored_data_'+self.suffix] = []
            return

        if self.tb_markets is None:
            market_name = market.market_book.market_definition.name
        else:
            market_name = [x['marketName'] for x in self.tb_markets if x['marketId'] == market.market_id]

        if not market_name:
            return

        race_name_split = market_name[0].split()
        race_type = race_name_split[2] if len(race_name_split) > 2 else "Unknown"

        dict_to_score = self.collate_batch(
            [process_dict(market.context['price_list'],
                          track_name=market.market_book.market_definition.venue.lower(),
                          race_type=race_type.lower(),
                          max_traded_length=self.max_traded_length_train,
                          min_sts=30,
                          max_sts=self.max_sts,
                          back_lay_length=10,
                          last_trades_len=100,
                          track_to_int=self.track_to_int,
                          rt_to_int=self.rt_to_int
                          )],
            has_target=False
        )

        with torch.no_grad():
            prediction = self.model(dict_to_score['pred_tensors']).view(1, 6, 3)

        # Transform prices into ratio to LPT
        prediction = (prediction * dict_to_score['pred_tensors']['lpts'].unsqueeze(2))

        runner_dicts = []
        for i, runner in enumerate(dict_to_score['metadata']['selection_ids'][0]):
            dict_to_append = {
                'selection_id': runner,
                'predicted_max_price': round(prediction[0, i, 0].item(),2),
                'predicted_min_price': round(prediction[0, i, 1].item(),2),
                'predicted_wap': round(prediction[0, i, 2].item(),2)
            }

            runner_dicts.append(dict_to_append)

        market.context['scored_data_'+self.suffix] = runner_dicts