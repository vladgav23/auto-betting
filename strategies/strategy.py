import flumine.utils
import shin
import logging
import telebot
import keyring
import pandas as pd
import torch

from flumine.markets.market import Market
from betfairlightweight.resources import MarketBook
from flumine import BaseStrategy
from flumine.utils import get_price
from flumine.order.trade import Trade
from flumine.order.order import LimitOrder, OrderStatus

logger = logging.getLogger(__name__)

class NeuralAutoTrader(BaseStrategy):
    def __init__(self, stake_unit, max_back_price, max_seconds_to_start, run_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stake_unit = stake_unit
        self.max_seconds_to_start = max_seconds_to_start
        self.run_type = run_type
        self.max_back_price = max_back_price

    def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
        # Ignore closed or in-play markets
        if market_book.status != "CLOSED" and not market_book.inplay:
            return True

    def process_market_book(self, market: Market, market_book: MarketBook) -> None:
        # If we don't have model_data, return
        if not market.context.get('scored_data'):
            return

        # If too far out from jump, return
        # if not self.max_seconds_to_start >= int(market.seconds_to_start) >= 90:
        #     return

        # Process runners
        for runner in market_book.runners:
            if runner.status != "ACTIVE":
                continue

            runner_data = [d for d in market.context['scored_data'] if d['selection_id'] == runner.selection_id]

            if not runner_data:
                continue

            runner_data = runner_data[0]
            best_back_price = get_price(runner.ex.available_to_back,0)
            best_lay_price = get_price(runner.ex.available_to_lay, 0)

            # Skip runner if back_price is above desired threshold
            if best_back_price > self.max_back_price:
                continue

            pred_max_price = runner_data['predicted_max_price']
            pred_min_price = runner_data['predicted_min_price']
            predicted_wap = runner_data['predicted_wap']
            mover_flag = runner.selection_id in market.context['vp_trigger_selections']

            runner_context = self.get_runner_context(market.market_id, runner.selection_id, runner.handicap)

            print("Mover flag: " + str(mover_flag))
            print("WAP ratio: " + str(round(predicted_wap / best_back_price,3)))
            print("Last price traded: " + str(runner.last_price_traded) + ", Best back price: " + str(best_back_price))

            if runner_context.live_trade_count == 0:
                if (predicted_wap / best_back_price) <= 0.975 and mover_flag == True and runner.last_price_traded == best_back_price:
                    # create trade
                    trade = Trade(market_book.market_id, runner.selection_id, runner.handicap, self)
                    # create order
                    entry_order = trade.create_order(side='BACK', order_type=LimitOrder(best_back_price, self.stake_unit),
                                                     notes={
                                                         'predicted_max': pred_max_price,
                                                         'predicted_min': pred_min_price,
                                                         'pred_price': predicted_wap,
                                                         'market_seconds_to_start': market.seconds_to_start,
                                                         # 'commission': market.context['commission'],
                                                         'mover': mover_flag,
                                                         'last_price': runner.last_price_traded
                                                     })

                    market.place_order(entry_order)

    def process_orders(self, market, orders):
        for order in orders:
            if not order.complete:
                if order.elapsed_seconds_created >= 60:
                    market.cancel_order(order)

