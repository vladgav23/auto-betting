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
        # if market.event_name # contains trot or pace - then return False

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

            back_movers_exist = len(market.context['back_movers']) > 0
            lay_movers_exist = len(market.context['lay_movers']) > 0
            runner_is_firmer = runner.selection_id in market.context['back_movers']
            runner_is_drifter = runner.selection_id in market.context['lay_movers']

            if back_movers_exist:
                mover_price = market.context['back_mover_price'][0]
            elif lay_movers_exist:
                mover_price = market.context['lay_mover_price'][0]

            runner_context = self.get_runner_context(market.market_id, runner.selection_id, runner.handicap)

            if runner_context.live_trade_count == 0:
                if (1.03 > (predicted_wap / runner.last_price_traded) > 1.015) and predicted_wap > best_lay_price and not runner_is_firmer:
                    # create trade
                    trade = Trade(market_book.market_id,runner.selection_id,runner.handicap,self)
                    # create order
                    entry_order = trade.create_order(side='LAY',order_type=LimitOrder(best_lay_price,self.stake_unit),
                                                notes={
                                                    'predicted_max': pred_max_price,
                                                    'pred_price': predicted_wap,
                                                    'market_seconds_to_start': market.seconds_to_start,
                                                    'commission': market.context['commission'],
                                                    'runner_firmer': runner_is_firmer,
                                                    'runner_drifter': runner_is_drifter
                                                })

                    market.place_order(entry_order)

                    # if back_movers_exist:
                    #     back_trade = Trade(market_book.market_id, market.context['back_movers'][0],0,self)
                    #     best_back_mover = [flumine.utils.get_price(x.ex.available_to_back,0) for x in market_book.runners if x.selection_id == market.context['back_movers'][0]][0]
                    #
                    #     back_order = back_trade.create_order(side='BACK', order_type=LimitOrder(best_back_mover, self.stake_unit),
                    #                                     notes={
                    #                                         'pred_price': predicted_wap,
                    #                                         'market_seconds_to_start': market.seconds_to_start,
                    #                                         'commission': market.context['commission']
                    #                                     }
                    #                                     )
                    #
                    #     market.place_order(back_order)
                # elif predicted_wap < best_back_price:
                #     trade = Trade(market_book.market_id, runner.selection_id, runner.handicap, self)
                #     # create order
                #     entry_order = trade.create_order(side='BACK', order_type=LimitOrder(best_back_price, self.stake_unit),
                #                                notes={
                #                                    'type': 'entry',
                #                                    'exit_price_required': pred_min_price,
                #                                    'pred_price': predicted_wap,
                #                                    'projected': 'decrease',
                #                                    'market_seconds_to_start': market.seconds_to_start,
                #                                    'commission': market.context['commission']
                #                                })

                    market.place_order(entry_order)

    # def process_orders(self, market, orders):
    #     if not market.context.get('commission'):
    #         return
    #
    #     for order in orders:
    #         if order.notes['type'] == 'entry':
    #             if not order.complete:
    #                 if order.elapsed_seconds_created >= 10:
    #                     market.cancel_order(order)
    #
    #             if order.complete and order.size_matched > 0:
    #                 # Need to check if there are exit orders in place already
    #                 existing_exit_orders = [x for x in order.trade.orders if x.notes['type'] == 'exit' and x.size_lapsed == 0.0 and x.status != OrderStatus.VIOLATION]
    #
    #                 if not existing_exit_orders:
    #                     for tick in range(1, 3):
    #                         if order.notes['projected'] == 'increase':
    #                             exit_side = 'BACK'
    #                             exit_price = flumine.utils.price_ticks_away(flumine.utils.get_nearest_price(order.average_price_matched), tick)
    #
    #                         elif order.notes['projected'] == 'decrease':
    #                             exit_side = 'LAY'
    #                             exit_price = flumine.utils.price_ticks_away(flumine.utils.get_nearest_price(order.average_price_matched), -tick)
    #
    #                         exit_order = order.trade.create_order(
    #                             side=exit_side, order_type=LimitOrder(exit_price,round(order.size_matched/len(range(1,3)),2)),
    #                             notes={
    #                                 'type': 'exit',
    #                                 'projected': order.notes['projected'],
    #                                 'market_seconds_to_start': market.seconds_to_start,
    #                                 'commission': market.context['commission'],
    #                                 'seconds_to_execute_entry': order.elapsed_seconds_created
    #                             }
    #                         )
    #
    #                         market.place_order(exit_order)
    #
    #         elif order.notes['type'] == 'exit' and order.status == OrderStatus.EXECUTABLE:
    #             runner = next(r for r in market.market_book.runners if r['selection_id'] == order.selection_id)
    #             entry_order = next(x for x in order.trade.orders if x.notes['type'] == 'entry')
    #             # If exit order not matched after 15 seconds, take best price as a stop loss
    #             if not order.complete and order.elapsed_seconds_created + order.notes['seconds_to_execute_entry'] >= 65:
    #                 if order.notes['projected'] == 'increase' and (order.notes.get('needs_replacing') == 1 or flumine.utils.price_ticks_away(runner.last_price_traded, 1) < entry_order.average_price_matched):
    #                     order.notes['needs_replacing'] = 1
    #                     market.replace_order(order, get_price(runner.ex.available_to_back,0))
    #                 elif order.notes['projected'] == 'decrease' and (order.notes.get('needs_replacing')  == 1 or flumine.utils.price_ticks_away(runner.last_price_traded, -1) > entry_order.average_price_matched):
    #                     order.notes['needs_replacing'] = 1
    #                     market.replace_order(order, get_price(runner.ex.available_to_lay, 0))
    #
    #                 # if order.notes['seconds_to_execute_entry'] + order.elapsed_seconds_created >= 60:
    #                 #     order.notes['replace_type'] = "too long"
    #                 #     if order.notes['projected'] == 'increase':
    #                 #         order.replace(new_price=get_price(runner.ex.available_to_back,0))
    #                 #     elif order.notes['projected'] == 'decrease':
    #                 #         order.replace(new_price=get_price(runner.ex.available_to_lay, 0))


