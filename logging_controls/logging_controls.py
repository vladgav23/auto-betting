import os
import logging
import csv

import keyring
import telebot
from flumine.controls.loggingcontrols import LoggingControl
from flumine.order.ordertype import OrderTypes

logger = logging.getLogger(__name__)
CHAT_ID = keyring.get_password("telegram", "chat_id")
bot = telebot.TeleBot(keyring.get_password("telegram", "token"))

class OrderRecorder(LoggingControl):
    NAME = "ORDER_RECORDER"

    def __init__(self, logname, *args, **kwargs):
        self.path = "logs/" + logname + ".csv"
        self.field_names = [
            "bet_id",
            "strategy_name",
            "market_id",
            "selection_id",
            "trade_id",
            "date_time_placed",
            "price",
            "price_matched",
            "size",
            "size_matched",
            "side",
            "elapsed_seconds_executable",
            "order_status",
            "profit",
            "market_note",
            "trade_notes",
            "order_notes"
        ]
        super().__init__(*args, **kwargs)
        self._setup()

    def _setup(self):
        if os.path.exists(self.path):
            logging.info("Results file exists")
        else:
            with open(self.path, "w", newline='') as m:
                csv_writer = csv.DictWriter(m, delimiter=",", fieldnames=self.field_names)
                csv_writer.writeheader()

    def _process_cleared_orders_meta(self, event):
        orders = event.event
        with open(self.path, "a", newline='') as m:
            for order in orders:
                if order.order_type.ORDER_TYPE == OrderTypes.LIMIT:
                    size = order.order_type.size
                else:
                    size = order.order_type.liability
                if order.order_type.ORDER_TYPE == OrderTypes.MARKET_ON_CLOSE:
                    price = None
                else:
                    price = order.order_type.price

                order_data = {
                    "bet_id": order.bet_id,
                    "strategy_name": order.trade.strategy,
                    "market_id": order.market_id,
                    "selection_id": order.selection_id,
                    "trade_id": order.trade.id,
                    "date_time_placed": order.responses.date_time_placed,
                    "price": price,
                    "price_matched": order.average_price_matched,
                    "size": size,
                    "size_matched": order.size_matched,
                    "side": order.side,
                    "elapsed_seconds_executable": order.elapsed_seconds_executable,
                    "profit": order.profit,
                    "order_status": order.status.value,
                    "market_note": order.trade.market_notes,
                    "trade_notes": order.trade.notes_str,
                    "order_notes": order.notes_str
                }
                csv_writer = csv.DictWriter(m, delimiter=",", fieldnames=self.field_names)
                csv_writer.writerow(order_data)

                bot.send_message(CHAT_ID, "Race resulted: "
                                          "\nRace: " + order.notes['race'] +
                                          "\nRunner: " + order.notes['runner_name'] +
                                          "\nSide: " + order.side +
                                 "\nProfit: " + str(order.profit))

            logger.info("Orders updated", extra={"order_count": len(orders)})

    def _process_cleared_markets(self, event):
        cleared_markets = event.event
        for cleared_market in cleared_markets.orders:
            logger.info(
                "Cleared market",
                extra={
                    "market_id": cleared_market.market_id,
                    "bet_count": cleared_market.bet_count,
                    "profit": cleared_market.profit,
                    "commission": cleared_market.commission,
                },
            )