import time
from collections import deque
import traceback
import asyncio
import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
from uuid import uuid4
import numpy as np
import json
import pprint

from passivbot import Passivbot, logging
from njit_funcs import round_
import passivbot_rust as pbr
from pure_funcs import (
    floatify,
    ts_to_date_utc,
    calc_hash,
    determine_pos_side_ccxt,
    flatten,
    shorten_custom_id,
)
from procedures import (
    print_async_exception,
    utc_ms,
    assert_correct_ccxt_version,
    load_broker_code,
)

assert_correct_ccxt_version(ccxt=ccxt_async)


class BinanceBot(Passivbot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.custom_id_max_length = 36

        # ─── For sliding‐window rate limiting ───
        # A deque to hold timestamps (in seconds) of the last Futures copy‐trading REST calls:
        self._futures_call_timestamps = deque()

    def _wait_for_copy_trading_slot(self):
        """
        Ensure we never send more than 20 Futures copy‐trading REST calls in any rolling 10-second window.
        If we have already sent 20 calls in the last 10 s, sleep until we drop below that.
        """
        now = time.time()
        window_start = now - 10.0  # 10 seconds ago

        # Pop out any timestamps older than 10 seconds
        while self._futures_call_timestamps and self._futures_call_timestamps[0] < window_start:
            self._futures_call_timestamps.popleft()

        # If already used up 20 calls in the past 10 seconds, wait until the oldest timestamp is >10 s ago
        if len(self._futures_call_timestamps) >= 20:
            oldest = self._futures_call_timestamps[0]
            sleep_time = (oldest + 10.0) - now
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Now we have < 20 calls in the last 10s, so it’s OK to proceed.

    async def _ccxt_request(self, method, *args, **kwargs):
        """
        Wrapper around any CCXT async method to enforce “20 calls / 10 s”:
          1) Wait if needed (_wait_for_copy_trading_slot).
          2) Call the actual CCXT method (e.g., create_order, fetch_ticker, etc.).
          3) Record the timestamp of this call.
        """
        # 1) Possibly sleep so we don’t exceed 20 calls/10s
        self._wait_for_copy_trading_slot()

        # 2) Invoke the real CCXT method
        result = await method(*args, **kwargs)

        # 3) Record this call’s timestamp
        self._futures_call_timestamps.append(time.time())

        return result

    def create_ccxt_sessions(self):
        # ─── FUTURES COPY TRADING ───
        self.broker_code_fut = load_broker_code("binance_futures")

        for ccx, ccxt_module in [("cca", ccxt_async), ("ccp", ccxt_pro)]:
            # 1) Instantiate the USDT-M FUTURES “binanceusdm” exchange
            fut_cls = getattr(ccxt_module, "binanceusdm")
            setattr(
                self,
                ccx,
                fut_cls({
                    "apiKey":           self.user_info["key"],
                    "secret":           self.user_info["secret"],
                    "password":         self.user_info["passphrase"],
                    "timeout":          60000,     # 60 s timeout
                    "enableRateLimit":  False,     
                    "aiohttp_proxy":    None,
                    "asyncio_loop":     None,
                })
            )

            exchange = getattr(self, ccx)
            # 2) Force “swap” (perpetual‐futures) endpoints
            exchange.options["defaultType"] = "swap"

            # 3) Increase recvWindow to avoid timestamp‐offset errors
            exchange.options["recvWindow"] = 50000
            exchange.options["adjustForTimeDifference"] = True
            exchange.options["retries"] = 3

            # 4) Required headers (User-Agent + APIKEY)
            exchange.headers = {
                "User-Agent":   f"PassivBot/{self.version if hasattr(self, 'version') else '1.0'}",
                "X-MBX-APIKEY": self.user_info["key"],
            }

            # 5) Attach all Futures subtypes with broker code
            if self.broker_code_fut:
                exchange.options.setdefault("broker", {})
                for subtype in ["future", "delivery", "swap", "option"]:
                    exchange.options["broker"][subtype] = "x-" + self.broker_code_fut

    async def print_new_user_suggestion(self):
        between_print_wait_ms = 1000 * 60 * 60 * 4
        if hasattr(self, "previous_user_suggestion_print_ts"):
            if utc_ms() - self.previous_user_suggestion_print_ts < between_print_wait_ms:
                return
        self.previous_user_suggestion_print_ts = utc_ms()

        res = None
        try:
            # ───── wrap the CCXT call ─────
            res = await self._ccxt_request(
                self.cca.fapiprivate_get_apireferral_ifnewuser,
                params={"brokerid": self.broker_code_fut},
            )
        except Exception as e:
            logging.error(f"failed to fetch fapiprivate_get_apireferral_ifnewuser {e}")
            print_async_exception(res)
            return

        if res["ifNewUser"] and res["rebateWorking"]:
            return

        lines = [
            "To support continued Passivbot development, please use a Binance account which",
            "1) was created after 2024-09-21 and",
            "2) either:",
            "  a) was created without a referral link, or",
            '  b) was created with referral ID: "TII4B07C".',
            " ",
            "Passivbot receives commissions from trades only for accounts meeting these criteria.",
            " ",
            json.dumps(res),
        ]
        front_pad = " " * 8 + "##"
        back_pad = "##"
        max_len = max([len(line) for line in lines])
        print("\n\n")
        print(front_pad + "#" * (max_len + 2) + back_pad)
        for line in lines:
            print(front_pad + " " + line + " " * (max_len - len(line) + 1) + back_pad)
        print(front_pad + "#" * (max_len + 2) + back_pad)
        print("\n\n")

    async def execute_to_exchange(self):
        res = await super().execute_to_exchange()
        await self.print_new_user_suggestion()
        return res

    def set_market_specific_settings(self):
        super().set_market_specific_settings()
        for symbol in self.markets_dict:
            elm = self.markets_dict[symbol]
            self.min_costs[symbol] = (
                0.1 if elm["limits"]["cost"]["min"] is None else elm["limits"]["cost"]["min"]
            )
            self.min_qtys[symbol] = elm["limits"]["amount"]["min"]
            self.price_steps[symbol] = elm["precision"]["price"]
            self.qty_steps[symbol] = elm["precision"]["amount"]
            self.c_mults[symbol] = elm["contractSize"]

    async def watch_balance(self):
        while True:
            try:
                if self.stop_websocket:
                    break
                # ───── wrap the CCXT watch_balance call ─────
                res = await self._ccxt_request(self.ccp.watch_balance)
                self.handle_balance_update(res)
            except Exception as e:
                logging.error(f"exception watch_balance {e}")
                traceback.print_exc()
                await asyncio.sleep(1)

    async def watch_orders(self):
        while True:
            try:
                if self.stop_websocket:
                    break
                # ───── wrap the CCXT watch_orders call ─────
                res = await self._ccxt_request(self.ccp.watch_orders)
                for i in range(len(res)):
                    res[i]["position_side"] = res[i]["info"]["ps"].lower()
                    res[i]["qty"] = res[i]["amount"]
                self.handle_order_update(res)
            except Exception as e:
                if "Abnormal closure of client" not in str(e):
                    logging.error(f"exception watch_orders {e}")
                    traceback.print_exc()
                await asyncio.sleep(1)

    async def fetch_open_orders(self, symbol: str = None, all=False) -> [dict]:
        fetched = None
        open_orders = {}
        try:
            if all:
                self.cca.options["warnOnFetchOpenOrdersWithoutSymbol"] = False
                logging.info(f"fetching all open orders for binance")
                # ───── wrap the CCXT fetch_open_orders call ─────
                fetched = await self._ccxt_request(self.cca.fetch_open_orders)
                self.cca.options["warnOnFetchOpenOrdersWithoutSymbol"] = True
            else:
                symbols_ = set()
                symbols_.update([s for s in self.open_orders if self.open_orders[s]])
                symbols_.update([s for s in self.get_symbols_with_pos()])
                if hasattr(self, "active_symbols") and self.active_symbols:
                    symbols_.update(list(self.active_symbols))
                # Instead of launching all fetches simultaneously, we need to await them one by one
                # so that our 20/10s wrapper can work properly.
                fetched_lists = []
                for symbol_ in sorted(symbols_):
                    fetched_part = await self._ccxt_request(self.cca.fetch_open_orders, symbol_)
                    fetched_lists.append(fetched_part)
                fetched = [x for sublist in fetched_lists for x in sublist]

            for elm in fetched:
                elm["position_side"] = elm["info"]["positionSide"].lower()
                elm["qty"] = elm["amount"]
                open_orders[elm["id"]] = elm

            return sorted(open_orders.values(), key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error fetching open orders {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_positions(self) -> ([dict], float):
        fetched_positions, fetched_balance = None, None
        try:
            # ───── wrap both CCXT calls in one gather ─────
            fetched_positions, fetched_balance = await asyncio.gather(
                self._ccxt_request(self.cca.fapiprivatev3_get_positionrisk),
                self._ccxt_request(self.cca.fetch_balance),
            )
            positions = []
            for elm in fetched_positions:
                if float(elm["positionAmt"]) != 0.0:
                    positions.append(
                        {
                            "symbol": self.get_symbol_id_inv(elm["symbol"]),
                            "position_side": elm["positionSide"].lower(),
                            "size": float(elm["positionAmt"]),
                            "price": float(elm["entryPrice"]),
                        }
                    )
            balance = float(fetched_balance["info"]["totalCrossWalletBalance"])
            if not hasattr(self, "previous_rounded_balance"):
                self.previous_rounded_balance = balance
            self.previous_rounded_balance = pbr.hysteresis_rounding(
                balance,
                self.previous_rounded_balance,
                self.hyst_rounding_balance_pct,
                self.hyst_rounding_balance_h,
            )
            return positions, self.previous_rounded_balance
        except Exception as e:
            logging.error(f"error fetching positions {e}")
            print_async_exception(fetched_positions)
            print_async_exception(fetched_balance)
            traceback.print_exc()
            return False

    async def fetch_tickers(self):
        fetched = None
        try:
            # ───── wrap the CCXT fapipublic_get_ticker_bookticker call ─────
            fetched = await self._ccxt_request(self.cca.fapipublic_get_ticker_bookticker)
            tickers = {
                self.get_symbol_id_inv(elm["symbol"]): {
                    "bid": float(elm["bidPrice"]),
                    "ask": float(elm["askPrice"]),
                }
                for elm in fetched
            }
            for sym in tickers:
                tickers[sym]["last"] = np.random.choice([tickers[sym]["bid"], tickers[sym]["ask"]])
            return tickers
        except Exception as e:
            logging.error(f"error fetching tickers {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            if "bybit does not have market symbol" in str(e):
                # ccxt is raising bad symbol error
                # restart might help…
                raise Exception("ccxt gives bad symbol error… attempting bot restart")
            return False

    async def fetch_ohlcv(self, symbol: str, timeframe="1m"):
        fetched = None
        try:
            # ───── wrap the CCXT fetch_ohlcv call ─────
            fetched = await self._ccxt_request(self.cca.fetch_ohlcv, symbol, timeframe, 1000)
            return fetched
        except Exception as e:
            logging.error(f"error fetching ohlcv for {symbol} {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_pnls(
        self,
        start_time: int = None,
        end_time: int = None,
        limit: int = None,
    ):
        pnls = await self.fetch_pnls_sub(start_time, end_time, limit)
        symbols = sorted(set(self.positions) | set([x["symbol"] for x in pnls]))
        tasks = {}
        for symbol in symbols:
            tasks[symbol] = asyncio.create_task(
                self.fetch_fills_sub(symbol, start_time, end_time, limit)
            )
        fills = {}
        for symbol in tasks:
            fills[symbol] = await tasks[symbol]
        fills = flatten(fills.values())

        if start_time:
            pnls = [x for x in pnls if x["timestamp"] >= start_time]
            fills = [x for x in fills if x["timestamp"] >= start_time]
        unified = {x["id"]: x for x in pnls}
        for x in fills:
            if x["id"] in unified:
                unified[x["id"]].update(x)
            else:
                unified[x["id"]] = x
        result = []
        for x in sorted(unified.values(), key=lambda x: x["timestamp"]):
            if "position_side" not in x:
                logging.info(f"debug: pnl without corresponding fill {x}")
                x["position_side"] = "unknown"
            result.append(x)
        return result

    async def fetch_pnls_sub(
        self,
        start_time: int = None,
        end_time: int = None,
        limit: int = None,
    ):
        if limit is None:
            limit = 1000
        else:
            limit = min(limit, 1000)
        if start_time is None and end_time is None:
            return await self.fetch_pnl(limit=limit)
        all_fetched = {}
        while True:
            fetched = await self.fetch_pnl(start_time, end_time, limit)
            if fetched == []:
                break
            if fetched[0]["tradeId"] in all_fetched and fetched[-1]["tradeId"] in all_fetched:
                break
            for elm in fetched:
                all_fetched[elm["tradeId"]] = elm
            if start_time and end_time and len(fetched) < limit:
                # means fetched all pnls in [start_time, end_time]
                break
            logging.info(f"fetched pnls until {ts_to_date_utc(fetched[-1]['timestamp'])[:19]}")
            start_time = fetched[-1]["timestamp"]
        return sorted(all_fetched.values(), key=lambda x: x["timestamp"])

    async def fetch_fills_sub(self, symbol, start_time=None, end_time=None, limit=None):
        try:
            if symbol not in self.markets_dict:
                return []
            max_limit = 1000
            limit = min(max_limit, limit) if limit else max_limit
            all_fills = {}

            if start_time is None and end_time is None:
                # ───── wrap the CCXT fetch_my_trades call ─────
                fills = await self._ccxt_request(self.cca.fetch_my_trades, symbol, limit)
                for x in fills:
                    all_fills[x["id"]] = x
            elif start_time is None:
                fills = await self._ccxt_request(
                    self.cca.fetch_my_trades, symbol, limit, {"endTime": int(end_time)}
                )
                for x in fills:
                    all_fills[x["id"]] = x
            else:
                if end_time is None:
                    end_time = self.get_exchange_time() + 1000 * 60 * 60
                params = {}
                week = 1000 * 60 * 60 * 24 * 7.0
                start_time_sub = start_time
                while True:
                    fills = await self._ccxt_request(
                        self.cca.fetch_my_trades,
                        symbol,
                        limit,
                        {
                            "startTime": int(min(start_time_sub, self.get_exchange_time() - 1000 * 60)),
                            "endTime":   int(min(end_time, start_time_sub + week * 0.999)),
                        },
                    )
                    if not fills:
                        if end_time - start_time_sub < week * 0.9:
                            self.debug_print("debug fetch_fills_sub a", symbol)
                            break
                        else:
                            logging.info(
                                f"fetched 0 fills for {symbol} between "
                                f"{ts_to_date_utc(start_time_sub)[:19]} and {ts_to_date_utc(end_time)[:19]}"
                            )
                            start_time_sub += week
                            continue
                    if fills[0]["id"] in all_fills and fills[-1]["id"] in all_fills:
                        if end_time - start_time_sub < week * 0.9:
                            self.debug_print("debug fetch_fills_sub b", symbol)
                            break
                        else:
                            logging.info(
                                f"fetched 0 new fills for {symbol} between "
                                f"{ts_to_date_utc(start_time_sub)[:19]} and {ts_to_date_utc(end_time)[:19]}"
                            )
                            start_time_sub += week
                            continue
                    else:
                        for x in fills:
                            all_fills[x["id"]] = x
                    if end_time - start_time_sub < week * 0.9 and len(fills) < limit:
                        self.debug_print("debug fetch_fills_sub c", symbol)
                        break
                    start_time_sub = fills[-1]["timestamp"]
                    logging.info(
                        f"fetched {len(fills)} fill{'s' if len(fills) > 1 else ''} for {symbol} "
                        f"{ts_to_date_utc(fills[0]['timestamp'])[:19]}"
                    )
            all_fills = sorted(all_fills.values(), key=lambda x: x["timestamp"])
            for i in range(len(all_fills)):
                all_fills[i]["pnl"] = float(all_fills[i]["info"]["realizedPnl"])
                all_fills[i]["position_side"] = all_fills[i]["info"]["positionSide"].lower()
            return all_fills
        except Exception as e:
            logging.error(f"error with fetch_fills_sub {symbol} {e}")
            return []

    async def fetch_pnl(
        self,
        start_time: int = None,
        end_time: int = None,
        limit: int = None,
    ):
        fetched = None
        max_limit = 1000
        if limit is None:
            limit = max_limit
        try:
            params = {"incomeType": "REALIZED_PNL", "limit": min(max_limit, limit)}
            if start_time is not None:
                params["startTime"] = int(start_time)
            if end_time is not None:
                params["endTime"] = int(end_time)
            # ───── wrap CCXT’s fapiprivate_get_income ─────
            fetched = await self._ccxt_request(self.cca.fapiprivate_get_income, params=params)
            for i in range(len(fetched)):
                fetched[i]["symbol"] = self.get_symbol_id_inv(fetched[i]["symbol"])
                fetched[i]["pnl"] = float(fetched[i]["income"])
                fetched[i]["timestamp"] = float(fetched[i]["time"])
                fetched[i]["id"] = fetched[i]["tradeId"]
            return sorted(fetched, key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error with fetch_pnl {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def execute_cancellation(self, order: dict) -> dict:
        executed = None
        try:
            # ───── wrap CCXT’s cancel_order ─────
            executed = await self._ccxt_request(self.cca.cancel_order, order["id"], order["symbol"])
            return executed
        except Exception as e:
            logging.error(f"error cancelling order {order} {e}")
            if "-2011" not in str(e):
                print_async_exception(executed)
                traceback.print_exc()
            return {}

    async def execute_cancellations(self, orders: [dict]) -> [dict]:
        if len(orders) == 0:
            return []
        if len(orders) == 1:
            return [await self.execute_cancellation(orders[0])]
        return await self.execute_multiple(orders, "execute_cancellation")

    async def execute_order(self, order: dict) -> dict:
        executed = None
        try:
            order_type = order["type"] if "type" in order else "limit"
            params = {
                "positionSide":    order["position_side"].upper(),
                "newClientOrderId": order["custom_id"],
            }
            if order_type == "limit":
                params["timeInForce"] = (
                    "GTX" if self.config["live"]["time_in_force"] == "post_only" else "GTC"
                )
            # ───── wrap CCXT’s create_order ─────
            executed = await self._ccxt_request(
                self.cca.create_order,
                symbol=order["symbol"],
                type=order_type,
                side=order["side"],
                amount=abs(order["qty"]),
                price=order["price"],
                params=params,
            )
            return executed
        except Exception as e:
            logging.error(f"error executing order {order} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return {}

    async def execute_orders(self, orders: [dict]) -> [dict]:
        if len(orders) == 0:
            return []
        if len(orders) == 1:
            return [await self.execute_order(orders[0])]

        to_execute = []
        for order in orders:
            params = {
                "positionSide":    order["position_side"].upper(),
                "newClientOrderId": order["custom_id"],
            }
            if order["type"] == "limit":
                params["timeInForce"] = (
                    "GTX" if self.config["live"]["time_in_force"] == "post_only" else "GTC"
                )
            to_execute.append({
                "type":   "limit",
                "symbol": order["symbol"],
                "side":   order["side"],
                "amount": abs(order["qty"]),
                "price":  order["price"],
                "params": params.copy(),
            })

        try:
            # ───── wrap CCXT’s create_orders ─────
            executed = await self._ccxt_request(self.cca.create_orders, to_execute)
            return executed
        except Exception as e:
            logging.error(f"error executing orders {orders} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return []

    async def update_exchange_config_by_symbols(self, symbols):
        coros_to_call_lev = {}
        coros_to_call_margin_mode = {}
        for symbol in symbols:
            try:
                coros_to_call_margin_mode[symbol] = asyncio.create_task(
                    self._ccxt_request(self.cca.set_margin_mode, "cross", symbol=symbol)
                )
            except Exception as e:
                logging.error(f"{symbol}: error setting cross mode {e}")
            try:
                coros_to_call_lev[symbol] = asyncio.create_task(
                    self._ccxt_request(self.cca.set_leverage, int(self.live_configs[symbol]["leverage"]), symbol=symbol)
                )
            except Exception as e:
                logging.error(f"{symbol}: a error setting leverage {e}")

        for symbol in symbols:
            res = None
            to_print = ""
            try:
                res = await coros_to_call_lev[symbol]
                to_print += f"set leverage {res} "
            except Exception as e:
                logging.error(f"{symbol}: b error setting leverage {e}")
            try:
                res = await coros_to_call_margin_mode[symbol]
                to_print += f"set cross mode {res}"
            except:
                logging.error(f"error setting cross mode {res}")
            if to_print:
                logging.info(f"{symbol}: {to_print}")

    async def update_exchange_config(self):
        try:
            # ───── wrap CCXT’s set_position_mode ─────
            res = await self._ccxt_request(self.cca.set_position_mode, True)
            logging.info(f"set hedge mode {res}")
        except Exception as e:
            if '"code":-4059' in e.args[0]:
                logging.info(f"hedge mode: {e}")
            else:
                logging.error(f"error setting hedge mode {e}")

    async def determine_utc_offset(self, verbose=True):
        # returns millis to add to utc to get exchange timestamp
        # ───── wrap CCXT’s fetch_ticker ─────
        result = await self._ccxt_request(self.cca.fetch_ticker, "BTC/USDT:USDT")
        self.utc_offset = round((result["timestamp"] - utc_ms()) / (1000 * 60 * 60)) * (1000 * 60 * 60)
        if verbose:
            logging.info(f"Exchange time offset is {self.utc_offset}ms compared to UTC")

    async def fetch_ohlcvs_1m(self, symbol: str, since: float = None, limit=None):
        n_candles_limit = 1500 if limit is None else limit
        if since is None:
            # ───── wrap CCXT’s fetch_ohlcv ─────
            result = await self._ccxt_request(self.cca.fetch_ohlcv, symbol, "1m", n_candles_limit)
            return result

        since = since // 60000 * 60000
        max_n_fetches = 5000 // n_candles_limit
        all_fetched = []
        for i in range(max_n_fetches):
            fetched = await self._ccxt_request(self.cca.fetch_ohlcv, symbol, "1m", n_candles_limit, since=since, limit=n_candles_limit)
            all_fetched += fetched
            if len(fetched) < n_candles_limit:
                break
            since = fetched[-1][0]
        all_fetched_d = {x[0]: x for x in all_fetched}
        return sorted(all_fetched_d.values(), key=lambda x: x[0])

    def format_custom_ids(self, orders: [dict]) -> [dict]:
        # Binance needs broker code (“x-<broker>”) at the start of custom_id
        new_orders = []
        for order in orders:
            order["custom_id"] = (
                "x-" + self.broker_code_fut + shorten_custom_id(order["custom_id"]) + uuid4().hex
            )[: self.custom_id_max_length]
            new_orders.append(order)
        return new_orders
