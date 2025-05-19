from passivbot import Passivbot, logging
from uuid import uuid4
from njit_funcs import round_
import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
import pprint
import asyncio
import traceback
import numpy as np
import json
from copy import deepcopy
from pure_funcs import (
    floatify,
    ts_to_date_utc,
    calc_hash,
    determine_pos_side_ccxt,
    flatten,
    shorten_custom_id,
    hysteresis_rounding,
)
from procedures import print_async_exception, utc_ms, assert_correct_ccxt_version, load_broker_code

assert_correct_ccxt_version(ccxt=ccxt_async)


class BinanceBot(Passivbot):
    def __init__(self, config: dict):
        # Set request_timeout before calling super().__init__
        self.request_timeout = 60000  # increased from 30000 to 60000 milliseconds
        # Enhanced retry settings
        self.max_retries = 10  # increased from 5
        self.retry_delay = 3  # increased from 2 seconds
        
        super().__init__(config)
        self.custom_id_max_length = 36
        self.session_active = True  # Flag to track session status
        
        # Initialize context manager for resources
        self._resources = []

    async def __cleanup_resources(self):
        """Close all active resources properly"""
        for resource in self._resources:
            try:
                if hasattr(resource, 'close') and callable(resource.close):
                    await resource.close()
            except Exception as e:
                logging.error(f"Error closing resource: {e}")
        self._resources = []

    async def shutdown(self):
        """Properly shutdown the bot and cleanup resources"""
        self.session_active = False
        self.stop_websocket = True
        await self.__cleanup_resources()
        logging.info("BinanceBot resources cleaned up successfully")

    def create_ccxt_sessions(self):
        self.broker_code_spot = load_broker_code("binance_spot")
        for ccx, ccxt_module in [("cca", ccxt_async), ("ccp", ccxt_pro)]:
            exchange_class = getattr(ccxt_module, "binanceusdm")
            session = exchange_class(
                {
                    "apiKey": self.user_info["key"],
                    "secret": self.user_info["secret"],
                    "password": self.user_info["passphrase"],
                    "timeout": self.request_timeout,
                    "enableRateLimit": True,
                }
            )
            session.options["defaultType"] = "swap"
            # Increase recvWindow parameter to avoid timestamp issues
            session.options["recvWindow"] = 60000  
            
            if self.broker_code:
                for key in ["future", "delivery", "swap", "option"]:
                    session.options["broker"][key] = "x-" + self.broker_code
            if self.broker_code_spot:
                for key in ["spot", "margin"]:
                    session.options["broker"][key] = "x-" + self.broker_code_spot
                    
            setattr(self, ccx, session)
            # Add session to resources for cleanup
            self._resources.append(session)

    async def _execute_with_retry(self, func, max_retries=None, retry_delay=None):
        """Execute a function with enhanced retry logic for handling connection issues"""
        if max_retries is None:
            max_retries = self.max_retries
        if retry_delay is None:
            retry_delay = self.retry_delay
            
        last_error = None
        for attempt in range(max_retries):
            try:
                return await func()
            except (ccxt_async.NetworkError, ccxt_async.RequestTimeout) as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Calculate delay with jitter to prevent thundering herd problem
                    base_delay = retry_delay * (2 ** attempt)  # Exponential backoff
                    jitter = base_delay * 0.2 * (np.random.random() * 2 - 1)  # ±20% jitter
                    delay = max(0.5, base_delay + jitter)  # Ensure minimum delay
                    
                    error_type = "Network error" if isinstance(e, ccxt_async.NetworkError) else "Request timeout"
                    logging.warning(f"{error_type}: {e}. Retrying in {delay:.2f}s... ({attempt+1}/{max_retries})")
                    await asyncio.sleep(delay)
            except ccxt_async.ExchangeError as e:
                if "rate limit" in str(e).lower():
                    last_error = e
                    if attempt < max_retries - 1:
                        # Much longer delay for rate limits
                        delay = retry_delay * (3 ** attempt)  # More aggressive backoff for rate limits
                        logging.warning(f"Rate limit exceeded: {e}. Retrying in {delay}s... ({attempt+1}/{max_retries})")
                        await asyncio.sleep(delay)
                else:
                    # For non-rate limit exchange errors, check if it's retriable
                    if any(retriable_err in str(e).lower() for retriable_err in ["timeout", "connection", "network", "socket"]):
                        last_error = e
                        if attempt < max_retries - 1:
                            delay = retry_delay * (2 ** attempt)
                            logging.warning(f"Retriable exchange error: {e}. Retrying in {delay}s... ({attempt+1}/{max_retries})")
                            await asyncio.sleep(delay)
                    else:
                        # Non-retriable exchange error
                        raise e
        
        # If we've exhausted all retries, raise the last error
        if last_error:
            logging.error(f"Failed after {max_retries} attempts: {last_error}")
            raise last_error
        return None

    async def print_new_user_suggestion(self):
        between_print_wait_ms = 1000 * 60 * 60 * 4
        if hasattr(self, "previous_user_suggestion_print_ts"):
            if utc_ms() - self.previous_user_suggestion_print_ts < between_print_wait_ms:
                return
        self.previous_user_suggestion_print_ts = utc_ms()

        try:
            res = await self._execute_with_retry(
                lambda: self.cca.fapiprivate_get_apireferral_ifnewuser(
                    params={"brokerid": self.broker_code}
                )
            )
            if not res or not isinstance(res, dict):
                return
            if res.get("ifNewUser") and res.get("rebateWorking"):
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
        except Exception as e:
            logging.error(f"failed to fetch fapiprivate_get_apireferral_ifnewuser {e}")

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
        while self.session_active and not self.stop_websocket:
            try:
                res = await self.ccp.watch_balance()
                self.handle_balance_update(res)
            except Exception as e:
                print(f"exception watch_balance", e)
                traceback.print_exc()
                if not self.session_active:
                    break
                await asyncio.sleep(3)  # Increased sleep time for stability

    async def watch_orders(self):
        while self.session_active and not self.stop_websocket:
            try:
                res = await self._execute_with_retry(lambda: self.ccp.watch_orders())
                for i in range(len(res)):
                    # Safely access position side with proper error handling
                    if isinstance(res[i], dict) and 'info' in res[i] and isinstance(res[i]['info'], dict) and 'ps' in res[i]['info']:
                        res[i]["position_side"] = res[i]["info"]["ps"].lower()
                    else:
                        res[i]["position_side"] = "none"  # Default value if missing
                    res[i]["qty"] = res[i].get("amount", 0)
                self.handle_order_update(res)
            except Exception as e:
                if "Abnormal closure of client" not in str(e):
                    print(f"exception watch_orders", e)
                    traceback.print_exc()
                if not self.session_active:
                    break
                await asyncio.sleep(3)  # Increased sleep time for stability

    async def fetch_open_orders(self, symbol: str = None, all=False) -> [dict]:
        try:
            # binance has expensive fetch_open_orders without specified symbol
            if all:
                self.cca.options["warnOnFetchOpenOrdersWithoutSymbol"] = False
                logging.info(f"fetching all open orders for binance")
                fetched = await self._execute_with_retry(lambda: self.cca.fetch_open_orders())
                self.cca.options["warnOnFetchOpenOrdersWithoutSymbol"] = True
            else:
                symbols_ = set()
                symbols_.update([s for s in self.open_orders if self.open_orders[s]])
                symbols_.update([s for s in self.get_symbols_with_pos()])
                if hasattr(self, "active_symbols") and self.active_symbols:
                    symbols_.update(list(self.active_symbols))
                
                # Create tasks for each symbol with proper error handling
                tasks = []
                for symbol in sorted(symbols_):
                    tasks.append(self._execute_with_retry(
                        lambda s=symbol: self.cca.fetch_open_orders(symbol=s)
                    ))
                
                # Execute all tasks and handle errors
                results = await asyncio.gather(*tasks, return_exceptions=True)
                fetched = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        symbol_name = sorted(symbols_)[i]
                        logging.error(f"Error fetching orders for {symbol_name}: {result}")
                    else:
                        fetched.extend(result)
                
            open_orders = {}
            for elm in fetched:
                # Safely access data with proper error handling
                if isinstance(elm, dict) and 'info' in elm and isinstance(elm['info'], dict):
                    elm["position_side"] = elm["info"].get("positionSide", "").lower()
                    elm["qty"] = elm.get("amount", 0)
                    open_orders[elm["id"]] = elm
            return sorted(open_orders.values(), key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error fetching open orders {e}")
            traceback.print_exc()
            return []

    async def fetch_positions(self) -> ([dict], float):
        try:
            # We'll use the retry wrapper for each API call separately
            fetch_positions_task = self._execute_with_retry(
                lambda: self.cca.fapiprivatev3_get_positionrisk()
            )
            fetch_balance_task = self._execute_with_retry(
                lambda: self.cca.fetch_balance()
            )
            
            # Wait for both tasks to complete
            fetched_positions, fetched_balance = await asyncio.gather(
                fetch_positions_task, fetch_balance_task
            )
            
            positions = []
            for elm in fetched_positions:
                # Ensure we don't get type errors by checking data properly
                pos_amt = float(elm.get("positionAmt", 0))
                if pos_amt != 0.0:
                    positions.append(
                        {
                            "symbol": self.get_symbol_id_inv(elm["symbol"]),
                            "position_side": elm.get("positionSide", "").lower(),
                            "size": pos_amt,
                            "price": float(elm.get("entryPrice", 0)),
                        }
                    )
            
            # Safely access balance data
            if isinstance(fetched_balance, dict) and "info" in fetched_balance:
                balance = float(fetched_balance["info"].get("totalCrossWalletBalance", 0))
            else:
                balance = 0
                
            if not hasattr(self, "previous_rounded_balance"):
                self.previous_rounded_balance = balance
            self.previous_rounded_balance = hysteresis_rounding(
                balance, self.previous_rounded_balance, 0.02, 0.5
            )
            return positions, self.previous_rounded_balance
        except Exception as e:
            logging.error(f"error fetching positions {e}")
            traceback.print_exc()
            return [], 0.0

    async def fetch_tickers(self):
        try:
            fetched = await self._execute_with_retry(
                lambda: self.cca.fapipublic_get_ticker_bookticker()
            )
            tickers = {}
            for elm in fetched:
                if isinstance(elm, dict) and "symbol" in elm:
                    symbol = self.get_symbol_id_inv(elm["symbol"])
                    tickers[symbol] = {
                        "bid": float(elm.get("bidPrice", 0)),
                        "ask": float(elm.get("askPrice", 0)),
                    }
                    tickers[symbol]["last"] = np.random.choice([tickers[symbol]["bid"], tickers[symbol]["ask"]])
            return tickers
        except Exception as e:
            logging.error(f"error fetching tickers {e}")
            traceback.print_exc()
            if "bybit does not have market symbol" in str(e):
                # ccxt is raising bad symbol error
                # restart might help...
                raise Exception("ccxt gives bad symbol error... attempting bot restart")
            return {}

    async def fetch_ohlcv(self, symbol: str, timeframe="1m"):
        try:
            fetched = await self._execute_with_retry(
                lambda: self.cca.fetch_ohlcv(symbol, timeframe=timeframe, limit=1000)
            )
            return fetched
        except Exception as e:
            logging.error(f"error fetching ohlcv for {symbol} {e}")
            traceback.print_exc()
            return []

    async def fetch_pnls(
        self,
        start_time: int = None,
        end_time: int = None,
        limit: int = None,
    ):
        try:
            pnls = await self.fetch_pnls_sub(start_time, end_time, limit)
            symbols = sorted(set(getattr(self, 'positions', {}).keys()) | set([x.get("symbol", "") for x in pnls if isinstance(x, dict)]))
            tasks = {}
            for symbol in symbols:
                if not symbol:  # Skip empty symbols
                    continue
                tasks[symbol] = asyncio.create_task(
                    self.fetch_fills_sub(symbol, start_time, end_time, limit)
                )
            fills = {}
            for symbol in tasks:
                try:
                    fills[symbol] = await tasks[symbol]
                except Exception as e:
                    logging.error(f"Error fetching fills for {symbol}: {e}")
                    fills[symbol] = []
                    
            # Safely flatten fills - ensure they are all lists before flattening
            valid_fills = [fills[symbol] for symbol in fills if isinstance(fills[symbol], list)]
            flattened_fills = flatten(valid_fills)
            
            if start_time:
                pnls = [x for x in pnls if isinstance(x, dict) and x.get("timestamp", 0) >= start_time]
                flattened_fills = [x for x in flattened_fills if isinstance(x, dict) and x.get("timestamp", 0) >= start_time]
                
            unified = {x["id"]: x for x in pnls if isinstance(x, dict) and "id" in x}
            for x in flattened_fills:
                if isinstance(x, dict) and "id" in x:
                    if x["id"] in unified:
                        unified[x["id"]].update(x)
                    else:
                        unified[x["id"]] = x
                        
            result = []
            for x in sorted(unified.values(), key=lambda x: x.get("timestamp", 0)):
                if "position_side" not in x:
                    logging.info(f"debug: pnl without corresponding fill {x}")
                    x["position_side"] = "unknown"
                result.append(x)
            return result
        except Exception as e:
            logging.error(f"Error in fetch_pnls: {e}")
            traceback.print_exc()
            return []

    async def fetch_pnls_sub(
        self,
        start_time: int = None,
        end_time: int = None,
        limit: int = None,
    ):
        try:
            # binance needs symbol specified for fetch fills
            # but can fetch pnls for all symbols
            # fetch fills for all symbols with pos
            # fetch pnls for all symbols
            # fills only needed for symbols with pos for trailing orders
            if limit is None:
                limit = 1000
            if start_time is None and end_time is None:
                return await self.fetch_pnl(limit=limit)
            all_fetched = {}
            while True:
                fetched = await self.fetch_pnl(start_time, end_time, limit)
                if not fetched or fetched == []:
                    break
                if len(fetched) > 0 and "tradeId" in fetched[0] and "tradeId" in fetched[-1]:
                    if fetched[0]["tradeId"] in all_fetched and fetched[-1]["tradeId"] in all_fetched:
                        break
                    for elm in fetched:
                        if "tradeId" in elm:
                            all_fetched[elm["tradeId"]] = elm
                if len(fetched) < limit:
                    break
                logging.info(f"debug fetching pnls {ts_to_date_utc(fetched[-1]['timestamp'])}")
                start_time = fetched[-1]["timestamp"]
            return sorted(all_fetched.values(), key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"Error in fetch_pnls_sub: {e}")
            return []

    async def fetch_fills_sub(self, symbol, start_time=None, end_time=None, limit=None):
        try:
            if symbol not in self.markets_dict:
                return []
            # limit is max 1000
            if limit is None:
                limit = 1000
            if start_time is None:
                all_fills = await self._execute_with_retry(
                    lambda: self.cca.fetch_my_trades(symbol, limit=limit)
                )
            else:
                week = 1000 * 60 * 60 * 24 * 7.0
                all_fills = {}
                if end_time is None:
                    end_time = self.get_exchange_time() + 1000 * 60 * 60
                sts = start_time
                retries = 0
                max_retries = 5  # Increased max retries per chunk
                
                while True:
                    ets = min(end_time, sts + week * 0.999)
                    try:
                        fills = await self._execute_with_retry(
                            lambda: self.cca.fetch_my_trades(
                                symbol, 
                                limit=limit, 
                                params={"startTime": int(sts), "endTime": int(ets)}
                            ),
                            max_retries=max_retries
                        )
                        retries = 0  # Reset retry counter on success
                        
                        if fills:
                            # Check for duplicate data safely
                            has_first = any(fill.get("id") == fills[0].get("id") for fill in all_fills.values()) if fills[0].get("id") else False
                            has_last = any(fill.get("id") == fills[-1].get("id") for fill in all_fills.values()) if fills[-1].get("id") else False
                            
                            if has_first and has_last and len(fills) > 1:
                                break
                                
                            for x in fills:
                                if "id" in x:
                                    all_fills[x["id"]] = x
                                    
                            # Check if we've reached the end
                            last_timestamp = fills[-1].get("timestamp", 0) if fills else 0
                            if last_timestamp >= end_time:
                                break
                                
                            if end_time - sts < week and len(fills) < limit:
                                break
                                
                            sts = last_timestamp
                            logging.info(
                                f"fetched {len(fills)} fill{'s' if len(fills) > 1 else ''} for {symbol} {ts_to_date_utc(fills[0]['timestamp'])}"
                            )
                        else:
                            if end_time - sts < week:
                                break
                            sts = sts + week * 0.999
                        limit = 1000
                    except Exception as e:
                        retries += 1
                        if retries >= max_retries:
                            logging.error(f"Failed to fetch trades for {symbol} after {max_retries} retries: {e}")
                            sts = sts + week * 0.5  # Skip a smaller time period after max retries
                            retries = 0  # Reset for next chunk
                        else:
                            logging.warning(f"Error fetching trades for {symbol}, retry {retries}/{max_retries}: {e}")
                            await asyncio.sleep(2 * (2 ** retries))  # Exponential backoff with higher base
                
                # Convert dictionary to sorted list
                all_fills = sorted(all_fills.values(), key=lambda x: x.get("timestamp", 0))
                
            # Process the fills safely
            for i in range(len(all_fills)):
                if isinstance(all_fills[i], dict) and "info" in all_fills[i]:
                    all_fills[i]["pnl"] = float(all_fills[i]["info"].get("realizedPnl", 0))
                    all_fills[i]["position_side"] = all_fills[i]["info"].get("positionSide", "").lower()
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
        try:
            # max limit is 1000
            if limit is None:
                limit = 1000
            
            params = {"incomeType": "REALIZED_PNL", "limit": 1000}
            if start_time is not None:
                params["startTime"] = int(start_time)
            if end_time is not None:
                params["endTime"] = int(end_time)
                
            fetched = await self._execute_with_retry(
                lambda: self.cca.fapiprivate_get_income(params=params),
                max_retries=8  # Higher retries for this critical operation
            )
            
            # Process the data safely
            processed = []
            for i in range(len(fetched)):
                if isinstance(fetched[i], dict):
                    item = dict(fetched[i])  # Create a copy to avoid modifying the original
                    if "symbol" in item:
                        item["symbol"] = self.get_symbol_id_inv(item["symbol"])
                    item["pnl"] = float(item.get("income", 0))
                    item["timestamp"] = float(item.get("time", 0))
                    item["id"] = item.get("tradeId", "")
                    processed.append(item)
                    
            return sorted(processed, key=lambda x: x.get("timestamp", 0))
        except Exception as e:
            logging.error(f"error with fetch_pnl {e}")
            traceback.print_exc()
            return []

    async def execute_cancellation(self, order: dict) -> dict:
        try:
            executed = await self._execute_with_retry(
                lambda: self.cca.cancel_order(order["id"], symbol=order["symbol"])
            )
            if isinstance(executed, dict) and "code" in executed and executed["code"] == -2011:
                logging.info(f"{executed}")
                return {}
                
            # Ensure we have all the required fields
            if isinstance(executed, dict):
                return {
                    "symbol": executed.get("symbol", ""),
                    "side": executed.get("side", ""),
                    "id": executed.get("id", ""),
                    "position_side": executed.get("info", {}).get("positionSide", "").lower() if "info" in executed else "",
                    "qty": executed.get("amount", 0),
                    "price": executed.get("price", 0),
                }
            return {}
        except Exception as e:
            if "-2011" in str(e):
                # Order does not exist error - already cancelled or filled
                logging.info(f"Order {order.get('id', 'unknown')} already cancelled or filled")
                return {}
            logging.error(f"error cancelling order {order} {e}")
            traceback.print_exc()
            return {}

    async def execute_cancellations(self, orders: [dict]) -> [dict]:
        if len(orders) == 0:
            return []
        if len(orders) == 1:
            return [await self.execute_cancellation(orders[0])]
        return await self.execute_multiple(
            orders, "execute_cancellation", self.config["live"]["max_n_cancellations_per_batch"]
        )

    async def execute_order(self, order: dict) -> dict:
        try:
            order_type = order.get("type", "limit")
            params = {
                "positionSide": order.get("position_side", "").upper(),
                "newClientOrderId": order.get("custom_id", ""),
            }
            if order_type == "limit":
                params["timeInForce"] = (
                    "GTX" if self.config["live"]["time_in_force"] == "post_only" else "GTC"
                )
                
            executed = await self._execute_with_retry(
                lambda: self.cca.create_order(
                    type=order_type,
                    symbol=order.get("symbol", ""),
                    side=order.get("side", ""),
                    amount=abs(order.get("qty", 0)),
                    price=order.get("price", 0),
                    params=params,
                ),
                max_retries=8  # Higher retries for order execution
            )
            
            if isinstance(executed, dict):
                if "info" in executed and "code" in executed["info"] and executed["info"]["code"] == "-5022":
                    logging.info(f"{executed['info'].get('msg', 'Order positioning error')}")
                    return {}
                elif "status" in executed and executed["status"] in ["open", "closed"]:
                    executed["position_side"] = executed.get("info", {}).get("positionSide", "").lower() if "info" in executed else ""
                    executed["qty"] = executed.get("amount", 0)
                    executed["reduce_only"] = executed.get("reduceOnly", False)
                    return executed
            return {}
        except Exception as e:
            logging.error(f"Error executing order: {e}")
            traceback.print_exc()
            return {}

    async def execute_orders(self, orders: list[dict]) -> list[dict]:
        """
        Execute one or more orders. If a single order is provided, delegate to execute_order.
        Otherwise, send in batch respecting max creations per batch.
        """
        if not orders:
            return []

        # Single order: delegate to existing method for retry, normalization, etc.
        if len(orders) == 1:
            result = await self.execute_order(orders[0])
            return [result]

        # Prepare batch
        to_execute = []
        max_batch = self.config["live"]["max_n_creations_per_batch"]
        for order in orders[:max_batch]:
            params = {
                "positionSide": order.get("position_side", "").upper(),
                "newClientOrderId": order.get("custom_id", ""),
            }
            if order.get("type", "limit") == "limit":
                tif = self.config["live"]["time_in_force"]
                params["timeInForce"] = "GTX" if tif == "post_only" else "GTC"

            to_execute.append({
                "type": "limit",
                "symbol": order.get("symbol", ""),
                "side": order.get("side", ""),
                "amount": abs(order.get("qty", 0)),
                "price": order.get("price", 0),
                "params": deepcopy(params),
            })

        # Send batch with retry
        try:
            executed = await self._execute_with_retry(
                lambda: self.cca.create_orders(to_execute),
                max_retries=8,
            )

            # Normalize and filter individual errors
            for idx, item in enumerate(executed):
                if isinstance(item, dict):
                    info = item.get("info", {})
                    item["position_side"] = info.get("positionSide", "").lower()
                    item["qty"] = item.get("amount", 0)
                    item["reduce_only"] = item.get("reduceOnly", False)
                    # Handle client-side rejection code
                    if info.get("code") == "-5022":
                        logging.info(f"Order placement error: {info.get('msg', '')}")
                        executed[idx] = {}
            return executed

        except Exception as e:
            logging.error(f"Error executing batch orders: {e}")
            traceback.print_exc()
            return [{}] * len(to_execute)
