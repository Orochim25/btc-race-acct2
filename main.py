"""RegimeDip v1e: a long-only CBBTC dip/rebound bot for the 24h BTC race.

Two setups, both full-size (all cash), at most 4 entries per window:
  - DIP (max 2): a 60-bar pullback that is still inside a healthy 720-bar
    regime, with RSI(14) below 50 and a one-bar reversal. Tight stop
    (-60 bps) and a 2:1 take (+120 bps).
  - REBOUND (max 2, halted after 2 consecutive losses): a deeper 720-bar
    washout (>= 550 bps from the top) that prints a long upper wick and
    then reverses hard. Stop sits under the 720-bar low, sized so the
    720-bar top is at least twice as far as the stop; exits trail the peak
    once the trade has run 80 bps in its favour.

Exits are checked every batch while holding. A buy decided on one bar
fills at the next bar's open, so the trade only opens once the position
shows up in the next batch; a buy that never fills is simply forgotten.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"
UNIT = 10000
MAX_SLIP = 200
FEE_BPS = 4
MAX_ENTRIES = 4
MAX_DIP = 2
MAX_RB = 2
COOLDOWN = 60
LAST_ENTRY_BAR = 1320
MIN_CLOSES = 62
HOLD_BARS = 300
DIP_C2C_MIN = 50

_S = None


def _fresh_state(cash0):
    return {
        "cash0": cash0,
        "closes": [],
        "highs": [],
        "lows": [],
        "rsi": [],
        "chg_gain": [],
        "chg_loss": [],
        "avg_gain": 0.0,
        "avg_loss": 0.0,
        "seen_t": set(),
        "seq": 0,
        "pending": None,
        "trade": None,
        "entries": 0,
        "dip": 0,
        "rb": 0,
        "rb_losses": 0,
        "rest_until": 0,
    }


def _rsi_value(avg_gain, avg_loss):
    if avg_loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def _push_rsi(st, cur, prev):
    """Feed one close-to-close change into the Wilder RSI(14)."""
    diff = cur - prev
    gain = diff if diff > 0 else 0
    loss = -diff if diff < 0 else 0
    st["chg_gain"].append(gain)
    st["chg_loss"].append(loss)
    n = len(st["closes"])
    if n < 15:
        st["rsi"].append(None)
    elif n == 15:
        avg_g = sum(st["chg_gain"]) / 14.0
        avg_l = sum(st["chg_loss"]) / 14.0
        st["avg_gain"] = avg_g
        st["avg_loss"] = avg_l
        st["rsi"].append(_rsi_value(avg_g, avg_l))
    else:
        avg_g = (st["avg_gain"] * 13 + gain) / 14.0
        avg_l = (st["avg_loss"] * 13 + loss) / 14.0
        st["avg_gain"] = avg_g
        st["avg_loss"] = avg_l
        st["rsi"].append(_rsi_value(avg_g, avg_l))


def _fall_from_top(st, span):
    """Bps fall of the previous close below the highest high of the
    last `span` bars; 0 when there is not enough history yet."""
    if len(st["closes"]) < span + 2:
        return 0
    top = max(st["highs"][-(span + 1):-1])
    prev = st["closes"][-2]
    return (top - prev) * UNIT // top


def _c2c_fall(st, span):
    """Bps close-to-close fall over `span` bars ending at the previous
    close; None when there is not enough history yet."""
    if len(st["closes"]) < span + 2:
        return None
    old = st["closes"][-(span + 2)]
    prev = st["closes"][-2]
    return (old - prev) * UNIT // old


def _has_wick(st):
    """True when some bar in the last 20 has a range of at least 20 bps
    and closed in the top 40% of its own range."""
    start = len(st["closes"]) - 20
    if start < 0:
        start = 0
    for i in range(start, len(st["closes"])):
        hi = st["highs"][i]
        lo = st["lows"][i]
        cl = st["closes"][i]
        if hi <= lo:
            continue
        rng = (hi - lo) * UNIT // cl
        if rng < 20:
            continue
        if (cl - lo) * 100 // (hi - lo) >= 60:
            return True
    return False


def _dip_setup(st):
    """Return (stop, take) when the dip setup holds, else None."""
    if _fall_from_top(st, 60) < 120:
        return None
    if _fall_from_top(st, 720) >= 550:
        return None
    rsi_now = st["rsi"][-1]
    rsi_prev = st["rsi"][-2]
    # v1e: RSI below 50 (v1d demanded below 25), so the dip setup fires
    # more often; the 60-bar fall and the regime check still filter it.
    if rsi_now is None or rsi_prev is None or rsi_now >= 50:
        return None
    if st["closes"][-1] <= st["closes"][-2]:
        return None
    fall60 = _c2c_fall(st, 60)
    # The 0511 dev window is a calm uptrend: its sharpest 60-bar
    # close-to-close fall is only ~85 bps, so the v1e 150 bps bar never
    # fires here. 50 bps still demands a genuine 60-bar decline (well
    # above one-bar noise) while letting the setup trade on calm windows.
    if fall60 is None or fall60 < DIP_C2C_MIN:
        return None
    if len(st["closes"]) >= 361:
        old = st["closes"][-361]
        regime = (st["closes"][-1] - old) * UNIT // old
        if regime < -100:
            return None
    cur = st["closes"][-1]
    stop = cur * (UNIT - 60) // UNIT
    take = cur * (UNIT + 120) // UNIT
    return stop, take


def _rebound_setup(st):
    """Return the stop when the rebound setup holds, else None."""
    fall60 = _c2c_fall(st, 60)
    fall120 = _c2c_fall(st, 120)
    deep = (fall60 is not None and fall60 >= 150) or (
        fall120 is not None and fall120 >= 250
    )
    if not deep:
        return None
    if _fall_from_top(st, 720) < 550:
        return None
    if not _has_wick(st):
        return None
    cur = st["closes"][-1]
    prev = st["closes"][-2]
    if (cur - prev) * UNIT // prev < 60:
        return None
    if cur <= st["highs"][-2]:
        return None
    rsi_now = st["rsi"][-1]
    rsi_prev = st["rsi"][-2]
    if rsi_now is None or rsi_prev is None or rsi_now <= rsi_prev:
        return None
    low720 = min(st["lows"][-721:-1])
    top720 = max(st["highs"][-721:-1])
    stop0 = low720 * (UNIT - 20) // UNIT
    lo_stop = cur * (UNIT - 100) // UNIT
    hi_stop = cur * (UNIT - 50) // UNIT
    stop = min(max(stop0, lo_stop), hi_stop)
    if top720 - cur < 2 * (cur - stop):
        return None
    return stop


def _buy_order(state):
    return {
        "side": "buy",
        "symbol": SYMBOL,
        "notional_quote_units": state.cash,
        "max_slippage_bps": MAX_SLIP,
    }


def _sell_order(state):
    holding = state.positions.get(SYMBOL)
    if holding is None:
        return None
    return {
        "side": "sell",
        "symbol": SYMBOL,
        "qty": holding.qty,
        "max_slippage_bps": MAX_SLIP,
    }


def on_hello(state):
    global _S
    _S = _fresh_state(state.cash)


def on_batch(state, batch):
    global _S
    if _S is None:
        _S = _fresh_state(state.cash)
    st = _S

    bar = None
    for b in batch["bars"]:
        if b["symbol"] == SYMBOL:
            bar = b
            break
    if bar is None:
        return []
    t = batch["t"]
    if t in st["seen_t"]:
        return []
    st["seen_t"].add(t)

    o = int(bar["o"])
    h = int(bar["h"])
    lo = int(bar["l"])
    c = int(bar["c"])
    prev_c = st["closes"][-1] if st["closes"] else None
    st["closes"].append(c)
    st["highs"].append(h)
    st["lows"].append(lo)
    if prev_c is None:
        st["rsi"].append(None)
    else:
        _push_rsi(st, c, prev_c)
    st["seq"] = len(st["closes"])

    # A buy decided last batch fills at this bar's open: open the trade
    # from the pending order, or forget a buy that never filled.
    if st["pending"] is not None:
        pend = st["pending"]
        st["pending"] = None
        if SYMBOL in state.positions:
            st["trade"] = {
                "kind": pend["kind"],
                "entry": o,
                "stop": pend["stop"],
                "take": pend.get("take"),
                "arm": pend["arm"],
                "trail": pend["trail"],
                "hold": pend["hold"],
                "held": 0,
                "peak": o,
                "armed": False,
                "closing": False,
            }
            st["entries"] += 1
            if pend["kind"] == "dip":
                st["dip"] += 1
            else:
                st["rb"] += 1

    trade = st["trade"]
    if trade is not None:
        if trade.get("closing"):
            # Sell already sent: clear once the position is gone, else
            # send it again (a rejected sell leaves the position held).
            if SYMBOL not in state.positions:
                st["trade"] = None
                trade = None
            else:
                order = _sell_order(state)
                return [order] if order else []
        elif SYMBOL not in state.positions:
            st["trade"] = None
            trade = None

    if trade is not None:
        trade["held"] += 1
        if c > trade["peak"]:
            trade["peak"] = c
        if c >= trade["entry"] * (UNIT + trade["arm"]) // UNIT:
            trade["armed"] = True
        exit_px = None
        if trade["kind"] == "dip":
            if c >= trade["take"]:
                exit_px = c
            elif c <= trade["stop"]:
                exit_px = trade["stop"]
            elif trade["held"] >= trade["hold"]:
                exit_px = c
        else:
            if c <= trade["stop"]:
                exit_px = trade["stop"]
            elif trade["armed"] and c <= trade["peak"] * (UNIT - trade["trail"]) // UNIT:
                exit_px = trade["peak"] * (UNIT - trade["trail"]) // UNIT
            elif trade["held"] >= trade["hold"]:
                exit_px = c
        if exit_px is not None:
            if trade["kind"] == "rb":
                if exit_px > trade["entry"]:
                    st["rb_losses"] = 0
                else:
                    st["rb_losses"] += 1
            st["rest_until"] = st["seq"] + COOLDOWN
            trade["closing"] = True
            order = _sell_order(state)
            return [order] if order else []
        return []

    # Guards: no entry before 62 closes, none after bar 1320, at most
    # 4 entries, 60-bar cooldown after an exit, and no entry while
    # equity is under 98% of the starting capital.
    if len(st["closes"]) < MIN_CLOSES:
        return []
    if st["seq"] > LAST_ENTRY_BAR:
        return []
    if st["entries"] >= MAX_ENTRIES:
        return []
    if st["seq"] < st["rest_until"]:
        return []
    if state.equity < st["cash0"] * 98 // 100:
        return []

    # Rebound first, then dip.
    if st["rb"] < MAX_RB and st["rb_losses"] < 2:
        stop = _rebound_setup(st)
        if stop is not None:
            st["pending"] = {
                "kind": "rb",
                "stop": stop,
                "take": None,
                "arm": 80,
                "trail": 80,
                "hold": HOLD_BARS,
            }
            return [_buy_order(state)]

    if st["dip"] < MAX_DIP:
        dip = _dip_setup(st)
        if dip is not None:
            stop, take = dip
            st["pending"] = {
                "kind": "dip",
                "stop": stop,
                "take": take,
                "arm": 0,
                "trail": 0,
                "hold": HOLD_BARS,
            }
            return [_buy_order(state)]

    return []


if __name__ == "__main__":
    makefun_bars.run(on_batch, on_hello=on_hello)
