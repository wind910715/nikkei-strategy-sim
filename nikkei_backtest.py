# -*- coding: utf-8 -*-
"""日經策略回測 — 凍結規格 2026-10-28 v1（見 goals/nikkei-futures-strategy-and-live-monitoring/files/nikkei_strategy_spec_2026-10-28.md）
全程現貨指數／CME 期貨代理，結果一律標「現貨指數代理階段、尚未實盤」。
無 Telegram、無憑證、無下單。執行：python nikkei_backtest.py（完整）/ --smoke（僅 5 日程式碼冒煙檢查，不產出結論）。
"""
import json, sys
import numpy as np, pandas as pd, yfinance as yf

SEED = 42
COSTS = (10.0, 20.0)

# ---------------- 資料 ----------------
def load(sym, interval, period):
    d = yf.Ticker(sym).history(period=period, interval=interval)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d.columns = [str(c).lower() for c in d.columns]
    d = d.tz_convert("Asia/Tokyo")
    d = d[["open", "high", "low", "close", "volume"]].copy()
    return d

def mins_of(idx):
    return idx.hour * 60 + idx.minute

def add_ma_atr(d):
    d = d.copy()
    for p in (5, 10, 20, 60):
        d[f"ma{p}"] = d["close"].rolling(p).mean()
    d["atr"] = (d["high"] - d["low"]).rolling(14).mean()
    return d.dropna()

def day_series(d):
    m = mins_of(d.index)
    return d[(m >= 540) & (m < 930)]

def night_series(d):
    m = mins_of(d.index)
    return d[(m >= 990) | (m < 300)]

# ---------------- MA 交叉引擎（原型邏輯忠實移植） ----------------
def run_ma(df, long_only=False, hard_atr=2.0):
    trades = []
    pos = None  # (dir, entry, stop)
    n = len(df)
    for i in range(1, n - 1):
        r, nx = df.iloc[i], df.iloc[i + 1]
        if pos is not None:
            d_, e_, s_ = pos
            hit_stop = (d_ == 1 and r["low"] <= s_) or (d_ == -1 and r["high"] >= s_)
            exit_now = hit_stop
            if d_ == 1 and r["ma5"] < r["ma20"]:
                exit_now = True
            if d_ == -1 and r["ma5"] > r["ma20"]:
                exit_now = True
            if exit_now or i == n - 2:
                px = s_ if hit_stop else nx["open"]
                trades.append({"dir": d_, "entry": e_, "exit": px, "sess": sess_of(r.name)})
                pos = None
            continue
        atr = r["atr"]
        prev = df.iloc[i - 1]
        if prev["ma5"] <= prev["ma20"] and r["ma5"] > r["ma20"]:
            e = nx["open"]; pos = (1, e, e - hard_atr * atr)
        elif not long_only and prev["ma5"] >= prev["ma20"] and r["ma5"] < r["ma20"]:
            e = nx["open"]; pos = (-1, e, e + hard_atr * atr)
    if pos is not None:  # 序列尾強制平倉
        d_, e_, s_ = pos
        trades.append({"dir": d_, "entry": e_, "exit": df.iloc[-1]["close"], "sess": sess_of(df.index[-1])})
    return trades

def sess_of(ts):
    m = ts.hour * 60 + ts.minute
    if 540 <= m < 930:
        return "day"
    return "night"

# ---------------- VCP-lite 引擎（原型規則移植＋可選 regime 濾網） ----------------
def session_key(ts):
    h = ts.hour + ts.minute / 60
    if 9.0 <= h < 15.5:
        return ("day", ts.date())
    if h >= 16.5:
        return ("night", ts.date())
    if h < 6.0:
        from datetime import timedelta
        return ("night", ts.date() - timedelta(days=1))
    return (None, ts.date())

def run_vcp(df15, regime=None, cost_list=COSTS):
    """df15: 15m NKD=F（含日夜）. regime: 可選 DataFrame（60m 全序列，含 ma60），用 merge_asof 取訊號當下最近 60m 收盤與 ma60。"""
    df = df15.copy()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["sess"] = [session_key(ts) for ts in df.index]
    df = df[df["sess"].apply(lambda s: s[0] is not None)]
    if regime is not None:
        reg = regime[["close", "ma60"]].copy()
        reg.index = pd.to_datetime(reg.index)
        df = pd.merge_asof(df.sort_index(), reg.sort_index(), left_index=True, right_index=True,
                           suffixes=("", "_60"), direction="backward")
    trades = []
    for skey, g in df.groupby("sess"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < 30:
            continue
        i = 0; in_pos_until = -1
        while i < n - 6:
            atr = g["atr"].iloc[i]
            if not np.isfinite(atr) or atr <= 0:
                i += 1; continue
            bar = g.iloc[i]
            impulse = 0; imp_size = 0.0
            rng = bar["high"] - bar["low"]
            if rng >= 1.8 * atr:
                if bar["close"] >= bar["low"] + 0.75 * rng:
                    impulse, imp_size = 1, rng
                elif bar["close"] <= bar["low"] + 0.25 * rng:
                    impulse, imp_size = -1, rng
            if impulse == 0 and i >= 3:
                mv = g["close"].iloc[i] - g["close"].iloc[i - 3]
                if abs(mv) >= 2.2 * atr:
                    impulse, imp_size = (1 if mv > 0 else -1), abs(mv)
            if impulse == 0 or i <= in_pos_until:
                i += 1; continue
            if regime is not None:  # 濾網：方向須與 60m regime 一致
                c60 = g["close_60"].iloc[i] if "close_60" in g else np.nan
                m60 = g["ma60"].iloc[i] if "ma60" in g else np.nan
                if not np.isfinite(c60) or not np.isfinite(m60):
                    i += 1; continue
                if (impulse == 1 and c60 <= m60) or (impulse == -1 and c60 >= m60):
                    i += 1; continue
            widths = []; c_hi = c_lo = None; found = -1
            for j in range(i + 1, min(i + 15, n - 1)):
                w = g["high"].iloc[max(j - 2, 0):j + 1].max() - g["low"].iloc[max(j - 2, 0):j + 1].min()
                widths.append(w)
                if len(widths) >= 3 and widths[-1] < widths[-2] < widths[-3] and widths[-1] <= 0.5 * imp_size:
                    c_hi = g["high"].iloc[i + 1:j + 1].max(); c_lo = g["low"].iloc[i + 1:j + 1].min()
                    found = j; break
            if found < 0:
                i += 1; continue
            entered = False
            for k in range(found + 1, min(found + 9, n - 1)):
                bk = g.iloc[k]
                vol_ok = (not np.isfinite(bk["vol_ma"])) or bk["vol_ma"] <= 0 or bk["volume"] >= 1.05 * bk["vol_ma"]
                if impulse == 1 and bk["close"] > c_hi and vol_ok:
                    entry = g["open"].iloc[k + 1]; stop = c_lo; direction = 1
                elif impulse == -1 and bk["close"] < c_lo and vol_ok:
                    entry = g["open"].iloc[k + 1]; stop = c_hi; direction = -1
                else:
                    continue
                R = abs(entry - stop)
                if R <= 0 or R > 3.0 * atr:
                    continue
                target = entry + direction * 2.0 * R
                exit_px = g["close"].iloc[-1]; exit_i = n - 1; be = False
                for m in range(k + 1, n):
                    b = g.iloc[m]
                    if direction == 1:
                        if b["low"] <= stop: exit_px, exit_i = stop, m; break
                        if b["high"] >= target: exit_px, exit_i = target, m; break
                        if not be and b["high"] >= entry + R: stop = entry; be = True
                    else:
                        if b["high"] >= stop: exit_px, exit_i = stop, m; break
                        if b["low"] <= target: exit_px, exit_i = target, m; break
                        if not be and b["low"] <= entry - R: stop = entry; be = True
                    exit_px, exit_i = b["close"], m
                trades.append({"dir": direction, "entry": entry, "exit": exit_px,
                               "sess": skey[0], "Rsize": R})
                in_pos_until = exit_i; i = exit_i; entered = True; break
            if not entered:
                i = found + 1
            else:
                i = in_pos_until + 1
    return trades

# ---------------- ORB（開盤區間突破） ----------------
def run_orb2(df15, target_r=2.0):
    """df15 index 為東京時間 DatetimeIndex。日盤 09:00-15:15。"""
    trades = []
    df = df15.copy()
    df["date"] = df.index.date
    df["mins"] = mins_of(df.index)
    for date, g in df.groupby("date"):
        g = g.sort_index()
        sess = g[(g["mins"] >= 540) & (g["mins"] <= 915)]
        if len(sess) < 4:
            continue
        first2 = sess.iloc[:2]
        rh, rl = first2["high"].max(), first2["low"].min()
        if rh <= rl:
            continue
        entry = None; direction = 0; stop = None; target = None
        bars = sess.iloc[2:]
        for idx in range(len(bars)):
            b = bars.iloc[idx]
            if entry is None:
                if b["close"] > rh:
                    direction, entry, stop = 1, None, rl
                elif b["close"] < rl:
                    direction, entry, stop = -1, None, rh
                else:
                    continue
                if idx + 1 < len(bars):
                    entry = bars.iloc[idx + 1]["open"]
                else:
                    entry = b["close"]
                R = abs(entry - stop)
                if R <= 0:
                    entry = None; continue
                target = entry + direction * target_r * R
                continue
            # 持倉檢查（從進場後次根起）
            if direction == 1:
                if b["low"] <= stop:
                    trades.append({"dir": 1, "entry": entry, "exit": stop, "sess": "day"}); entry = "done"; break
                if b["high"] >= target:
                    trades.append({"dir": 1, "entry": entry, "exit": target, "sess": "day"}); entry = "done"; break
            else:
                if b["high"] >= stop:
                    trades.append({"dir": -1, "entry": entry, "exit": stop, "sess": "day"}); entry = "done"; break
                if b["low"] <= target:
                    trades.append({"dir": -1, "entry": entry, "exit": target, "sess": "day"}); entry = "done"; break
        if entry is not None and entry != "done":
            trades.append({"dir": direction, "entry": entry, "exit": sess.iloc[-1]["close"], "sess": "day"})
    return trades

# ---------------- VWAP 均值回歸 ----------------
def run_vwap(df15, k=1.0, win=20):
    trades = []
    df = df15.copy()
    df["date"] = df.index.date
    df["mins"] = mins_of(df.index)
    for date, g in df.groupby("date"):
        sess = g[(g["mins"] >= 540) & (g["mins"] <= 915)].sort_index()
        if len(sess) < win + 5:
            continue
        tp = (sess["high"] + sess["low"] + sess["close"]) / 3.0
        vol = sess["volume"].fillna(0)
        cumvol = vol.cumsum()
        vwap = (tp * vol).cumsum() / cumvol.replace(0, np.nan)
        dev = sess["close"] - vwap
        sig = dev.rolling(win).std()
        entry = None; direction = 0; stop = None; sig_e = None
        bars = sess.reset_index(drop=True)
        vwap_r = vwap.reset_index(drop=True); sig_r = sig.reset_index(drop=True)
        for idx in range(len(bars)):
            b = bars.iloc[idx]
            if entry is None:
                if idx < win or not np.isfinite(sig_r.iloc[idx]) or sig_r.iloc[idx] <= 0:
                    continue
                if b["close"] < vwap_r.iloc[idx] - k * sig_r.iloc[idx]:
                    direction = 1
                elif b["close"] > vwap_r.iloc[idx] + k * sig_r.iloc[idx]:
                    direction = -1
                else:
                    continue
                if idx + 1 < len(bars):
                    entry = bars.iloc[idx + 1]["open"]
                else:
                    break
                sig_e = sig_r.iloc[idx]
                stop = entry - direction * 1.0 * sig_e
                continue
            # 持倉：VWAP 觸及出場、停損、收盤平倉
            vw = vwap_r.iloc[idx]
            if direction == 1:
                if b["high"] >= vw:
                    px = bars.iloc[idx + 1]["open"] if idx + 1 < len(bars) else b["close"]
                    trades.append({"dir": 1, "entry": entry, "exit": px, "sess": "day"}); entry = "done"; break
                if b["low"] <= stop:
                    trades.append({"dir": 1, "entry": entry, "exit": stop, "sess": "day"}); entry = "done"; break
            else:
                if b["low"] <= vw:
                    px = bars.iloc[idx + 1]["open"] if idx + 1 < len(bars) else b["close"]
                    trades.append({"dir": -1, "entry": entry, "exit": px, "sess": "day"}); entry = "done"; break
                if b["high"] >= stop:
                    trades.append({"dir": -1, "entry": entry, "exit": stop, "sess": "day"}); entry = "done"; break
        if entry is not None and entry != "done":
            trades.append({"dir": direction, "entry": entry, "exit": bars.iloc[-1]["close"], "sess": "day"})
    return trades

# ---------------- 指標 ----------------
def metrics(name, trades, bh_pts, series_label, window):
    tr = pd.DataFrame(trades)
    out = {"name": name, "series": series_label, "window": window, "n": len(tr),
           "bh_pts": round(float(bh_pts), 1)}
    if len(tr) == 0:
        out.update({"pass": False, "reason": "0 trades"})
        return out
    raw = tr["dir"] * (tr["exit"] - tr["entry"])
    res = {}
    for c in COSTS:
        p = raw - c
        pf = p[p > 0].sum() / max(1e-9, -p[p < 0].sum())
        half = len(p) // 2
        eq = p.cumsum(); dd = float((eq - eq.cummax()).min())
        streak = cur = 0
        for v in p:
            cur = cur + 1 if v < 0 else 0; streak = max(streak, cur)
        res[f"cost{int(c)}"] = {
            "win_pct": round(float((p > 0).mean() * 100), 1),
            "avg": round(float(p.mean()), 1), "total": round(float(p.sum()), 1),
            "pf": round(float(pf), 2), "half1_avg": round(float(p.iloc[:half].mean()), 1) if half else None,
            "half2_avg": round(float(p.iloc[half:].mean()), 1) if half else None,
            "maxdd": round(dd, 1), "max_consec_loss": int(streak),
            "long_n": int((tr["dir"] == 1).sum()),
            "long_avg": round(float(p[tr["dir"] == 1].mean()), 1) if (tr["dir"] == 1).any() else None,
            "short_n": int((tr["dir"] == -1).sum()),
            "short_avg": round(float(p[tr["dir"] == -1].mean()), 1) if (tr["dir"] == -1).any() else None,
        }
    # 排列檢定：符號翻轉（20 點成本），固定種子
    rng = np.random.default_rng(SEED)
    p20 = (raw - 20.0).values
    obs = p20.mean()
    flips = rng.choice([-1, 1], size=(2000, len(p20)))
    perm_means = (flips * p20).mean(axis=1)
    pval = float((perm_means >= obs).mean())
    out.update(res)
    out["perm_p_cost20"] = round(pval, 4)
    c20 = res["cost20"]
    passed = (len(tr) >= 30 and c20["avg"] > 0 and c20["pf"] >= 1.3
              and c20["half1_avg"] is not None and c20["half1_avg"] > 0 and c20["half2_avg"] > 0)
    out["pass"] = bool(passed)
    out["candidate_main"] = bool(passed and c20["pf"] >= 1.5)
    return out

def main():
    smoke = "--smoke" in sys.argv
    per = "60d" if smoke else "730d"  # 冒煙僅需足夠算 MA60 的輕量資料，不產出結論
    per15 = "5d" if smoke else "60d"
    results = []
    print("== 資料載入 ==")
    n225_60 = add_ma_atr(day_series(load("^N225", "60m", per)))
    nkdf_60 = load("NKD=F", "60m", per)
    nkdf_60_day = add_ma_atr(day_series(nkdf_60))
    nkdf_60_night = add_ma_atr(night_series(nkdf_60))
    nkdf_60_full = add_ma_atr(nkdf_60)
    nkdf_15 = load("NKD=F", "15m", per15)
    print(f"^N225 60m day bars={len(n225_60)} {n225_60.index[0].date()}~{n225_60.index[-1].date()}")
    print(f"NKD=F 60m day bars={len(nkdf_60_day)} night bars={len(nkdf_60_night)}")
    print(f"NKD=F 15m bars={len(nkdf_15)} {nkdf_15.index[0].date()}~{nkdf_15.index[-1].date()}")
    if smoke:
        checks = [
            ("MA", run_ma(n225_60, False, 2.0)),
            ("MA-night", run_ma(nkdf_60_night, False, 2.0)),
            ("VCP", run_vcp(nkdf_15)),
            ("VCP-filter", run_vcp(nkdf_15, regime=nkdf_60_full)),
            ("ORB", run_orb2(nkdf_15)),
            ("VWAP", run_vwap(nkdf_15)),
        ]
        for nm, tr in checks:
            print(f"  engine {nm}: {len(tr)} trades (smoke only)")
        print("SMOKE OK：引擎可執行，數字不具結論意義。")
        return

    def bh(d):
        return float(d["close"].iloc[-1] - d["close"].iloc[0])

    # MA 家族
    results.append(metrics("M0 MA交叉基準複現", run_ma(n225_60, False, 2.0), bh(n225_60), "^N225 60m 日盤", "730d"))
    results.append(metrics("M1a MA交叉停損1.0ATR", run_ma(n225_60, False, 1.0), bh(n225_60), "^N225 60m 日盤", "730d"))
    results.append(metrics("M1b MA交叉停損1.5ATR", run_ma(n225_60, False, 1.5), bh(n225_60), "^N225 60m 日盤", "730d"))
    results.append(metrics("M2-day NKD日盤交叉", run_ma(nkdf_60_day, False, 2.0), bh(nkdf_60_day), "NKD=F 60m 日盤", "730d"))
    results.append(metrics("M2-night NKD夜盤交叉", run_ma(nkdf_60_night, False, 2.0), bh(nkdf_60_night), "NKD=F 60m 夜盤", "730d"))
    results.append(metrics("M3-long N225只做多", run_ma(n225_60, True, 2.0), bh(n225_60), "^N225 60m 日盤", "730d"))
    t_day_long = run_ma(nkdf_60_day, True, 2.0)
    t_night_long = run_ma(nkdf_60_night, True, 2.0)
    results.append(metrics("M3-longN NKD日夜只做多", t_day_long + t_night_long,
                           bh(nkdf_60_day) + bh(nkdf_60_night), "NKD=F 60m 日夜聯集", "730d"))
    # VCP 家族
    results.append(metrics("V0 VCP-lite無濾網", run_vcp(nkdf_15), bh(nkdf_15), "NKD=F 15m 日夜", "60d"))
    results.append(metrics("V1 VCP-lite+60m regime濾網", run_vcp(nkdf_15, regime=nkdf_60_full),
                           bh(nkdf_15), "NKD=F 15m 日夜", "60d"))
    # 文獻策略
    results.append(metrics("O1 開盤區間突破ORB", run_orb2(nkdf_15), bh(nkdf_15), "NKD=F 15m 日盤", "60d"))
    results.append(metrics("W1 VWAP均值回歸", run_vwap(nkdf_15), bh(nkdf_15), "NKD=F 15m 日盤", "60d"))

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump({"spec": "nikkei_strategy_spec_2026-10-28 v1",
                   "stage": "現貨指數代理階段、尚未實盤",
                   "results": results}, f, ensure_ascii=False, indent=2)
    print("\n== 候選表（20 點成本判定） ==")
    print(f"{'組別':<28} {'n':>4} {'勝率':>6} {'平均':>8} {'PF':>6} {'前半':>8} {'後半':>8} {'回撤':>9} {'連虧':>5} {'p':>7} {'過關':>6}")
    for r in results:
        c = r.get("cost20")
        if c is None:
            print(f"{r['name']:<28} n=0 未過關"); continue
        print(f"{r['name']:<28} {r['n']:>4} {c['win_pct']:>5.1f}% {c['avg']:>+8.1f} {c['pf']:>6.2f} "
              f"{c['half1_avg']:>+8.1f} {c['half2_avg']:>+8.1f} {c['maxdd']:>9.1f} {c['max_consec_loss']:>5} "
              f"{r['perm_p_cost20']:>7.4f} {'PASS' if r['pass'] else 'FAIL':>6}")
    mains = [r["name"] for r in results if r.get("candidate_main")]
    print("\n主推候選（過關＋PF≥1.5）:", mains if mains else "無")

if __name__ == "__main__":
    main()
