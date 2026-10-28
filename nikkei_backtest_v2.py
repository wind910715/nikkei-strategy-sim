# -*- coding: utf-8 -*-
"""日經策略回測第二輪 — 凍結規格 v2（見 goals/nikkei-futures-strategy-and-live-monitoring/files/nikkei_strategy_spec_v2_2026-10-28.md）
任務：使用者問「看勝率能不能拉到五成」。本輪不是對目標調參：五組變體一次凍結，勝率 ≥50% 只是五道門檻之一，
拉勝率若把期望值或 PF 換掉，一律判失敗。對照組 C0 = 第一輪主推 M3-long 原樣重跑。
全程現貨指數代理，結果一律標「現貨指數代理階段、尚未實盤」。
無 Telegram、無憑證、無下單。執行：python nikkei_backtest_v2.py（完整）/ --smoke（僅程式碼冒煙檢查，不產出結論）。
"""
import json, sys
import numpy as np, pandas as pd, yfinance as yf

SEED = 42
COSTS = (10.0, 20.0)

# ---------------- 資料（與 v1 相同） ----------------
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
    return d[(m >= 540) & (m < 930)]  # 東京 09:00–15:30 開盤的 K 棒（v1 程式定義）

# ---------------- MA 交叉引擎 v2（v1 邏輯＋停利帽＋MA60 濾網） ----------------
def run_ma_v2(df, hard_atr=2.0, tp_atr=None, regime_filter=False):
    """只做多。訊號：前根 ma5 ≤ 前根 ma20 且當根 ma5 > 當根 ma20 → 次根開盤做多。
    停損 = 進場 − hard_atr × 訊號當根 ATR；停利帽（若有）= 進場 ＋ tp_atr × 訊號當根 ATR（進場時凍結）。
    出場優先順序：停損（當根低點 ≤ 停損價 → 停損價）＞ 停利（當根高點 ≥ 停利價 → 停利價）＞
    死叉（當根 ma5 < 當根 ma20 → 次根開盤）。濾網：訊號當根收盤須 > 當根 ma60，否則訊號作廢（不遞延）。"""
    trades = []
    pos = None  # (entry, stop, cap)
    n = len(df)
    for i in range(1, n - 1):
        r, nx = df.iloc[i], df.iloc[i + 1]
        if pos is not None:
            e_, s_, cap_ = pos
            hit_stop = r["low"] <= s_
            hit_cap = cap_ is not None and r["high"] >= cap_
            hit_cross = r["ma5"] < r["ma20"]
            exit_now = hit_stop or hit_cap or hit_cross
            if exit_now or i == n - 2:
                if hit_stop:
                    px = s_
                elif hit_cap:
                    px = cap_
                else:
                    px = nx["open"]
                trades.append({"dir": 1, "entry": e_, "exit": px})
                pos = None
            continue
        atr = r["atr"]
        prev = df.iloc[i - 1]
        if prev["ma5"] <= prev["ma20"] and r["ma5"] > r["ma20"]:
            if regime_filter and not (r["close"] > r["ma60"]):
                continue  # 訊號作廢，不進場
            e = nx["open"]
            stop = e - hard_atr * atr
            cap = e + tp_atr * atr if tp_atr is not None else None
            pos = (e, stop, cap)
    if pos is not None:  # 序列尾強制平倉
        e_, s_, cap_ = pos
        trades.append({"dir": 1, "entry": e_, "exit": df.iloc[-1]["close"]})
    return trades

# ---------------- 指標（v2 過關邏輯） ----------------
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
    # v2 過關：五道門檻同時滿足（20 點成本）
    c20 = res["cost20"]
    passed = (len(tr) >= 30 and c20["win_pct"] >= 50.0 and c20["avg"] > 0 and c20["pf"] >= 1.3
              and c20["half1_avg"] is not None and c20["half1_avg"] > 0 and c20["half2_avg"] > 0)
    out["pass"] = bool(passed)
    return out

def main():
    smoke = "--smoke" in sys.argv
    per = "60d" if smoke else "730d"  # 冒煙僅需足夠算 MA60 的輕量資料，不產出結論
    print("== 資料載入 ==")
    n225_60 = add_ma_atr(day_series(load("^N225", "60m", per)))
    print(f"^N225 60m day bars={len(n225_60)} {n225_60.index[0].date()}~{n225_60.index[-1].date()}")
    if smoke:
        checks = [
            ("C0-control", run_ma_v2(n225_60)),
            ("R1-cap1.0", run_ma_v2(n225_60, tp_atr=1.0)),
            ("R4-filter", run_ma_v2(n225_60, regime_filter=True)),
            ("R5-cap1.5+filter", run_ma_v2(n225_60, tp_atr=1.5, regime_filter=True)),
        ]
        for nm, tr in checks:
            print(f"  engine {nm}: {len(tr)} trades (smoke only)")
        print("SMOKE OK：引擎可執行（含停利帽與濾網路徑），數字不具結論意義。")
        return

    def bh(d):
        return float(d["close"].iloc[-1] - d["close"].iloc[0])

    results = []
    # 對照組 C0：M3-long 原樣重跑
    results.append(metrics("C0 M3-long對照組", run_ma_v2(n225_60), bh(n225_60), "^N225 60m 日盤", "730d"))
    # 五組變體（一個一個改，不多改）
    results.append(metrics("R1 停利帽+1.0ATR", run_ma_v2(n225_60, tp_atr=1.0), bh(n225_60), "^N225 60m 日盤", "730d"))
    results.append(metrics("R2 停利帽+1.5ATR", run_ma_v2(n225_60, tp_atr=1.5), bh(n225_60), "^N225 60m 日盤", "730d"))
    results.append(metrics("R3 停損放寬3.0ATR", run_ma_v2(n225_60, hard_atr=3.0), bh(n225_60), "^N225 60m 日盤", "730d"))
    results.append(metrics("R4 MA60濾網", run_ma_v2(n225_60, regime_filter=True), bh(n225_60), "^N225 60m 日盤", "730d"))
    results.append(metrics("R5 停利1.5ATR＋MA60濾網", run_ma_v2(n225_60, tp_atr=1.5, regime_filter=True),
                           bh(n225_60), "^N225 60m 日盤", "730d"))

    # Parity 檢查：C0 對照 v1（n=162、勝率 38.9%、20 點成本平均 +125.7、PF 1.52）
    c0 = results[0].get("cost20", {})
    parity_ok = True
    if c0:
        if abs(results[0]["n"] - 162) / 162 > 0.10 or abs(c0["avg"] - 125.7) / 125.7 > 0.10:
            parity_ok = False
    print(f"\nParity 檢查（C0 vs v1 M3-long）：{'OK' if parity_ok else 'PARITY BREAK，先修程式再談結論'}")

    # 主推：通過組中 PF 最高者（同分取平均最高者），最多一個
    passing = [r for r in results[1:] if r.get("pass")]
    main_pick = None
    if passing:
        main_pick = max(passing, key=lambda r: (r["cost20"]["pf"], r["cost20"]["avg"]))["name"]
    for r in results:
        r["candidate_main"] = bool(r["name"] == main_pick)

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump({"spec": "nikkei_strategy_spec_v2_2026-10-28",
                   "stage": "現貨指數代理階段、尚未實盤",
                   "parity_ok": parity_ok,
                   "main_pick": main_pick,
                   "results": results}, f, ensure_ascii=False, indent=2)
    print("\n== 候選表（20 點成本判定，v2 五道門檻） ==")
    print(f"{'組別':<28} {'n':>4} {'勝率':>6} {'平均':>8} {'PF':>6} {'前半':>8} {'後半':>8} {'回撤':>9} {'連虧':>5} {'p':>7} {'過關':>6}")
    for r in results:
        c = r.get("cost20")
        if c is None:
            print(f"{r['name']:<28} n=0 未過關"); continue
        print(f"{r['name']:<28} {r['n']:>4} {c['win_pct']:>5.1f}% {c['avg']:>+8.1f} {c['pf']:>6.2f} "
              f"{c['half1_avg']:>+8.1f} {c['half2_avg']:>+8.1f} {c['maxdd']:>9.1f} {c['max_consec_loss']:>5} "
              f"{r['perm_p_cost20']:>7.4f} {'PASS' if r['pass'] else 'FAIL':>6}")
    print("\n主推（通過組中 PF 最高者，最多一個）:", main_pick if main_pick else "無")

if __name__ == "__main__":
    main()
