# Nikkei Strategy Sim

日經策略回測（研究與模擬專用，不下單、不含券商 API、不含任何憑證與 Telegram 程式碼）。

## 凍結規格
見 `nikkei_backtest.py` 開頭註解所指的規格檔（2026-10-28 凍結版 v1）。
先凍結再跑；任何規則修改須另開規格 v2 並重跑全部，不得覆寫 v1 結論。

## 執行方式
GitHub Actions → 本 workflow → Run workflow（`workflow_dispatch`）。
結果以 artifact `nikkei-backtest`（`results.json`）回傳，並印在 Actions log 的候選表。

## 階段聲明
**現貨指數代理階段、尚未實盤。**
資料為 Yahoo `^N225` 現貨指數與 `NKD=F`（CME 日經期貨，美元計價，與 JPX 有基差）；
目標商品為大阪 JPX 微日經，正式驗證須用真實期貨分鐘資料後才談實盤。

## 候選清單（13 組）
MA 交叉：M0 基準複現、M1a／M1b 停損敏感度、M2-day／M2-night 日夜分開、M3-long／M3-longN 只做多。
VCP-lite：V0 無濾網複現、V1＋60 分 regime 濾網（15 分 K，60d 窗口限制須註明）。
文獻策略：O1 開盤區間突破 ORB、W1 VWAP 均值回歸（教科書通用定義，不引用績效宣稱）。

## 過關標準（20 點成本判定）
n ≥ 30、每筆平均 > 0、PF ≥ 1.3、前半與後半平均皆正，並報告回撤、連虧、多空分開、B&H、排列檢定 p 值。
主推另須 PF ≥ 1.5；無組別同時滿足時直說沒有主推。
