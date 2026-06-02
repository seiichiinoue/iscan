# 別リポジトリ移管チェックリスト（Python再実装）

## A. 初期セットアップ
- [ ] Python 3.11+ 環境を作成
- [ ] 依存導入: `numpy`, `numba`, `pandas`, `pyyaml`（必要に応じて `scipy`）
- [ ] 乱数seedポリシーを決定（例: `seed=2` 固定）
- [ ] ベンチ結果の出力先（`bench/results/*.csv`）を作成

## B. 疑似データ生成の標準化
- [ ] `tools/create_pseudo_data.py` 互換の生成ロジックを移植または呼び出し
- [ ] `small/medium/large` の3プリセットをコード化
- [ ] 生成メタデータ（T, K, V, L, D, seed）をJSON保存

## C. C++比較のための基準取得
- [ ] C++参照実装で `t_iter_cpp`, `t_z_cpp`, `t_phi_cpp`, `t_psi_cpp`, `t_ll_cpp` を取得
- [ ] `tokens_per_sec_cpp` を取得
- [ ] 全presetで同一フォーマットCSV出力

## D. Pythonカーネルベンチ
- [ ] `sample_z` カーネル（pure Python / NumPy / Numba）
- [ ] `sample_psi` カーネル（pure Python / NumPy / Numba）
- [ ] `logistic_Psi` 更新カーネル（NumPy / Numba）
- [ ] `speed_ratio = t_kernel_py / t_kernel_cpp` を算出

## E. 予測とGo/No-Go判定
- [ ] `t_iter_pred = Σ (t_x_cpp * speed_ratio_x)` を算出
- [ ] `small/medium/large` でスケーリング整合性を確認
- [ ] 研究運用上の閾値と比較して判定
- [ ] 判定結果と次アクションを記録

## F. 実装着手順（Goの場合）
1. `ModelState` を定義（`Z`, `Phi`, `Psi`, `kappa_*`）。
2. 学習ループ骨格を実装。
3. 高負荷カーネルをNumPy/Numbaで実装。
4. 小規模同値性検証を実施。
5. 中〜大規模で性能評価し、追加最適化を実施。

## 推奨成果物
- `bench_summary.md`（結果サマリ）
- `bench_raw.csv`（生計測）
- `estimation_sheet.csv`（予測値算出表）
- `decision_log.md`（Go/No-Go理由）
