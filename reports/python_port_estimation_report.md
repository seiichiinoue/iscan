# Python再実装に向けた実行時間見積もり報告（別リポジトリ向け）

## 1. 目的
- 現行C++実装を参照基準として、別リポジトリで開始するPython再実装の実行時間を事前に見積もる。
- 見積もり時点で必要な計測項目、比較条件、受け入れ基準を固定し、実装後の評価ぶれを防ぐ。

## 2. 見積もり前提

### 2.1 計算支配項（C++実装から）
`src/model.hpp` のサンプリング実装より、1 iteration の主要負荷は次の3点。

1. `sample_z(t)`  
   - 各文書に対してトピック候補を走査し、文書内単語の尤度を積算  
   - 目安: `O(D * K * L)`
2. `sample_psi(t)`  
   - `(t, k)` ごとの語彙カウント再集計と語彙更新  
   - 目安: `O(T * D * L + T * K * V)`
3. `_update_logistic_Psi()` の全量再計算  
   - 目安: `O(T * K * V)`  
   - 内側ループ中に繰り返し呼ばれると全体時間を押し上げる

ここで `D`: 文書数, `T`: 時間ビン数, `K`: トピック数, `V`: 語彙数, `L`: 平均文書長。

### 2.2 見積もりデータ
- データ生成は `tools/create_pseudo_data.py` を基準に統一。
- 比較の再現性のため、同一seed・同一生成ロジックで `small/medium/large` を作る。
- 3段階は「生成ロジックを固定したまま、パラメータだけ拡大」する。

## 3. 擬似データ設定（推奨プリセット）

`tools/create_pseudo_data.py` の引数仕様に合わせて、見積もり時に次を固定する。

| preset | num_times (T) | num_senses (K) | context_window_size (L) | vocab_size (V) | num_sample (total snippets) |
|---|---:|---:|---:|---:|---:|
| small | 10 | 8 | 20 | 4,000 | 20,000 |
| medium | 20 | 16 | 30 | 16,000 | 100,000 |
| large | 40 | 32 | 40 | 64,000 | 400,000 |

固定推奨:
- `shift_type=random`
- `word_prior_type=zipf`
- `ratio_common_vocab=0.1`
- seed固定（スクリプト既定は `np.random.seed(2)`。別リポジトリでも同等に固定）

補足:
- `num_sample` はスクリプト内部で `num_times` で按分されるため、実効的な1時点あたり文書数も併記して管理する。

## 4. 計測項目

### 4.1 C++基準計測
各 preset で以下を収集:
- `t_iter_cpp` (1 iteration総時間)
- `t_z_cpp`, `t_phi_cpp`, `t_psi_cpp`, `t_ll_cpp`
- `tokens_per_sec_cpp`
- メモリ利用量（可能ならRSS）

### 4.2 Python候補比較
比較実装:
1. 純Python（基準）
2. NumPyベクトル化
3. Numba JIT

比較対象:
- `sample_z` 相当カーネル
- `sample_psi` 相当カーネル
- `logistic_Psi` 更新カーネル

収集値:
- `t_kernel_py`
- `speed_ratio_x = t_kernel_py / t_kernel_cpp`

## 5. 実行時間予測式

反復時間の推定:

`t_iter_pred = t_z_pred + t_phi_pred + t_psi_pred + t_ll_pred`

各項:

`t_x_pred = t_x_cpp * speed_ratio_x`

補正式（必要時）:
- `logistic_Psi` 再計算回数が設計で変わる場合は、`n_recompute_factor` を掛ける
- 例: `t_psi_pred = t_psi_cpp * speed_ratio_psi * n_recompute_factor`

## 6. 推奨アーキテクチャ（別リポジトリ実装）

### 6.1 モジュール分割
1. `data/`  
   - pseudo_data生成取り込み、時系列binning、語彙ID化
2. `state/`  
   - `ModelState(Z, Phi, Psi, kappa_*)` をNumPy配列で保持
3. `kernels/`  
   - `sample_z`, `sample_phi`, `sample_psi`, logistic変換
4. `runner/`  
   - iteration制御、ロギング、seed管理、メトリクス出力
5. `bench/`  
   - C++比較用ベンチ、preset実行、CSV出力

### 6.2 高速化境界
- 純Python3重ループは禁止（特に `D*K*L` と `T*K*V`）。
- `sample_z`/`sample_psi`/`logistic_Psi` は最初から NumPy/Numba 前提。
- 配列は連続領域（C-order）を維持し、不要コピーを避ける。
- 再計算は全量ではなく差分更新を基本戦略にする。

## 7. 受け入れ基準（見積もり段階）
次を満たした場合、実装着手可と判断:

1. **再現性**
   - 同一preset・同一seedでC++/Python比較が再実行可能。
2. **性能予測の妥当性**
   - `small/medium/large` の `t_iter_pred` が単調に増加し、説明可能なスケーリングを示す。
3. **運用可能性**
   - targetデータ相当（通常は `medium` 以上）で、1 iteration 時間が研究運用可能な範囲にある。
4. **ボトルネック可視化**
   - `t_z`, `t_psi`, `t_logistic` の比率が明示され、次の最適化優先度が決まる。

## 8. 主要リスク
- Polya-Gamma周りの実装差で挙動差が出る。
  - 対策: 最初はAPI境界を固定し、既存実装/外部ライブラリを差し替え可能にする。
- メモリ帯域律速でNumba化しても伸びない。
  - 対策: dtype固定、in-place更新、アクセス局所性を優先。
- `T*K*V` 再計算回数が多すぎる。
  - 対策: キャッシュ無効化範囲を明示し、差分更新設計を先に確定。

## 9. 別リポジトリへの引き継ぎ事項
1. pseudo_dataプリセット（small/medium/large）をコード化。
2. C++基準計測フォーマット（CSV列）を固定。
3. Pythonカーネルベンチを `sample_z` / `sample_psi` / `logistic_Psi` で先行作成。
4. 見積もり式で `t_iter_pred` を算出し、採否を判定。
5. 採用後に本体実装へ進む。
