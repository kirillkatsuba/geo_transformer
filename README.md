# geo_transformer

Experimental code for an assay-conditioned autoregressive Transformer for geochemical block-model filling.

The goal is not to replace the existing `microblock_v1/v2/DNN/GP` pipeline immediately. The first implementation treats those models as a baseline/trend and learns a spatial generative residual field:

```text
generated_field = baseline_field + generated_residual_field
block_values = block_operator(generated_field)
assay_values = assay_operator(generated_field)
```

The key modeling idea:

```text
p(y_i | x_i, geology_i, Au_i, c_knn_i, uncertainty_i,
       y_<i, assay_context, generation_order)
```

where `y_i = [AS, S, CORG-1, CA, FE]`.

## Main components

- `config.py` - dataclasses with model/training defaults.
- `model.py` - causal Transformer decoder with optional assay cross-attention.
- `operators.py` - sparse assay/block operators and consistency utilities.
- `ordering.py` - generation orders: strike/cross/Z, distance-to-data, random.
- `losses.py` - Gaussian NLL, Huber loss, assay/block consistency losses.
- `dataset.py` - sequence dataset for teacher-forced autoregressive training.
- `inference.py` - autoregressive generation with sampling and multiple orders.

## Expected data shape

The model is intentionally table-first. A minimal node table should contain one row per block, microblock, or quadrature node:

```text
node_id
X, Y, Z
coord_strike, coord_cross
Au_Final
geology/domain encoded features
baseline predictions for AS/S/CORG-1/CA/FE
uncertainty features
optional true targets for training
```

Assay and block operators are stored as long sparse tables:

```text
operator_id, node_id, weight
```

For assays, `operator_id` is an interval id. For blocks, `operator_id` is a block id.

## Customer datasets

The first implementation assumes three customer-provided sources:

```text
Вся_химия+литология+Au_final_all_data.XLSX
  Drillhole intervals from south/center with chemistry, Au, lithology and structural features.

md_nat250721_CEN_Отработано.csv
  Central/southern block model intersecting the drilled domain. Targets and Au are known.

md_nat241227(Модель_ресурсов_ind,inf).csv
  Northern extrapolation block model. Au is known, targets are missing or partially known.
```

Standardize them with:

```bash
python3 -m geo_transformer.prepare_customer_data --root /Users/kirill/hse_lambda/artem_2
```

This writes parquet tables under `geo_transformer/prepared/`.

## First intended experiment

1. Build node features on existing microblock/quadrature nodes.
2. Attach baseline predictions from existing models.
3. Build or approximate `assay_operator`.
4. Train with teacher forcing and causal masks.
5. Validate with spatial/drillhole masks, not only random splits.
6. Generate multiple realizations on the northern target zone.

## Train the first prototype

Prepare data:

```bash
python3 -m geo_transformer.prepare_customer_data --root /Users/kirill/hse_lambda/artem_2
```

Prepare data with KNN-projected drillhole chemistry features:

```bash
python3 -m geo_transformer.prepare_customer_data \
  --root /Users/kirill/hse_lambda/artem_2 \
  --output-dir /Users/kirill/hse_lambda/artem_2/geo_transformer/prepared_knn_chem \
  --add-knn-chemistry \
  --knn-neighbors 16 \
  --knn-power 2
```

Prepare data with KNN chemistry and existing model predictions as baseline
features/trends:

```bash
python3 -m geo_transformer.prepare_customer_data \
  --root /Users/kirill/hse_lambda/artem_2 \
  --output-dir /Users/kirill/hse_lambda/artem_2/geo_transformer/prepared_full \
  --add-knn-chemistry \
  --knn-neighbors 16 \
  --knn-power 2 \
  --baseline v1=bm_models/interolation_models/bm_v1.csv \
  --baseline v2=bm_models/interolation_models/bm_v2.csv \
  --baseline dnn=bm_models/interolation_models/bm_dnn.csv \
  --baseline gp=bm_models/interolation_models/bm_gp.csv \
  --baseline v1=bm_models/extrapolation_model/bm_v1.csv
```

The same `--baseline NAME=PATH` can be passed multiple times. Tables are
concatenated by name and joined to CEN/NTH blocks by `X/Y/Z`.

Run a small smoke training job:

```bash
python3 -m geo_transformer.train \
  --prepared-dir /Users/kirill/hse_lambda/artem_2/geo_transformer/prepared \
  --output-dir /Users/kirill/hse_lambda/artem_2/geo_transformer/runs/smoke \
  --epochs 1 \
  --max-sequences 8 \
  --sequence-length 256 \
  --d-model 64 \
  --n-heads 4 \
  --n-layers 2 \
  --device cpu
```

Run a larger first experiment:

```bash
python3 -m geo_transformer.train \
  --prepared-dir /Users/kirill/hse_lambda/artem_2/geo_transformer/prepared \
  --output-dir /Users/kirill/hse_lambda/artem_2/geo_transformer/runs/center_v1 \
  --epochs 10 \
  --sequence-length 512 \
  --batch-size 2 \
  --order domain_strike \
  --device auto
```

Multi-order training with scheduled sampling:

```bash
python3 -m geo_transformer.train \
  --prepared-dir geo_transformer/prepared_knn_chem \
  --output-dir geo_transformer/runs/knn_chem_multi_sched \
  --epochs 100 \
  --sequence-length 1024 \
  --batch-size 4 \
  --orders domain_strike,strike,random \
  --d-model 384 \
  --n-heads 8 \
  --n-layers 8 \
  --dropout 0.15 \
  --learning-rate 8e-5 \
  --scheduled-sampling-prob 0.15 \
  --context-dropout 0.05 \
  --val-mode y_high \
  --device cuda
```

Residual training on top of attached baseline predictions:

```bash
python3 -m geo_transformer.train \
  --prepared-dir geo_transformer/prepared_full \
  --output-dir geo_transformer/runs/full_multi_sched_residual \
  --epochs 100 \
  --sequence-length 1024 \
  --batch-size 4 \
  --orders domain_strike,strike,random \
  --d-model 384 \
  --n-heads 8 \
  --n-layers 8 \
  --dropout 0.15 \
  --learning-rate 8e-5 \
  --scheduled-sampling-prob 0.15 \
  --context-dropout 0.05 \
  --target-baseline mean_baselines \
  --val-mode y_high \
  --device cuda
```

`--target-baseline mean_baselines` averages available columns such as `v1_AS`,
`v2_AS`, `dnn_AS`, `gp_AS` and trains the Transformer to generate the residual
field. Without this flag those columns are used only as input features. The
trainer applies a scale guard: if a baseline target is not on the same scale as
the supervised target, it remains a feature but is excluded from the residual
trend. In the current files this is especially important for `AS`.

Outputs:

- `best_model.pt` - checkpoint with model weights, feature schema and target scaler.
- `metrics.csv` - train/validation loss by epoch.
- `run_config.json` - command configuration.

## Evaluate and visualize

Fast teacher-forced evaluation on center and known northern blocks:

```bash
python3 -m geo_transformer.evaluate \
  --prepared-dir geo_transformer/prepared \
  --checkpoint geo_transformer/runs/center_v1/best_model.pt \
  --output-dir geo_transformer/eval/center_v1 \
  --domain both \
  --mode teacher_forced \
  --device cuda
```

This writes, per domain:

- `predictions.csv` - true/pred/error per target.
- `metrics.csv` - MAE, RMSE, R2, MAPE and bias.
- `plots/xy_<TARGET>.png` - XY maps for true, prediction and error.

Autoregressive evaluation is slower but closer to real inference:

```bash
python3 -m geo_transformer.evaluate \
  --prepared-dir geo_transformer/prepared \
  --checkpoint geo_transformer/runs/center_v1/best_model.pt \
  --output-dir geo_transformer/eval/center_v1_ar \
  --domain north \
  --mode autoregressive \
  --sequence-length 256 \
  --max-sequences 20 \
  --device cuda
```

Autoregressive multi-order ensemble:

```bash
python3 -m geo_transformer.evaluate \
  --prepared-dir geo_transformer/prepared_full \
  --checkpoint geo_transformer/runs/full_multi_sched_residual/best_model.pt \
  --output-dir geo_transformer/eval/full_multi_sched_residual_ar_ens \
  --domain north \
  --mode autoregressive \
  --sequence-length 256 \
  --max-sequences 50 \
  --sample-sequences \
  --ensemble-orders domain_strike,strike,random \
  --device cuda
```

## Assay/block operators

When geometry intersections are available, build normalized sparse operator tables:

```bash
python3 -m geo_transformer.build_operator \
  --intersections assay_interval_node_intersections.csv \
  --output geo_transformer/prepared/assay_operator.csv \
  --operator-col interval_id \
  --node-col node_id \
  --measure-col intersection_length
```

For block aggregation:

```bash
python3 -m geo_transformer.build_operator \
  --intersections block_node_intersections.csv \
  --output geo_transformer/prepared/block_operator.csv \
  --operator-col block_id \
  --node-col node_id \
  --measure-col intersection_volume
```
