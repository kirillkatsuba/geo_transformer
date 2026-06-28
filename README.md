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

Outputs:

- `best_model.pt` - checkpoint with model weights, feature schema and target scaler.
- `metrics.csv` - train/validation loss by epoch.
- `run_config.json` - command configuration.
