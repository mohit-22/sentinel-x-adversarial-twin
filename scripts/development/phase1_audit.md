# CURRENT DATA FLOW
1. **DataFrame Generation**: The exact attack transaction DataFrame for the final test is generated inside `arena.py`'s `re_test` function (using `generate_matched_population_attacks`, applying mutations, and engineering features).
2. **DataFrame Lifecycle**: It is discarded after `re_test` returns. Only aggregate metrics (`evasion_rate`, `total_fraud`, `false_negatives`, etc.) are returned to `run_arena_mvp_gate`.
3. **hard_examples_count**: Represents the number of mutated fraud rows that successfully passed domain validation in `harvest_hard_negatives` (harvested hard negatives used for retraining). It does not represent attacked rows or newly caught rows.
4. **M0 Predictions**: Computed inside `run_attack` for the initial attack batch, but not on the mutated final test set. They are not stored.
5. **Post-Hardening Predictions**: Computed inside `re_test` using the retrained model (M1), used to calculate `final_evasion_rate`. They are not stored.

# AVAILABLE REAL DATA
The actual attack amounts and prediction masks are briefly available in memory during the execution of `re_test` (inside `fraud_rows["amount"]` and `y_pred`).

# MISSING DATA
Since the DataFrames and prediction masks are not returned or cached, we cannot calculate the exact financial impact (`total_attack_value`, `m0_caught_value`, `post_hardening_caught_value`, `incremental_value_prevented`) directly from the already existing `_LATEST_ARENA_RUN` data in `endpoints.py`.

# CURRENT WRONG FORMULA
In `backend/app/api/endpoints.py`, `api_observatory_impact()`:
```python
hard_negatives = _LATEST_ARENA_RUN.hard_examples_count
avg_amount = float(_APP_STATE["clean_history"]["amount"].mean())
fraud_prevented_inr = hard_negatives * avg_amount
```
This is not scientifically defensible because it multiplies the count of training examples (hard negatives harvested) by the average clean transaction amount, which has no connection to the actual value of fraud caught by the retrained model.

# MINIMAL SAFE DESIGN
1. Create a `_LATEST_ARENA_IMPACT = {}` session-scoped cache in `arena.py` or `endpoints.py`.
2. Modify `run_arena_mvp_gate` to also compute M0 predictions on the same final test batch (`fraud_rows` used for M1) without changing the official metrics. We can return the calculated economic metrics from `re_test` (or compute them in `run_arena_mvp_gate` by passing `model_0` into `re_test`) and store them in `_LATEST_ARENA_IMPACT` keyed by `run_id`.
3. Update `/api/v1/observatory/impact` to read from this cache.
