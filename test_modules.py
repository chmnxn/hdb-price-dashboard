"""
Unit tests for modules.py — HDB Resale Price Prediction Pipeline
Run with: pytest test_modules.py -v
"""

import io
import numpy as np
import pandas as pd
import pytest

from modules import (
    load_and_clean,
    prepare_xy,
    temporal_split,
    log_transform,
    evaluate,
    predict,
    merge_train_val,
    validate_train_columns,
    validate_predict_columns,
    validate_batch_columns,
    REQUIRED_TRAIN_COLS,
    REQUIRED_PREDICT_COLS,
    COLS_TO_DROP,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_row(**overrides) -> dict:
    """Return a minimal valid row matching REQUIRED_PREDICT_COLS."""
    base = {
        'month':                         '2023-01',
        'lease_commence_date':           1990,
        'remaining_lease_years':         70,
        'remaining_lease_months':        0,
        'storey_range':                  '07 TO 09',
        'town':                          'ANG MO KIO',
        'flat_model':                    'Model A',
        'planning_area_ura':             'Ang Mo Kio',
        'closest_mrt_station':           'ANG MO KIO',
        'closest_pri_school':            'ANG MO KIO PRIMARY SCHOOL',
        'flat_type':                     '4 ROOM',
        'region_ura':                    'North Region',
        'transport_type':                'MRT',
        'line_color':                    'Red',
        'floor_area_sqft':               1000,
        'latitude':                      1.37,
        'longitude':                     103.85,
        'distance_to_mrt_meters':        400,
        'distance_to_cbd':               8000,
        'distance_to_pri_school_meters': 300,
    }
    base.update(overrides)
    return base


def _make_df(n=10, shuffled=False, with_price=True, with_duplicates=False) -> pd.DataFrame:
    """Build a minimal valid DataFrame for pipeline testing."""
    months = pd.date_range('2020-01', periods=n, freq='MS').strftime('%Y-%m').tolist()
    rows = [_make_row(month=m) for m in months]
    if with_price:
        for i, r in enumerate(rows):
            r['resale_price'] = 300_000 + i * 10_000
    if with_duplicates:
        rows.append(rows[0].copy())
    df = pd.DataFrame(rows)
    if shuffled:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# load_and_clean
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadAndClean:

    def test_accepts_filepath(self, tmp_path):
        df = _make_df()
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        result = load_and_clean(str(p))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_accepts_bytesio(self):
        df = _make_df()
        buf = io.BytesIO(df.to_csv(index=False).encode())
        result = load_and_clean(buf)
        assert isinstance(result, pd.DataFrame)

    def test_drops_unnamed_index_column(self, tmp_path):
        df = _make_df()
        p = tmp_path / "data.csv"
        df.to_csv(p, index=True)   # index=True produces 'Unnamed: 0'
        result = load_and_clean(str(p))
        assert 'Unnamed: 0' not in result.columns

    def test_removes_duplicate_rows(self, tmp_path):
        df = _make_df(n=5, with_duplicates=True)
        assert len(df) == 6
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        result = load_and_clean(str(p))
        assert len(result) == 5


# ══════════════════════════════════════════════════════════════════════════════
# prepare_xy
# ══════════════════════════════════════════════════════════════════════════════

class TestPrepareXY:

    def test_separates_target(self):
        df = _make_df()
        X, y = prepare_xy(df)
        assert 'resale_price' not in X.columns
        assert y.name == 'resale_price'
        assert len(X) == len(y)

    def test_drops_leakage_columns(self):
        df = _make_df()
        # Add leakage columns that should be dropped
        df['price_per_sqft'] = df['resale_price'] / 1000
        df['floor_area_sqm'] = 100
        X, _ = prepare_xy(df)
        for col in COLS_TO_DROP:
            assert col not in X.columns

    def test_missing_drop_cols_are_ignored(self):
        """prepare_xy should not crash if COLS_TO_DROP columns are absent."""
        df = _make_df()
        X, y = prepare_xy(df)   # none of COLS_TO_DROP are in _make_df
        assert len(X) == len(df)


# ══════════════════════════════════════════════════════════════════════════════
# temporal_split
# ══════════════════════════════════════════════════════════════════════════════

class TestTemporalSplit:

    def _split(self, n=100, shuffled=True):
        df = _make_df(n=n, shuffled=shuffled)
        X, y = prepare_xy(df)
        return temporal_split(X, y)

    def test_returns_six_parts(self):
        result = self._split()
        assert len(result) == 6

    def test_sizes_sum_to_total(self):
        n = 100
        X_tr, X_val, X_te, y_tr, y_val, y_te = self._split(n=n)
        assert len(X_tr) + len(X_val) + len(X_te) == n
        assert len(y_tr) + len(y_val) + len(y_te) == n

    def test_default_ratio_70_15_15(self):
        n = 100
        X_tr, X_val, X_te, *_ = self._split(n=n)
        assert len(X_tr) == 70
        assert len(X_val) == 15
        assert len(X_te) == 15

    def test_split_is_chronological_not_random(self):
        """Train months must all be earlier than val months,
        which must all be earlier than test months."""
        X_tr, X_val, X_te, *_ = self._split(n=60, shuffled=True)
        train_max = pd.to_datetime(X_tr['month']).max()
        val_min   = pd.to_datetime(X_val['month']).min()
        val_max   = pd.to_datetime(X_val['month']).max()
        test_min  = pd.to_datetime(X_te['month']).min()
        assert train_max <= val_min, "Train bleeds into val (not chronological)"
        assert val_max   <= test_min, "Val bleeds into test (not chronological)"

    def test_x_and_y_indices_are_aligned(self):
        X_tr, X_val, X_te, y_tr, y_val, y_te = self._split()
        assert list(X_tr.index)  == list(y_tr.index)
        assert list(X_val.index) == list(y_val.index)
        assert list(X_te.index)  == list(y_te.index)


# ══════════════════════════════════════════════════════════════════════════════
# log_transform
# ══════════════════════════════════════════════════════════════════════════════

class TestLogTransform:

    def test_single_array(self):
        a = np.array([1.0, np.e, np.e**2])
        result = log_transform(a)
        np.testing.assert_allclose(result, [0.0, 1.0, 2.0])

    def test_multiple_arrays_returns_tuple(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = log_transform(a, b)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_inverse_is_exp(self):
        """log_transform followed by np.exp should recover original values."""
        original = np.array([100_000.0, 350_000.0, 800_000.0])
        recovered = np.exp(log_transform(original))
        np.testing.assert_allclose(recovered, original, rtol=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# evaluate
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluate:

    def test_perfect_predictions(self):
        y = np.array([100_000.0, 200_000.0, 300_000.0])
        m = evaluate(y, y, n_features=5)
        assert m['RMSE']   == pytest.approx(0.0)
        assert m['MAE']    == pytest.approx(0.0)
        assert m['R2']     == pytest.approx(1.0)

    def test_returns_all_keys(self):
        # n=20, n_features=2 → denominator = 20 - 2 - 1 = 17, safe
        rng  = np.random.default_rng(0)
        y    = rng.uniform(100_000, 800_000, 20)
        pred = y + rng.normal(0, 5_000, 20)
        m = evaluate(y, pred, n_features=2)
        assert set(m.keys()) == {'RMSE', 'MAE', 'R2', 'Adj_R2'}

    def test_rmse_is_always_non_negative(self):
        y    = np.random.default_rng(0).uniform(100_000, 800_000, 50)
        pred = y + np.random.default_rng(1).normal(0, 10_000, 50)
        m = evaluate(y, pred, n_features=10)
        assert m['RMSE'] >= 0

    def test_adj_r2_penalises_more_features(self):
        y    = np.random.default_rng(0).uniform(100_000, 800_000, 100)
        pred = y + np.random.default_rng(1).normal(0, 10_000, 100)
        m_few  = evaluate(y, pred, n_features=2)
        m_many = evaluate(y, pred, n_features=50)
        assert m_few['Adj_R2'] > m_many['Adj_R2']

    def test_known_mae_value(self):
        # n=10, n_features=1 → denominator = 10 - 1 - 1 = 8, safe
        # All errors are exactly 10_000, so MAE = 10_000
        y    = np.array([100_000.0, 200_000.0, 300_000.0, 400_000.0, 500_000.0,
                         600_000.0, 700_000.0, 800_000.0, 900_000.0, 1_000_000.0])
        pred = y + 10_000.0
        m = evaluate(y, pred, n_features=1)
        assert m['MAE'] == pytest.approx(10_000.0)


# ══════════════════════════════════════════════════════════════════════════════
# predict
# ══════════════════════════════════════════════════════════════════════════════

class TestPredict:

    class _DummyModel:
        """Stub model that returns log-scale values directly."""
        def __init__(self, log_values):
            self._vals = np.array(log_values)
        def predict(self, X):
            return self._vals

    def test_inverse_log_transform(self):
        log_prices = np.log([300_000.0, 500_000.0, 800_000.0])
        model = self._DummyModel(log_prices)
        result = predict(model, X=None)
        np.testing.assert_allclose(result, [300_000.0, 500_000.0, 800_000.0], rtol=1e-6)

    def test_no_negative_predictions(self):
        """clip(0) must ensure all predictions are non-negative."""
        # Force a very negative log value to simulate edge case
        model = self._DummyModel([-1000.0, np.log(500_000)])
        result = predict(model, X=None)
        assert np.all(result >= 0)

    def test_output_shape_matches_input(self):
        n = 20
        model = self._DummyModel(np.log(np.full(n, 400_000.0)))
        result = predict(model, X=np.zeros((n, 5)))
        assert result.shape == (n,)


# ══════════════════════════════════════════════════════════════════════════════
# merge_train_val
# ══════════════════════════════════════════════════════════════════════════════

class TestMergeTrainVal:

    def test_concatenates_correctly(self):
        X_tr  = np.ones((70, 5))
        X_val = np.ones((15, 5)) * 2
        y_tr  = np.zeros(70)
        y_val = np.ones(15)
        X_out, y_out = merge_train_val(X_tr, X_val, y_tr, y_val)
        assert X_out.shape == (85, 5)
        assert y_out.shape == (85,)

    def test_train_comes_before_val(self):
        X_tr  = np.array([[1], [2]])
        X_val = np.array([[3], [4]])
        y_tr  = np.array([10.0, 20.0])
        y_val = np.array([30.0, 40.0])
        X_out, y_out = merge_train_val(X_tr, X_val, y_tr, y_val)
        np.testing.assert_array_equal(X_out[:2], X_tr)
        np.testing.assert_array_equal(X_out[2:], X_val)
        np.testing.assert_array_equal(y_out[:2], y_tr)
        np.testing.assert_array_equal(y_out[2:], y_val)


# ══════════════════════════════════════════════════════════════════════════════
# Validation functions
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateTrainColumns:

    def test_valid_df_returns_no_errors(self):
        df = _make_df(with_price=True)
        assert validate_train_columns(df) == []

    def test_missing_column_detected(self):
        df = _make_df(with_price=True).drop(columns=['town'])
        errors = validate_train_columns(df)
        assert len(errors) == 1
        assert 'town' in errors[0]

    def test_missing_target_detected(self):
        df = _make_df(with_price=False)
        errors = validate_train_columns(df)
        assert any('resale_price' in e for e in errors)

    def test_out_of_range_numeric_detected(self):
        df = _make_df(with_price=True)
        df.loc[0, 'latitude'] = 99.0   # clearly outside Singapore
        errors = validate_train_columns(df)
        assert any('latitude' in e for e in errors)

    def test_string_numeric_column_handled(self):
        """Mixed-type numeric columns (str + int) must not crash."""
        df = _make_df(with_price=True)
        df['floor_area_sqft'] = df['floor_area_sqft'].astype(str)
        errors = validate_train_columns(df)   # must not raise
        assert isinstance(errors, list)


class TestValidatePredictColumns:

    def test_valid_df_returns_no_errors(self):
        df = pd.DataFrame([_make_row()])
        assert validate_predict_columns(df) == []

    def test_missing_column_detected(self):
        df = pd.DataFrame([_make_row()])
        df = df.drop(columns=['floor_area_sqft'])
        errors = validate_predict_columns(df)
        assert any('floor_area_sqft' in e for e in errors)

    def test_out_of_range_distance_detected(self):
        df = pd.DataFrame([_make_row(distance_to_mrt_meters=99999)])
        errors = validate_predict_columns(df)
        assert any('distance_to_mrt_meters' in e for e in errors)


class TestValidateBatchColumns:

    def test_rejects_file_with_resale_price(self):
        df = _make_df(with_price=True)   # has resale_price
        errors = validate_batch_columns(df)
        assert any('resale_price' in e for e in errors)

    def test_accepts_file_without_resale_price(self):
        df = pd.DataFrame([_make_row()])   # no resale_price
        assert validate_batch_columns(df) == []

    def test_still_catches_missing_columns(self):
        df = pd.DataFrame([_make_row()])
        df = df.drop(columns=['town', 'flat_type'])
        errors = validate_batch_columns(df)
        assert len(errors) > 0