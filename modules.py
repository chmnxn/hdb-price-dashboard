import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from custom import TemporalFeatureExtractor, LeaseFeatureCreator, CVTargetEncoderWrapper

# ── Known categorical values ──────────────────────────────────────────────────
# Used for UI dropdowns and validation when no training data is loaded yet.

STOREY_ORDER = [
    '01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15',
    '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30',
    '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45',
    '46 TO 48', '49 TO 51', '52 TO 54', '55 TO 57', '58 TO 60',
]

FLAT_TYPES = [
    '1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM',
    'EXECUTIVE', 'MULTI-GENERATION',
]

REGION_URA = ['Central Region', 'East Region', 'West Region', 'North Region', 'North-East Region']

TRANSPORT_TYPES = ['MRT', 'LRT']

LINE_COLORS = ['Red', 'Green', 'Purple', 'Orange', 'Brown', 'Blue', 'Grey', 'Yellow']

# ── Column definitions ────────────────────────────────────────────────────────

COLS_TO_DROP = [
    'price_per_sqft',   # target leakage
    'floor_area_sqm',   # redundant with floor_area_sqft
    'blk_no',
    'road_name',
    'building',
    'postal',
    'x',                # redundant with lat/lng
    'y',
]

TEMPORAL_COL         = ['month', 'lease_commence_date']
LEASE_COL            = ['remaining_lease_years', 'remaining_lease_months']
ORDINAL_COL          = ['storey_range']
HIGH_CARDINALITY_COL = ['town', 'flat_model', 'planning_area_ura',
                         'closest_mrt_station', 'closest_pri_school']
LOW_CARDINALITY_COL  = ['flat_type', 'region_ura', 'transport_type', 'line_color']
NUMERICAL_COL        = ['floor_area_sqft', 'latitude', 'longitude',
                         'distance_to_mrt_meters', 'distance_to_cbd',
                         'distance_to_pri_school_meters']

# Columns the training CSV must have (before dropping)
REQUIRED_TRAIN_COLS = (
    TEMPORAL_COL + LEASE_COL + ORDINAL_COL +
    HIGH_CARDINALITY_COL + LOW_CARDINALITY_COL + NUMERICAL_COL +
    ['resale_price']
)

# Columns required for prediction (same as training minus the target)
REQUIRED_PREDICT_COLS = (
    TEMPORAL_COL + LEASE_COL + ORDINAL_COL +
    HIGH_CARDINALITY_COL + LOW_CARDINALITY_COL + NUMERICAL_COL
)

# Numeric bounds for sanity checks — (min, max)
NUMERIC_BOUNDS = {
    'floor_area_sqft':              (200,   50000),
    'remaining_lease_years':        (0,     99),
    'remaining_lease_months':       (0,     11),
    'lease_commence_date':          (1960,  2025),
    'latitude':                     (1.20,  1.50),
    'longitude':                    (103.60, 104.00),
    'distance_to_mrt_meters':       (0,     10000),
    'distance_to_cbd':              (0,     35000),
    'distance_to_pri_school_meters':(0,     10000),
}

# ── Validation ────────────────────────────────────────────────────────────────

def _check_columns(df: pd.DataFrame, required: list) -> list:
    """Return list of error strings for missing columns."""
    missing = sorted(set(required) - set(df.columns))
    if missing:
        return [f"Missing required column(s): {', '.join(missing)}"]
    return []


def _check_numeric_bounds(df: pd.DataFrame) -> list:
    """Return list of error strings for out-of-range numeric values."""
    errors = []
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col not in df.columns:
            continue
        # Coerce to numeric — anything unparseable becomes NaN and is skipped
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        bad = series[(series < lo) | (series > hi)]
        if not bad.empty:
            errors.append(
                f"Column '{col}': {len(bad)} value(s) outside expected range [{lo}, {hi}]. "
                f"Examples: {bad.head(3).tolist()}"
            )
    return errors


def validate_train_columns(df: pd.DataFrame) -> list:
    """Validate a training CSV. Returns list of error strings (empty = OK)."""
    errors = _check_columns(df, REQUIRED_TRAIN_COLS)
    if not errors:
        errors += _check_numeric_bounds(df)
    return errors


def validate_predict_columns(df: pd.DataFrame) -> list:
    """Validate a single-row or batch prediction DataFrame. Returns error strings."""
    errors = _check_columns(df, REQUIRED_PREDICT_COLS)
    if not errors:
        errors += _check_numeric_bounds(df)
    return errors


def validate_batch_columns(df: pd.DataFrame) -> list:
    """Validate a batch prediction CSV. Rejects if resale_price is present (likely wrong file)."""
    errors = validate_predict_columns(df)
    if 'resale_price' in df.columns:
        errors.append(
            "Column 'resale_price' found in batch file. "
            "Remove it — this file looks like a training dataset, not a prediction input."
        )
    return errors

# ── Pipeline Steps ────────────────────────────────────────────────────────────

def load_and_clean(source) -> pd.DataFrame:
    """Load CSV from a file path or file-like object (e.g. Streamlit uploader).

    pd.read_csv accepts both str paths and BytesIO objects — works in both
    notebook and Streamlit contexts without modification.
    """
    df = pd.read_csv(source)
    df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def prepare_xy(df: pd.DataFrame) -> tuple:
    """Drop irrelevant columns and separate features from target.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series  (resale_price)
    """
    df = df.drop(columns=COLS_TO_DROP, errors='ignore')
    return df.drop(columns=['resale_price']), df['resale_price']


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
) -> tuple:
    """Sort by month and split chronologically into train / val / test.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    sort_idx = X['month'].sort_values().index
    X, y = X.loc[sort_idx], y.loc[sort_idx]
    n         = len(X)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    return (
        X.iloc[:train_end],  X.iloc[train_end:val_end],  X.iloc[val_end:],
        y.iloc[:train_end],  y.iloc[train_end:val_end],  y.iloc[val_end:],
    )


def log_transform(*arrays):
    """Apply log1p-safe log transform to one or more target arrays.

    Usage: y_train_log, y_val_log, y_test_log = log_transform(y_train, y_val, y_test)
    """
    transformed = tuple(np.log(a) for a in arrays)
    return transformed if len(transformed) > 1 else transformed[0]


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for the full feature pipeline."""
    return ColumnTransformer(
        transformers=[
            ('temporal',      TemporalFeatureExtractor(),                       TEMPORAL_COL),
            ('lease',         LeaseFeatureCreator(),                            LEASE_COL),
            ('ordinal',       OrdinalEncoder(categories=[STOREY_ORDER]),        ORDINAL_COL),
            ('target_encode', CVTargetEncoderWrapper(n_folds=5, smoothing=1.0), HIGH_CARDINALITY_COL),
            ('onehot',        OneHotEncoder(sparse_output=False, drop='first',
                                            handle_unknown='ignore'),           LOW_CARDINALITY_COL),
            ('numerical',     StandardScaler(),                                 NUMERICAL_COL),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )


def fit_transform_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    *extra_sets,
) -> tuple:
    """Fit preprocessor on training data and transform all sets.

    Returns
    -------
    preprocessor, X_train_processed, [X_val_processed, X_test_processed, ...]
    """
    X_train_processed = preprocessor.fit_transform(X_train, y_train_log)
    extra_processed   = tuple(preprocessor.transform(X) for X in extra_sets)
    return (preprocessor, X_train_processed) + extra_processed


# ── Evaluation & Prediction ───────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> dict:
    """Compute RMSE, MAE, R², Adjusted R² on original (dollar) scale."""
    n      = len(y_true)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    mae    = mean_absolute_error(y_true, y_pred)
    r2     = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Adj_R2': adj_r2}


def predict(model, X: np.ndarray) -> np.ndarray:
    """Predict and inverse log-transform back to original price scale."""
    return np.clip(np.exp(model.predict(X)), 0, None)


# ── Final Model Helpers ───────────────────────────────────────────────────────

def merge_train_val(
    X_train: np.ndarray, X_val: np.ndarray,
    y_train_log: np.ndarray, y_val_log: np.ndarray,
) -> tuple:
    """Concatenate train and val for final model fitting."""
    return np.concatenate([X_train, X_val]), np.concatenate([y_train_log, y_val_log])