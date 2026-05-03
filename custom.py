import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import KFold, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Custom transformer for temporal features (modified to handle multiple date columns)
class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_out_ = None
        self.input_columns_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.input_columns_ = X.columns.tolist()
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            # If numpy array, assume column names from fit
            X_df = pd.DataFrame(X, columns=self.input_columns_)

        all_features = []

        # Process each date column
        for col in X_df.columns:
            # Convert to datetime
            date_series = pd.to_datetime(X_df[col])

            # Extract temporal features for this column
            col_features = np.column_stack([
                date_series.dt.year,
                date_series.dt.month,
                date_series.dt.day,
                np.sin(2 * np.pi * date_series.dt.month / 12),  # month_sin
                np.cos(2 * np.pi * date_series.dt.month / 12)   # month_cos
            ])

            all_features.append(col_features)

        # Concatenate all features horizontally
        return np.column_stack(all_features)

    def get_feature_names_out(self, input_features=None):
        if input_features is None and self.input_columns_ is not None:
            input_features = self.input_columns_
        elif input_features is None:
            input_features = ['date_col']

        feature_names = []
        for col in input_features:
            # Create unique feature names for each date column
            col_prefix = col.replace('_', '').replace(' ', '').lower()
            feature_names.extend([
                f'{col_prefix}_year',
                f'{col_prefix}_month',
                f'{col_prefix}_day',
                f'{col_prefix}_month_sin',
                f'{col_prefix}_month_cos'
            ])

        return feature_names

# Custom transformer for lease features
class LeaseFeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        # Assuming columns are [remaining_lease_years, remaining_lease_months]
        total_months = X_array[:, 0] * 12 + X_array[:, 1]
        
        return total_months.reshape(-1, 1)
    
    def get_feature_names_out(self, input_features=None):
        return ['total_remaining_lease_months']

# PyTorch-accelerated Target Encoder with Cross-Validation
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_folds=5, smoothing=1.0):
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.global_means_ = {}
        self.encodings_ = {}
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._is_fitted = False
        
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X_df = X
        else:
            # If X is numpy array, we need column names from somewhere else
            self.feature_names_in_ = [f'col_{i}' for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        
        # Convert target to PyTorch tensor
        if hasattr(y, 'values'):
            y_values = y.values
        else:
            y_values = y
            
        y_tensor = torch.tensor(y_values, dtype=torch.float32, device=self.device)
        
        for col in self.feature_names_in_:
            # Get unique categories and create mapping
            unique_cats = X_df[col].unique()
            cat_to_idx = {cat: idx for idx, cat in enumerate(unique_cats)}
            
            # Convert categories to indices tensor
            cat_indices = torch.tensor(
                [cat_to_idx[cat] for cat in X_df[col]], 
                dtype=torch.long, 
                device=self.device
            )
            
            # Calculate global mean for each category using PyTorch
            n_categories = len(unique_cats)
            category_sums = torch.zeros(n_categories, device=self.device)
            category_counts = torch.zeros(n_categories, device=self.device)
            
            # Use scatter_add for efficient aggregation
            category_sums.scatter_add_(0, cat_indices, y_tensor)
            category_counts.scatter_add_(0, cat_indices, torch.ones_like(y_tensor))
            
            # Apply smoothing and calculate means
            global_mean = y_tensor.mean()
            smoothed_means = (
                (category_sums + self.smoothing * global_mean) / 
                (category_counts + self.smoothing)
            )
            
            # Store encodings
            self.encodings_[col] = {
                unique_cats[i]: smoothed_means[i].cpu().item() 
                for i in range(n_categories)
            }
            self.global_means_[col] = global_mean.cpu().item()
        
        self.feature_names_out_ = [f'{col}_target_encoded' for col in self.feature_names_in_]
        self._is_fitted = True
        return self
    
    def transform(self, X):
        if not self._is_fitted:
            raise ValueError("This TargetEncoder instance is not fitted yet.")
            
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        
        encoded_features = []
        for col in self.feature_names_in_:
            # Apply encoding with fallback to global mean
            encoded_col = X_df[col].map(self.encodings_[col]).fillna(self.global_means_[col])
            encoded_features.append(encoded_col.values)
            
        return np.column_stack(encoded_features)
    
    def fit_transform_cv(self, X, y):
        """Fit and transform with cross-validation to prevent overfitting"""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X_df = X
        else:
            self.feature_names_in_ = [f'col_{i}' for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        
        kf = TimeSeriesSplit(n_splits=self.n_folds)
        encoded_features = []
        
        # Convert target to PyTorch tensor
        if hasattr(y, 'values'):
            y_values = y.values
        else:
            y_values = y
            
        y_tensor = torch.tensor(y_values, dtype=torch.float32, device=self.device)
        
        for col in self.feature_names_in_:
            encoded_values = np.zeros(len(X_df))
            
            for train_idx, val_idx in kf.split(X_df):
                # Get training fold data
                X_train_fold = X_df.iloc[train_idx]
                y_train_fold = y_tensor[train_idx]
                
                # Create temporary encoder for this fold
                unique_cats = X_train_fold[col].unique()
                cat_to_idx = {cat: idx for idx, cat in enumerate(unique_cats)}
                
                # Convert categories to indices
                cat_indices = torch.tensor(
                    [cat_to_idx.get(cat, -1) for cat in X_train_fold[col]], 
                    dtype=torch.long, 
                    device=self.device
                )
                
                # Filter out unknown categories
                valid_mask = cat_indices >= 0
                if valid_mask.sum() > 0:
                    cat_indices_valid = cat_indices[valid_mask]
                    y_train_valid = y_train_fold[valid_mask]
                    
                    # Calculate fold encodings
                    n_categories = len(unique_cats)
                    category_sums = torch.zeros(n_categories, device=self.device)
                    category_counts = torch.zeros(n_categories, device=self.device)
                    
                    category_sums.scatter_add_(0, cat_indices_valid, y_train_valid)
                    category_counts.scatter_add_(0, cat_indices_valid, torch.ones_like(y_train_valid))
                    
                    # Apply smoothing
                    fold_mean = y_train_valid.mean()
                    smoothed_means = (
                        (category_sums + self.smoothing * fold_mean) / 
                        (category_counts + self.smoothing)
                    )
                    
                    # Create fold encoding dictionary
                    fold_encoding = {
                        unique_cats[i]: smoothed_means[i].cpu().item() 
                        for i in range(n_categories)
                    }
                    
                    # Encode validation fold
                    for idx in val_idx:
                        cat = X_df.iloc[idx][col]
                        encoded_values[idx] = fold_encoding.get(cat, fold_mean.cpu().item())
            
            encoded_features.append(encoded_values)
            
        # Fit on full data for future transforms
        self.fit(X, y)
        return np.column_stack(encoded_features)
    
    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out_ is None:
            if input_features is not None:
                return [f'{col}_target_encoded' for col in input_features]
            else:
                return [f'target_encoded_{i}' for i in range(len(self.feature_names_in_ or []))]
        return self.feature_names_out_

# Custom wrapper for CV Target Encoding in ColumnTransformer
class CVTargetEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, n_folds=5, smoothing=1.0):
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.encoder = None
        self._target = None
        
    def fit(self, X, y=None):
        if y is not None:
            self._target = y
            self.encoder = TargetEncoder(n_folds=self.n_folds, smoothing=self.smoothing)
            self.encoder.fit(X, y)
        return self
    
    def transform(self, X):
        if self.encoder is None:
            raise ValueError("Encoder not fitted. Call fit first.")
        return self.encoder.transform(X)
    
    def fit_transform(self, X, y=None):
        if y is not None:
            self._target = y
            self.encoder = TargetEncoder(n_folds=self.n_folds, smoothing=self.smoothing)
            return self.encoder.fit_transform_cv(X, y)
        else:
            return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        if self.encoder is not None:
            return self.encoder.get_feature_names_out(input_features)
        elif input_features is not None:
            return [f'{col}_target_encoded' for col in input_features]
        else:
            return []

