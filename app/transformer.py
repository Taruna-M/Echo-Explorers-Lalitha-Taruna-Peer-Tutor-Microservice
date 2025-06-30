from sklearn.base import BaseEstimator, TransformerMixin
class BooleanToIntegerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in X_copy.select_dtypes(include=[bool]).columns:
            X_copy[col] = X_copy[col].astype(int)
        return X_copy