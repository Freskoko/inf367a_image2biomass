from sklearn.preprocessing import StandardScaler
import pandas as pd


def apply_scaling_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

    scaler = StandardScaler()

    Xtr_scaled = X_train.copy()
    Xte_scaled = X_test.copy()

    Xtr_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    Xte_scaled[num_cols] = scaler.transform(X_test[num_cols])

    return Xtr_scaled, Xte_scaled
