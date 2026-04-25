import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional,
    LSTM, Input, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    matthews_corrcoef, precision_score, recall_score,
    f1_score, confusion_matrix, average_precision_score
)

# =========================
# Models
# =========================
def create_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_bilstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_transformer(input_shape, embed_dim=128, num_heads=4, ff_dim=128):
    inputs = Input(shape=input_shape)
    x = Conv1D(embed_dim, 1, activation='relu')(inputs)

    attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    attn = Dropout(0.1)(attn)
    out1 = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))

    ffn = Dense(ff_dim, activation='relu')(out1)
    ffn = Dense(embed_dim)(ffn)
    ffn = Dropout(0.1)(ffn)
    out2 = LayerNormalization(epsilon=1e-6)(Add()([out1, ffn]))

    x = GlobalAveragePooling1D()(out2)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_meta_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# =========================
# Utils
# =========================
def load_data(train_x, train_y, test_x, test_y):
    X_train = pd.read_csv(train_x).values
    y_train = pd.read_csv(train_y).values.ravel()
    X_test = pd.read_csv(test_x).values
    y_test = pd.read_csv(test_y).values.ravel()
    return X_train, y_train, X_test, y_test


def reshape_data(X_train, X_test, num_channels=20):
    total_features = X_train.shape[1]
    assert total_features % num_channels == 0, "Invalid feature shape"

    max_length = total_features // num_channels

    X_train = X_train.reshape((-1, max_length, num_channels))
    X_test = X_test.reshape((-1, max_length, num_channels))

    return X_train, X_test


def evaluate_model(y_true, y_pred_prob):
    y_pred = (y_pred_prob > 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    auc = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.5

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc": auc,
        "pr_auc": average_precision_score(y_true, y_pred_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }


# =========================
# OOF STACKING
# =========================
def run_stacking_oof(X_train, y_train, X_test, y_test, n_splits=5, n_repeats=3):

    results = {k: [] for k in [
        "accuracy", "balanced_accuracy", "auc", "pr_auc",
        "mcc", "precision", "recall", "specificity", "f1"
    ]}

    for repeat in range(n_repeats):
        print(f"\n===== Run {repeat+1}/{n_repeats} =====")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)

        meta_X_train = np.zeros((X_train.shape[0], 3))
        meta_X_test = np.zeros((X_test.shape[0], 3))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"  Fold {fold+1}/{n_splits}")

            tf.keras.backend.clear_session()

            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr = y_train[train_idx]

            models = [
                create_cnn(X_tr.shape[1:]),
                create_bilstm(X_tr.shape[1:]),
                create_transformer(X_tr.shape[1:])
            ]

            es = EarlyStopping(patience=5, restore_best_weights=True)

            for i, model in enumerate(models):
                model.fit(X_tr, y_tr, epochs=30, batch_size=32,
                          verbose=0, callbacks=[es])

                meta_X_train[val_idx, i] = model.predict(X_val).flatten()
                meta_X_test[:, i] += model.predict(X_test).flatten() / n_splits

        meta_model = create_meta_model(meta_X_train.shape[1])
        meta_model.fit(meta_X_train, y_train,
                       epochs=50, batch_size=32, verbose=0)

        meta_pred = meta_model.predict(meta_X_test).flatten()

        metrics = evaluate_model(y_test, meta_pred)

        for k in results:
            results[k].append(metrics[k])

        print(f"→ Acc: {metrics['accuracy']:.3f}, AUC: {metrics['auc']:.3f}, MCC: {metrics['mcc']:.3f}")

    print("\n=== FINAL ===")
    for k, v in results.items():
        print(f"{k:18}: {np.mean(v):.3f} ± {np.std(v):.3f}")


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="OOF Stacking with Meta Model")

    parser.add_argument("--train_x", required=True)
    parser.add_argument("--train_y", required=True)
    parser.add_argument("--test_x", required=True)
    parser.add_argument("--test_y", required=True)

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=3)

    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_data(
        args.train_x, args.train_y, args.test_x, args.test_y
    )

    X_train, X_test = reshape_data(X_train, X_test)

    run_stacking_oof(
        X_train, y_train, X_test, y_test,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats
    )


if __name__ == "__main__":
    main()
