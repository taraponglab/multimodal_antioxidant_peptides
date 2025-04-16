import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional,
    LSTM, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, confusion_matrix, average_precision_score
)

# ====== Define Base Models (CNN, BiLSTM, Transformer) ======
def create_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
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
    x = Conv1D(embed_dim, kernel_size=1, activation='relu')(inputs)

    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    attn_output = Dropout(0.1)(attn_output)
    out1 = Add()([x, attn_output])
    out1 = LayerNormalization(epsilon=1e-6)(out1)

    ffn = Dense(ff_dim, activation='relu')(out1)
    ffn = Dense(embed_dim)(ffn)
    ffn = Dropout(0.1)(ffn)
    out2 = Add()([out1, ffn])
    out2 = LayerNormalization(epsilon=1e-6)(out2)

    x = GlobalAveragePooling1D()(out2)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ====== Define Meta Model ======
def create_meta_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ====== Evaluate Metrics ======
def evaluate_model(y_true, y_pred_prob):
    y_pred = (y_pred_prob > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_pred_prob),
        "pr_auc": average_precision_score(y_true, y_pred_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred)
    }

# ====== Run Stacking ======
def run_stacking(X_train, y_train, X_test, y_test, n_repeats=3):
    final_results = {metric: [] for metric in ["accuracy", "balanced_accuracy", "auc", "pr_auc", "mcc", "precision", "recall", "specificity", "f1"]}

    for repeat in range(n_repeats):
        print(f"\n===== Training Run {repeat+1}/{n_repeats} =====")

        models = [
            create_cnn(X_train.shape[1:]),
            create_bilstm(X_train.shape[1:]),
            create_transformer(X_train.shape[1:])
        ]

        model_preds = []
        for model in models:
            model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
            model_preds.append(model.predict(X_test).flatten())

        meta_X = np.column_stack(model_preds)
        meta_model = create_meta_model(meta_X.shape[1])
        meta_model.fit(meta_X, y_test, epochs=50, batch_size=32, verbose=0)

        meta_y_pred_prob = meta_model.predict(meta_X).flatten()
        metrics = evaluate_model(y_test, meta_y_pred_prob)

        for metric in final_results:
            final_results[metric].append(metrics[metric])

        print(f"→ Accuracy: {metrics['accuracy']:.3f}, AUROC: {metrics['auc']:.3f}, AUPRC: {metrics['pr_auc']:.3f}, MCC: {metrics['mcc']:.3f}")

    print("\n=== Final Evaluation (Mean ± Std) ===")
    for metric, values in final_results.items():
        print(f"{metric.capitalize():18}: {np.mean(values):.3f} ± {np.std(values):.3f}")

# ====== Main Execution ======
def main():
    # Read CSV files
    X_train_df = pd.read_csv("Antiox_x_train_onehot.csv")
    y_train = pd.read_csv("Antiox_y_train_onehot.csv").values.ravel()
    X_test_df = pd.read_csv("Antiox_x_test_onehot.csv")
    y_test = pd.read_csv("Antiox_y_test_onehot.csv").values.ravel()

    X_train = X_train_df.values
    X_test = X_test_df.values

    # Determine the dimensionality (flattened = max_length * 20)
    total_features = X_train.shape[1]
    assert total_features % 20 == 0, "⚠️ Error!!!"

    max_length = total_features // 20

    # 3D reshape: [samples, max_length, 20]
    X_train = X_train.reshape((-1, max_length, 20))
    X_test = X_test.reshape((-1, max_length, 20))

    print("✅ Input shapes:")
    print("  X_train:", X_train.shape)
    print("  X_test :", X_test.shape)

    # Run stacking model
    run_stacking(X_train, y_train, X_test, y_test, n_repeats=3)

if __name__ == "__main__":
    main()
