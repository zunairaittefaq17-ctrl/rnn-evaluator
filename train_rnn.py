"""
train_rnn.py
============
RNN-Style Student Performance Prediction
Uses scikit-learn MLPClassifier on flattened weekly sequences.
Works on ANY Python version — no TensorFlow needed.

Run: python train_rnn.py
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.neural_network  import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (accuracy_score,
                                     confusion_matrix,
                                     classification_report)

print("=" * 60)
print("  RNN-Style Student Performance Prediction System")
print("=" * 60)

# ── STEP 1 : Load dataset ──────────────────────────────────────
print("\n[1] Loading dataset ...")
df = pd.read_excel("dataset.xlsx")
print(f"    Shape    : {df.shape}")
print(f"    Students : {df['student_id'].nunique()}")
print(f"    Weeks    : {sorted(df['week'].unique())}")
print("\n", df.head(10).to_string(index=False))

# ── STEP 2 : Build sequence feature matrix ─────────────────────
print("\n[2] Building sequence data ...")

FEATURES  = ['attendance', 'assignment', 'quiz', 'study_hours']
TIMESTEPS = 5

rows, labels = [], []
for sid, grp in df.groupby('student_id'):
    grp = grp.sort_values('week')
    # Flatten 5 weeks × 4 features = 20 input values per student
    flat = grp[FEATURES].values.flatten()
    rows.append(flat)
    labels.append(int(grp['result'].iloc[-1]))

X = np.array(rows)    # (50, 20)
y = np.array(labels)  # (50,)

print(f"    X shape : {X.shape}  (students × flattened weekly features)")
print(f"    y shape : {y.shape}")
print(f"    Pass: {y.sum()}  |  Fail: {(y==0).sum()}")

# ── STEP 3 : Train / test split ────────────────────────────────
print("\n[3] Splitting 80/20 ...")
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Train: {X_tr.shape[0]}  |  Test: {X_te.shape[0]}")

# ── STEP 4 : Scale ─────────────────────────────────────────────
print("\n[4] Scaling features ...")
scaler   = StandardScaler()
X_tr_s   = scaler.fit_transform(X_tr)
X_te_s   = scaler.transform(X_te)
print("    StandardScaler applied.")

# ── STEP 5 : Build & train model ───────────────────────────────
print("\n[5] Training MLP (RNN-style sequence model) ...")

model = MLPClassifier(
    hidden_layer_sizes = (128, 64, 32),
    activation         = 'relu',
    solver             = 'adam',
    max_iter           = 500,
    random_state       = 42,
    early_stopping     = True,
    validation_fraction= 0.15,
    n_iter_no_change   = 20,
    verbose            = True,
)
model.fit(X_tr_s, y_tr)
print(f"\n    Done — {model.n_iter_} iterations.")

# ── STEP 6 : Evaluate ──────────────────────────────────────────
print("\n[6] Evaluating ...")
y_pred = model.predict(X_te_s)
acc    = accuracy_score(y_te, y_pred)
cm     = confusion_matrix(y_te, y_pred)

print(f"\n    Accuracy : {acc*100:.2f}%")
print("\n    Confusion Matrix:\n", cm)
print("\n    Classification Report:")
print(classification_report(y_te, y_pred, target_names=['Fail','Pass']))

# ── Plots ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(model.loss_curve_, color='blue', label='Training Loss')
axes[0].set_title('Training Loss Curve')
axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(True)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fail','Pass'],
            yticklabels=['Fail','Pass'], ax=axes[1])
axes[1].set_title('Confusion Matrix')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('training_results.png', dpi=150)
print("\n    Plot saved → training_results.png")

# ── STEP 7 : Save ──────────────────────────────────────────────
print("\n[7] Saving model and scaler ...")
joblib.dump(model,  'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("    model.joblib  saved ✅")
print("    scaler.joblib saved ✅")

print("\n" + "="*60)
print("  ✅ Training complete!  Run: streamlit run app.py")
print("="*60)
