# %%
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from regression import LogisticRegression
import random
import matplotlib.pyplot as plt
import numpy as np
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Uploading the dataset from UCI repository
url = "https://archive.ics.uci.edu/static/public/222/data.csv"
df_bank = pd.read_csv(url)

df_bank
# %%
df_bank.isna().sum()
# %%
df_bank = df_bank.dropna(subset=["job", "education"])
df_bank.isna().sum()
# %%
df_bank = df_bank.drop(columns=["contact", "poutcome"])
df_bank.isna().sum()
# %%
df_bank = pd.get_dummies(
    df_bank, columns=["job", "marital", "education", "month"], drop_first=True)

df_bank
# %%
for col in ["default", "housing", "loan", "y"]:
    df_bank[col] = df_bank[col].replace({"yes": 1, "no": 0})

df_bank
# %%
df_encoded = pd.get_dummies(df_bank, drop_first=True)
df_encoded = df_encoded.astype(
    {col: int for col in df_encoded.select_dtypes(bool)})
num_df = df_encoded.select_dtypes(include=[np.number]).copy()

num_df

# %%

corr = num_df.corr()

plt.figure(figsize=(12, 10))
plt.imshow(corr.values, aspect="auto")
plt.xticks(ticks=np.arange(corr.shape[1]), labels=corr.columns, rotation=90)
plt.yticks(ticks=np.arange(corr.shape[0]), labels=corr.index)
plt.title("Mapa de calor de correlaciones")
plt.colorbar()
plt.tight_layout()
plt.show()

# %%
target = "y"
if target in num_df.columns:
    corr_y = num_df.drop(
        columns=[c for c in [target] if c in num_df.columns]).corrwith(num_df[target])
    corr_y = corr_y.sort_values(key=lambda s: s.abs(), ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(corr_y.index, corr_y.values)
    plt.gca().invert_yaxis()
    plt.title(f"Correlación de cada variable con {target}")
    plt.xlabel("Correlación de Pearson")
    plt.tight_layout()
    plt.show()
# %%
vars_only = [c for c in num_df.columns if c != target]

pair_corr = num_df[vars_only].corr().abs()

for i in range(pair_corr.shape[0]):
    pair_corr.iat[i, i] = 0.0

pairs = []
cols_list = pair_corr.columns.tolist()
for i in range(len(cols_list)):
    for j in range(i+1, len(cols_list)):
        pairs.append((cols_list[i], cols_list[j], pair_corr.iat[i, j]))

pairs_sorted = sorted(pairs, key=lambda t: t[2], reverse=True)

top_k = 5

# %%
df = num_df.copy()
top_cols = list({c for a, b, _ in pairs_sorted[:top_k] for c in (a, b)})
top_cols
# %%


def stratified_train_test_split(X, y, test_size=0.2, seed=42):

    random.seed(seed)

    idx_pos = [i for i, t in enumerate(y) if t == 1]
    idx_neg = [i for i, t in enumerate(y) if t == 0]

    random.shuffle(idx_pos)
    random.shuffle(idx_neg)

    n_pos_test = max(1, int(len(idx_pos) * test_size)
                     ) if len(idx_pos) > 0 else 0
    n_neg_test = max(1, int(len(idx_neg) * test_size)
                     ) if len(idx_neg) > 0 else 0

    test_idx = set(idx_pos[:n_pos_test] + idx_neg[:n_neg_test])
    train_idx = [i for i in range(len(y)) if i not in test_idx]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, X_test, y_train, y_test


X = num_df.drop(columns=[target])
y = num_df["y"].astype(int)

X_list = X.values.tolist()
y_list = y.values.tolist()

X_train, X_test, y_train, y_test = stratified_train_test_split(
    X_list, y_list, test_size=0.2, seed=42
)

print(f"Train: {len(y_train)} | Test: {len(y_test)}")

# %%

TARGET_COL = "y"
feature_cols = top_cols

X = df[feature_cols].astype(float).values.tolist()
y = df[TARGET_COL].astype(int).values.tolist()

clf = LogisticRegression(lr=0.0005, epochs=1000, l2=0.0001, verbose_every=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X, threshold=0.5)
acc = clf.accuracy(y, y_pred)
prec, rec, f1 = clf.precision_recall_f1(y, y_pred)
print(
    f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

scores = clf.predict_proba(X)
fprs, tprs = clf.roc_curve(y, scores)
auc_val = clf.auc(fprs, tprs)
print(f"AUC={auc_val:.4f}")

# Curva de pérdida
clf.plot_loss()
# %%
y_pred_test = clf.predict(X_test, threshold=0.3)
y_prob_test = clf.predict_proba(X_test)

# Métricas
acc = clf.accuracy(y_test, y_pred_test)
prec, rec, f1 = clf.precision_recall_f1(y_test, y_pred_test)
print(
    f"Test -> Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

# (Opcional) ROC-AUC desde cero
fprs, tprs = clf.roc_curve(y_test, y_prob_test)
auc_val = clf.auc(fprs, tprs)
print(f"Test -> AUC={auc_val:.4f}")
# %%


def plot_confusion_matrix(y_true, y_pred, title="Matriz de confusión"):
    TP = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    TN = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    FP = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    FN = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))

    matrix = [[TN, FP],
              [FN, TP]]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred=0", "Pred=1"])
    ax.set_yticklabels(["Real=0", "Real=1"])
    ax.set_xlabel("Predicciones")
    ax.set_ylabel("Valores reales")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i][j], ha="center", va="center",
                    color="black", fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


plot_confusion_matrix(y_test, y_pred_test)
# %% md
# # ARBOLES DE DESICION
# %%

use_cols = top_cols + [target]
data = df[use_cols].dropna().copy()

X = data[top_cols].astype(float)
y = data[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train.shape, X_test.shape
# %%

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    min_samples_split=8,
    max_features=0.5,
    max_samples=0.7,
    bootstrap=True
)
rf.fit(X_train, y_train)
# %%

y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
# %%

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot(cmap="Blues")
plt.title("Matriz de Confusión - RandomForest")
plt.show()

# %%

# Probabilidades de la clase positiva
y_proba = rf.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC:", auc)

RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("Curva ROC - RandomForest")
plt.show()
# %%

importances = rf.feature_importances_
idx = np.argsort(importances)[::-1]  # de mayor a menor

plt.figure(figsize=(7, 5))
plt.barh(np.array(top_cols)[idx][:20][::-1], importances[idx][:20][::-1])
plt.xlabel("Importancia (Gini)")
plt.title("Top features - RandomForest")
plt.tight_layout()
plt.show()
# %%
use_cols = top_cols + [target]
data = df[use_cols].dropna().copy()

X_2 = data[top_cols].astype(float)
y_2 = data[target].astype(int)

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X_2, y_2, test_size=0.2, random_state=42, stratify=y
)

X_train_2.shape, X_test_2.shape
# %%
rf_2 = RandomForestClassifier(
    n_estimators=1000,
    max_depth=8,
    min_samples_leaf=4,
    min_samples_split=10,
    max_features=0.5,
    max_samples=0.7,
    bootstrap=True
)
rf_2.fit(X_train_2, y_train_2)
# %%

y_pred_2 = rf_2.predict(X_test_2)
print("Accuracy:", accuracy_score(y_test_2, y_pred_2))
print("\nReporte de clasificación:\n",
      classification_report(y_test_2, y_pred_2))
# %%
cm_2 = confusion_matrix(y_test_2, y_pred_2)
disp_2 = ConfusionMatrixDisplay(
    confusion_matrix=cm_2, display_labels=rf_2.classes_)
disp.plot(cmap="Blues")
plt.title("Matriz de Confusión - RandomForest")
plt.show()

# %%

train_sizes, train_scores, val_scores = learning_curve(
    estimator=rf_2,
    X=X, y=y,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
    shuffle=True,
    random_state=42
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)


# %%

test_scores = []

for m in train_sizes:
    subset = int(m)
    rf.fit(X_train[:subset], y_train[:subset])
    y_pred_test = rf.predict(X_test)
    test_scores.append(accuracy_score(y_test, y_pred_test))

# Plot con test
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label="Training")
plt.plot(train_sizes, val_mean, label="Validation")
plt.plot(train_sizes, test_scores, label="Testing")

plt.xlabel("Tamaño de entrenamiento")
plt.ylabel("Accuracy")
plt.title("Curva de Aprendizaje con Test Set")
plt.legend()
plt.grid(True)
plt.show()
