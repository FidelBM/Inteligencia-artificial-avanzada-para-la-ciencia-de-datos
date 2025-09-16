import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.model_selection import learning_curve
import math


__errors__ = []

###################################
# REGRESION LOGISTICA
###################################


def h(params, sample):
    acum = 0
    for i in range(len(params)):
        acum = acum + params[i]*sample[i]
    acum = acum*(-1)
    acum = 1/(1 + math.exp(acum))
    return acum


def show_errors(params, samples, y):
    global __errors__
    error_acum = 0
    for i in range(len(samples)):
        hyp = h(params, samples[i])
        if (y[i] == 1):
            if (hyp == 0):
                hyp = .0001
            error = (-1)*math.log(hyp)
        if (y[i] == 0):
            if (hyp == 1):
                hyp = .9999
            error = (-1)*math.log(1-hyp)

        error_acum = error_acum + error
    mean_error_param = error_acum/len(samples)
    __errors__.append(mean_error_param)
    return mean_error_param


def GD(params, samples, y, alfa):
    temp = list(params)
    for j in range(len(params)):
        acum = 0
        for i in range(len(samples)):
            error = h(params, samples[i]) - y[i]
            acum = acum + error*samples[i][j]
        temp[j] = params[j] - alfa*(1/len(samples))*acum
    return temp


def scaling(samples):
    acum = 0
    samples = np.asarray(samples).T.tolist()
    for i in range(1, len(samples)):
        for j in range(len(samples[i])):
            acum = + samples[i][j]
        avg = acum/(len(samples[i]))
        max_val = max(samples[i])
        print(
            "To scale feature %i use (Value - avg[%f]) / maxval[%f]" % (i, avg, max_val))
        for j in range(len(samples[i])):
            samples[i][j] = (samples[i][j] - avg)/max_val
    return np.asarray(samples).T.tolist()


def predict_proba(params, samples):
    return [h(params, s) for s in samples]


def predict(params, samples, threshold=0.5):
    return [1 if h(params, s) >= threshold else 0 for s in samples]

###################################
# CARGA Y PROCESAMIENTO DE DATOS
###################################


# Subir DataFrame
url = "https://archive.ics.uci.edu/static/public/222/data.csv"
df_bank = pd.read_csv(url)

# Convertir DataFrame
df_bank.isna().sum()
df_bank = df_bank.dropna(subset=["job", "education"])
df_bank = df_bank.drop(columns=["contact", "poutcome"])
df_bank = pd.get_dummies(
    df_bank, columns=["job", "marital", "education", "month"], drop_first=True)
for col in ["default", "housing", "loan", "y"]:
    df_bank[col] = df_bank[col].replace({"yes": 1, "no": 0})
df_bank.head()
df_encoded = pd.get_dummies(df_bank, drop_first=True)
df_encoded = df_encoded.astype(
    {col: int for col in df_encoded.select_dtypes(bool)})
num_df = df_encoded.select_dtypes(include=[np.number]).copy()

# Correlación del DataFrame
corr = num_df.corr()
plt.figure(figsize=(12, 10))
plt.imshow(corr.values, aspect="auto")
plt.xticks(ticks=np.arange(corr.shape[1]), labels=corr.columns, rotation=90)
plt.yticks(ticks=np.arange(corr.shape[0]), labels=corr.index)
plt.title("Mapa de calor de correlaciones")
plt.colorbar()
plt.tight_layout()
plt.show()

# Correlacion con la variable objetivo
target = "y"
if target in num_df.columns:
    corr_y = num_df.drop(
        columns=[c for c in [target] if c in num_df.columns]).corrwith(num_df[target])
    corr_y = corr_y.sort_values(key=lambda s: s.abs(), ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(corr_y.index, corr_y.values)
    plt.gca().invert_yaxis()
    plt.title(f"Correlación de cada variable con {target}")
    plt.xlabel("Correlación")
    plt.tight_layout()
    plt.show()

# Obtención de las variables más correlacionadas entre sí
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
df = num_df.copy()
top_cols = list({c for a, b, _ in pairs_sorted[:top_k] for c in (a, b)})


# División de los datos en train y test
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


# Regresión Logística
TARGET_COL = "y"
feature_cols = top_cols
X = df[feature_cols].astype(float).values.tolist()
y = df[TARGET_COL].astype(int).values.tolist()
X = [[1] + row for row in X]
X = scaling(X)
params = [0.0] * len(X[0])
epochs = 2000
lr = 0.001
for _ in range(epochs):
    params = GD(params, X, y, lr)
    show_errors(params, X, y)

print("Parámetros finales:", params)
plt.plot(__errors__)
plt.title("Curva de Aprendizaje - Error Logístico")
plt.xlabel("Épocas")
plt.ylabel("Error")
plt.grid(True)
plt.show()

y_prob = predict_proba(params, X)
y_pred_test = predict(params, X, threshold=0.3)

print("Probabilidades:", y_prob)
print("Predicciones :", y_pred_test)

# Matriz de confusión


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

##################################
# RANDOM FOREST
##################################

# Separación de los datos en train y test

use_cols = top_cols + [target]
data = df[use_cols].dropna().copy()

X = data[top_cols].astype(float)
y = data[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Entrenamiento del modelo RandomForest
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

# Evaluación del modelo
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot(cmap="Blues")
plt.title("Matriz de Confusión - RandomForest")
plt.show()

# Probabilidades de la clase positiva
y_proba = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC:", auc)
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("Curva ROC - RandomForest")
plt.show()

# Importancia de las características

importances = rf.feature_importances_
idx = np.argsort(importances)[::-1]
plt.figure(figsize=(7, 5))
plt.barh(np.array(top_cols)[idx][:20][::-1], importances[idx][:20][::-1])
plt.xlabel("Importancia (Gini)")
plt.title("Top features - RandomForest")
plt.tight_layout()
plt.show()

# Segunda prueba con diferentes hiperparámetros
use_cols = top_cols + [target]
data = df[use_cols].dropna().copy()

X_2 = data[top_cols].astype(float)
y_2 = data[target].astype(int)

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X_2, y_2, test_size=0.2, random_state=42, stratify=y_2
)

# Entrenamiento del modelo RandomForest con nuevos hiperparámetros
rf_2 = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_leaf=4,
    min_samples_split=10,
    max_features=0.5,
    max_samples=0.7,
    bootstrap=True,
    class_weight='balanced'
)
rf_2.fit(X_train_2, y_train_2)

y_proba_2 = rf_2.predict_proba(X_test_2)[:, 1]
y_pred_2 = rf_2.predict(X_test_2)
RocCurveDisplay.from_predictions(y_test_2, y_proba_2)
plt.title("Curva ROC - RandomForest")
plt.show()
print("Accuracy:", accuracy_score(y_test_2, y_pred_2))
print("\nReporte de clasificación:\n",
      classification_report(y_test_2, y_pred_2))

cm_2 = confusion_matrix(y_test_2, y_pred_2)
disp_2 = ConfusionMatrixDisplay(
    confusion_matrix=cm_2, display_labels=rf_2.classes_)
disp_2.plot(cmap="Blues")
plt.title("Matriz de Confusión - RandomForest")
plt.show()

# Probabilidades de la clase positiva
train_sizes, train_scores, val_scores = learning_curve(
    estimator=rf_2,
    X=X_2, y=y_2,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
    shuffle=True,
    random_state=42
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

sizes_abs = train_sizes.astype(int)

test_scores = []
for n in sizes_abs:
    n = min(n, len(X_train_2))
    clf_tmp = RandomForestClassifier(
        n_estimators=rf_2.n_estimators,
        max_depth=rf_2.max_depth,
        min_samples_leaf=rf_2.min_samples_leaf,
        min_samples_split=rf_2.min_samples_split,
        max_features=rf_2.max_features,
        max_samples=rf_2.max_samples,
        bootstrap=rf_2.bootstrap,
        random_state=42
    )
    clf_tmp.fit(X_train_2[:n], y_train_2[:n])
    y_pred_test = clf_tmp.predict(X_test_2)
    test_scores.append(accuracy_score(y_test_2, y_pred_test))

plt.figure(figsize=(8, 6))
plt.plot(sizes_abs, train_mean, label="Training")
plt.plot(sizes_abs, val_mean,   label="Validation")
plt.plot(sizes_abs, test_scores, label="Testing")
plt.xlabel("Tamaño de entrenamiento")
plt.ylabel("Accuracy")
plt.title("Curva de Aprendizaje")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.show()
