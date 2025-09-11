"""
REGRESIÓN LOGÍSTICA BINARIA
- Multivariada (múltiples features)
- Descenso de gradiente con regularización L2 opcional
- Estandarización (media 0, var 1) por feature
- Early stopping por tolerancia en pérdida
- Métricas: accuracy, precisión, recall, F1, ROC y AUC (trapezoidal manual)
"""

from typing import List, Tuple
import math
import random


class LogisticRegression:
    def __init__(
        self,
        lr: float = 1e-2,
        epochs: int = 6000,
        l2: float = 0.0,
        tol: float = 1e-7,
        early_stopping: bool = True,
        verbose_every: int = 0
    ):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.tol = tol
        self.early_stopping = early_stopping
        self.verbose_every = verbose_every

        self.w: List[float] = []
        self.b: float = 0.0

        self.loss_history: List[float] = []

        self._means: List[float] = []
        self._stds: List[float] = []

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    @staticmethod
    def _dot(a: List[float], b: List[float]) -> float:
        return sum(x*y for x, y in zip(a, b))

    @staticmethod
    def _bce(y_true: List[int], y_prob: List[float]) -> float:
        # Binary Cross-Entropy
        eps = 1e-12
        n = len(y_true)
        return -sum(yt * math.log(max(p, eps)) + (1-yt) * math.log(max(1-p, eps))
                    for yt, p in zip(y_true, y_prob)) / n

    @staticmethod
    def _transpose(M: List[List[float]]) -> List[List[float]]:
        # transponer lista de listas
        return list(map(list, zip(*M))) if M else []

    def _standardize_fit(self, X: List[List[float]]) -> List[List[float]]:
        # calcula medias y std por feature y transforma
        XT = self._transpose(X)
        self._means = []
        self._stds = []
        for col in XT:
            n = len(col)
            mu = sum(col)/n
            var = sum((xi - mu)**2 for xi in col)/n
            std = math.sqrt(var) if var > 0 else 1.0
            self._means.append(mu)
            self._stds.append(std)
        return self._standardize_transform(X)

    def _standardize_transform(self, X: List[List[float]]) -> List[List[float]]:
        Z = []
        for row in X:
            zr = []
            for j, xij in enumerate(row):
                zr.append((xij - self._means[j]) / self._stds[j])
            Z.append(zr)
        return Z

    def fit(self, X: List[List[float]], y: List[int]) -> None:
        assert len(X) == len(y) and len(X) > 0, "Datos vacíos o desalineados"
        n = len(X)
        d = len(X[0])

        Z = self._standardize_fit(X)

        rnd = random.Random(42)
        self.w = [rnd.uniform(-0.01, 0.01) for _ in range(d)]
        self.b = 0.0
        self.loss_history.clear()

        prev_loss = float("inf")

        for epoch in range(self.epochs):
            logits = [self._dot(self.w, zi) + self.b for zi in Z]
            probs = [self._sigmoid(z) for z in logits]
            loss = self._bce(y, probs)
            if self.l2 > 0:
                loss += (self.l2 / (2*n)) * sum(wj*wj for wj in self.w)
            self.loss_history.append(loss)

            grad_w = [0.0]*d
            grad_b = 0.0
            for i in range(n):
                err = probs[i] - y[i]
                grad_b += err
                zi = Z[i]
                for j in range(d):
                    grad_w[j] += err * zi[j]

            grad_b /= n
            for j in range(d):
                grad_w[j] = grad_w[j]/n + (self.l2/n)*self.w[j] if self.l2 > 0 else grad_w[j]/n

            for j in range(d):
                self.w[j] -= self.lr * grad_w[j]
            self.b -= self.lr * grad_b

            if self.verbose_every and epoch % self.verbose_every == 0:
                print(f"[{epoch:4d}] loss={loss:.6f} | |w|={math.sqrt(sum(w*w for w in self.w)):.4f} b={self.b:.4f}")

            if self.early_stopping and abs(prev_loss - loss) < self.tol:
                if self.verbose_every:
                    print(f"Early stopping en epoch {epoch} (Δloss={abs(prev_loss-loss):.2e})")
                break
            prev_loss = loss

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        Z = self._standardize_transform(X)
        logits = [self._dot(self.w, zi) + self.b for zi in Z]
        return [self._sigmoid(z) for z in logits]

    def predict(self, X: List[List[float]], threshold: float = 0.5) -> List[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    @staticmethod
    def confusion_matrix(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
        TP = FP = TN = FN = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 1:
                TP += 1
            elif yt == 0 and yp == 1:
                FP += 1
            elif yt == 0 and yp == 0:
                TN += 1
            elif yt == 1 and yp == 0:
                FN += 1
        return TP, FP, TN, FN

    @staticmethod
    def accuracy(y_true: List[int], y_pred: List[int]) -> float:
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        return correct / len(y_true)

    @staticmethod
    def precision_recall_f1(y_true: List[int], y_pred: List[int]) -> Tuple[float, float, float]:
        TP, FP, TN, FN = LogisticRegression.confusion_matrix(y_true, y_pred)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    @staticmethod
    def roc_curve(y_true: List[int], y_score: List[float]) -> Tuple[List[float], List[float]]:
        pairs = sorted(zip(y_score, y_true), key=lambda t: t[0], reverse=True)
        P = sum(y_true)
        N = len(y_true) - P
        TP = FP = 0
        prev_score = float("inf")
        FPRs = [0.0]
        TPRs = [0.0]

        for score, yt in pairs:
            if score != prev_score:
                FPRs.append(FP / N if N > 0 else 0.0)
                TPRs.append(TP / P if P > 0 else 0.0)
                prev_score = score
            if yt == 1:
                TP += 1
            else:
                FP += 1

        FPRs.append(FP / N if N > 0 else 0.0)
        TPRs.append(TP / P if P > 0 else 0.0)
        return FPRs, TPRs

    @staticmethod
    def auc(fprs: List[float], tprs: List[float]) -> float:
        pts = sorted(zip(fprs, tprs))
        area = 0.0
        for i in range(1, len(pts)):
            x0, y0 = pts[i-1]
            x1, y1 = pts[i]
            base = x1 - x0
            height = (y0 + y1) / 2.0
            area += base * height
        return area

    def plot_loss(self):
        if not self.loss_history:
            print("No hay historial de pérdida.")
            return
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(1, len(self.loss_history)+1), self.loss_history)
        plt.xlabel("Épocas")
        plt.ylabel("Loss (BCE)")
        plt.title("Curva de entrenamiento (Logistic)")
        plt.tight_layout()
        plt.show()
