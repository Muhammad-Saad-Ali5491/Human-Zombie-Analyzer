"""
Week 7 – Multivariate Linear Regression
========================================
Problem: Predict the Human-Zombie Score (0-100) from physical / lifestyle features.

Dataset columns
---------------
  Height (cm), Weight (kg), Screen Time (hrs), Junk Food (days/week),
  Physical Activity (hrs/week), Task Completion (scale)  →  Human-Zombie Score
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.linear_model import LinearRegression as SklearnLR, Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── 1. Load & Explore Data ───────────────────────────────────────────────────
df = pd.read_csv("human_zombie_dataset_v5.csv")
df.drop_duplicates(inplace=True)

print("Shape:", df.shape)
print(df.describe().T)
print("\nMissing values:\n", df.isnull().sum())
print("\nUnique values per column:\n", df.nunique())


# ── 2. Correlation Heatmap ───────────────────────────────────────────────────
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()


# ── 3. Feature / Target Split ────────────────────────────────────────────────
X = df.drop(columns=["Human-Zombie Score"])
y = df["Human-Zombie Score"]


# ── 4. EDA: Distributions & Scatter Plots ───────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, col in zip(axes.ravel(), X.columns):
    sns.histplot(df[col], kde=True, ax=ax, color='steelblue')
    ax.set_title(col)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, col in zip(axes.ravel(), X.columns):
    sns.scatterplot(data=df, x=col, y="Human-Zombie Score", ax=ax, alpha=0.6)
    ax.set_title(f"{col} vs Human-Zombie Score")
plt.tight_layout()
plt.show()


# ── 5. Train / Test Split & StandardScaler ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df  = pd.DataFrame(X_test_scaled,  columns=X_test.columns)

print("Scaler mean :", scaler.mean_)
print("Scaler scale:", scaler.scale_)


# ── 6. Custom Multivariate Linear Regression ─────────────────────────────────
class LinearRegressionCustom:
    """
    Multivariate Linear Regression with gradient descent.

    Supports cost functions: 'mse', 'mae', 'rmse'.
    Performs internal z-score normalisation so raw (un-scaled) data can be
    passed directly.
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000,
                 cost_function='mse'):
        self.learning_rate  = learning_rate
        self.num_iterations = num_iterations
        self.cost_function  = cost_function
        self.cost_history   = []
        self.theta          = None
        self.theta_history  = []
        self.X_mean         = None
        self.X_std          = None

    # ── helpers ──────────────────────────────────────────────────────────────
    def _add_bias(self, X):
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    def _normalize_features(self, X, is_training=False):
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if is_training:
            self.X_mean = X.mean(axis=0)
            self.X_std  = X.std(axis=0, ddof=0)
            self.X_std[self.X_std == 0] = 1.0
        if self.X_mean is None:
            raise ValueError("Call fit() before predict().")
        return (X - self.X_mean) / self.X_std

    def _compute_cost(self, predictions, y):
        y, predictions = np.array(y, dtype=float).flatten(), \
                         np.array(predictions, dtype=float).flatten()
        m, error = len(y), predictions - y
        if self.cost_function == 'mse':
            return (1 / (2 * m)) * np.sum(error ** 2)
        elif self.cost_function == 'mae':
            return (1 / m) * np.sum(np.abs(error))
        elif self.cost_function == 'rmse':
            return np.sqrt((1 / m) * np.sum(error ** 2))
        raise ValueError("cost_function must be 'mse', 'mae', or 'rmse'.")

    def _compute_gradients(self, X, predictions, y):
        y, predictions = np.array(y, dtype=float).flatten(), \
                         np.array(predictions, dtype=float).flatten()
        m, error = len(y), predictions - y
        if self.cost_function == 'mse':
            return (1 / m) * (X.T @ error)
        elif self.cost_function == 'rmse':
            rmse = np.sqrt((1 / m) * np.sum(error ** 2))
            denom = (m * rmse) if rmse != 0 else m
            return (1 / denom) * (X.T @ error)
        elif self.cost_function == 'mae':
            return (1 / m) * (X.T @ np.sign(error))
        raise ValueError("cost_function must be 'mse', 'mae', or 'rmse'.")

    # ── public API ───────────────────────────────────────────────────────────
    def fit(self, X, y, verbose=True):
        X_norm  = self._normalize_features(X, is_training=True)
        X_bias  = self._add_bias(X_norm)
        y       = np.array(y, dtype=float).flatten()
        self.theta        = np.zeros(X_bias.shape[1])
        self.cost_history = []
        self.theta_history= []

        log_every = max(1, self.num_iterations // 10)
        for i in range(self.num_iterations):
            predictions = X_bias @ self.theta
            cost        = self._compute_cost(predictions, y)
            gradients   = self._compute_gradients(X_bias, predictions, y)
            self.theta -= self.learning_rate * gradients
            self.cost_history.append(cost)
            self.theta_history.append(self.theta.copy())
            if verbose and (i % log_every == 0 or i == self.num_iterations - 1):
                print(f"Iter {i+1:>4}/{self.num_iterations} | cost = {cost:.6f}")
        return self.cost_history

    def predict(self, X):
        X_norm = self._normalize_features(X, is_training=False)
        return self._add_bias(X_norm) @ self.theta

    def score(self, X, y):
        y    = np.array(y, dtype=float).flatten()
        preds= self.predict(X)
        return np.mean((preds - y) ** 2)

    # ── plotting helpers ─────────────────────────────────────────────────────
    def plot_cost_history(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.cost_history, color='steelblue', linewidth=2)
        plt.title("Cost History")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_gradient_descent_path(self, X, y,
                                   theta0_range=(-10, 10),
                                   theta1_range=(-10, 10),
                                   resolution=50):
        if not self.theta_history:
            raise ValueError("Call fit() first.")
        X_norm = self._normalize_features(X, is_training=False)
        X_bias = self._add_bias(X_norm)
        t0_v = np.linspace(*theta0_range, resolution)
        t1_v = np.linspace(*theta1_range, resolution)
        T0, T1 = np.meshgrid(t0_v, t1_v)
        Z = np.zeros_like(T0)
        for i in range(resolution):
            for j in range(resolution):
                th = np.zeros(X_bias.shape[1])
                th[0], th[1] = T0[i, j], T1[i, j]
                Z[i, j] = self._compute_cost(X_bias @ th, y)
        path = np.array(self.theta_history)
        plt.figure(figsize=(8, 6))
        cs = plt.contour(T0, T1, Z, levels=30, cmap='viridis')
        plt.clabel(cs, inline=1, fontsize=8)
        plt.plot(path[:, 0], path[:, 1], marker='o', color='red', linewidth=2)
        plt.title("Gradient Descent Path (θ₀ vs θ₁)")
        plt.xlabel("θ₀")
        plt.ylabel("θ₁")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ── 7. Train Custom Model ────────────────────────────────────────────────────
custom_model = LinearRegressionCustom(learning_rate=0.1,
                                      num_iterations=500,
                                      cost_function='mse')
cost_history = custom_model.fit(X_train, y_train)
predictions  = custom_model.predict(X_test)

print("\nIntercept (θ₀):", custom_model.theta[0])
print("Feature weights:")
for name, weight in zip(X_train.columns, custom_model.theta[1:]):
    print(f"  {name}: {weight:.4f}")

custom_model.plot_cost_history()
custom_model.plot_gradient_descent_path(X_train, y_train)


# ── 8. Regularised Models (sklearn) ─────────────────────────────────────────
ridge_cv   = RidgeCV(alphas=np.logspace(-2, 3, 50), cv=5)
lasso_cv   = LassoCV(alphas=np.logspace(-2, 2, 50), cv=5, max_iter=10000)
elastic_cv = ElasticNetCV(alphas=np.logspace(-2, 2, 50), cv=5, max_iter=10000)

for mdl in (ridge_cv, lasso_cv, elastic_cv):
    mdl.fit(X_train_scaled, y_train)

print("\n🎯 Cross-Validation Best Alphas:")
print(f"  Ridge      : {ridge_cv.alpha_:.4f}")
print(f"  Lasso      : {lasso_cv.alpha_:.4f}")
print(f"  ElasticNet : {elastic_cv.alpha_:.4f}")

# Quick comparison on test set
models_reg = {
    "Ridge":      ridge_cv,
    "Lasso":      lasso_cv,
    "ElasticNet": elastic_cv,
}
print("\n📊 Test Performance:")
for name, m in models_reg.items():
    y_pred = m.predict(X_test_scaled)
    print(f"  {name:<12} MSE={mean_squared_error(y_test, y_pred):.2f}  "
          f"R²={r2_score(y_test, y_pred):.4f}")


# ── 9. Sigmoid Function (Logistic Regression preview) ────────────────────────
x_sig = np.linspace(-100, 100, 1000)
y_sig = 1 / (1 + np.exp(-x_sig))

plt.figure(figsize=(10, 5))
plt.plot(x_sig, y_sig, color='steelblue', linewidth=2,
         label='σ(x) = 1 / (1 + e⁻ˣ)')
plt.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
plt.axvline(0,   color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.title('Sigmoid Function')
plt.ylim(-0.05, 1.05)
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
