# %%
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression, Lasso

# %%
X, y = make_regression(n_features=5, n_targets=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
reg = Ridge().fit(X_train, y_train)

# %%
reg.intercept_

# %%
reg.coef_

# %%
reg = LinearRegression().fit(X_train, y_train)

# %%
reg.intercept_

# %%
reg.coef_

# %%
reg = Lasso().fit(X_train, y_train)

# %%
reg.intercept_

# %%
reg.coef_

# %%
