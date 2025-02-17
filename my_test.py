
# %%
# import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True, as_frame=True, scaled=True)

# %%
reg = LinearRegression()
reg.fit(X, y)

# %%
reg.coef_

# %%
reg.feature_names_in_

# %%
import pandas as pd

df = (
    pd.DataFrame([reg.intercept_]+list(reg.coef_), index=["intercept"]+list(reg.feature_names_in_))
)
df.index.name = "Feature"
df.columns = ["Coefficient"]
df
# %%
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
reg.score(X, y)
reg.coef_
reg.intercept_
reg.predict(np.array([[3, 5]]))

# %%
import pandas as pd

feature_names = reg.feature_names_in_ if hasattr(reg, "feature_names_in_") else range(reg.n_features_in_)

df = (
    pd.DataFrame([reg.intercept_]+list(reg.coef_), index=["intercept"]+list(feature_names))
)
df.index.name = "Feature"
df.columns = ["Coefficient"]
df

# %%
