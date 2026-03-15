import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
data1 = np.random.normal(245, 1, 1000)
data2 = np.random.normal(243, 1, 1000)

df1 = pd.DataFrame({'price': data1})
df2 = pd.DataFrame({'price': data2})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: histplot
sns.histplot(df1.price, ax=axes[0], label='Buy', kde=False, linewidth=0, alpha=0.5)
sns.histplot(df2.price, ax=axes[0], label='Sell', kde=False, linewidth=0, alpha=0.5)
axes[0].set_title('histplot')

# Plot 2: distplot
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    sns.distplot(df1.price, ax=axes[1], label='Buy', kde=False, hist_kws={'linewidth': 1, 'alpha': .5})
    sns.distplot(df2.price, ax=axes[1], label='Sell', kde=False, hist_kws={'linewidth': 1, 'alpha': .5})
axes[1].set_title('distplot')

plt.savefig('test_hist.png')
