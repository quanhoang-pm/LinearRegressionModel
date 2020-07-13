# qq_plot.py
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


from linearModelExample import x, y, b0, b1


residuals = y - (b0 + b1 * x)
sns.set()
fig, ax = plt.subplots(1, 1, figsize = (5, 5))
stats.probplot(residuals, dist = 'norm', plot = ax)
ax.figure.subplots_adjust(left = 0.17)
ax.set(xlabel = 'Quantiles of N(0,1)', ylabel = 'Quantiles of residuals', title = 'Quantile - quantile plot')
plt.savefig('../figures/QQplot.pdf')
plt.close()
