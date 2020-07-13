import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from yellowbrick import regressor

from linearModelExample import x, y


ax = sns.regplot(x, y, scatter = True, fit_reg = True, ci = None, truncate = False,
    scatter_kws = {'color': 'blue'}, line_kws = {'color': 'red'})
ax.set(xlabel = 'X values', ylabel = 'Y values')
plt.savefig('../figures/linearModelExample.pdf')
# plt.show()
plt.close()

ax = sns.regplot(x, y, scatter = True, fit_reg = True, ci = 95, truncate = False,
    scatter_kws = {'color': 'blue'}, line_kws = {'color': 'red'})
ax.set(xlabel = 'X values', ylabel = 'Y values')
plt.savefig('../figures/linearModelExample_withPrediction.pdf')
# plt.show()
plt.close()

ax = sns.regplot(x, y, scatter = True, fit_reg = False, truncate = False,
    scatter_kws = {'color': 'blue'}, line_kws = {'color': 'red'})
ax.set(xlabel = 'X values', ylabel = 'Y values')
plt.legend()
plt.savefig('../figures/data.pdf')
# plt.show()
plt.close()


x_new = np.append(x, [20])
y_new = np.append(y, [14])
ax = sns.regplot(x_new, y_new, scatter = True, fit_reg = True, ci = None, truncate = False,
    scatter_kws = {'color': 'blue'}, line_kws = {'color': 'red'})
ax.set(xlabel = 'X values', ylabel = 'Y values')
plt.savefig('../figures/linearModelExample_with_influential_points.pdf')
# plt.show()
plt.close()


# regressor.cooks_distance(x_new.reshape(-1, 1), y_new, draw_threshold = True)
ax = plt.gca()
reg = regressor.CooksDistance(ax = ax)
reg.fit(x_new.reshape(-1, 1), y_new)
ax.set_xlabel('Instance index')
ax.set_ylabel('Influence')
plt.legend()
plt.savefig('../figures/cooks_distance.pdf')
plt.close()
