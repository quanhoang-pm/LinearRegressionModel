# slope_intercept_distribution.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from samples_generator import (
    slopeValuesList, theoreticalSlopeExpectedValue, theoreticalSlopeVariance,
    interceptValuesList, theoreticalInterceptExpectedValue, theoreticalInterceptVariance,
    nTrials,
)

nBins = int(nTrials ** (1 / 3))
# distribution of estimated slope
scale = np.sqrt(theoreticalSlopeVariance)
minValue = theoreticalSlopeExpectedValue - 3 * scale
maxValue = theoreticalSlopeExpectedValue + 3 * scale
x = np.linspace(minValue, maxValue, 100)
y = stats.norm.pdf(x, loc = theoreticalSlopeExpectedValue, scale = scale)

sns.set()
fig, ax = plt.subplots(1, 1, figsize = (5, 5))
ax = sns.distplot(slopeValuesList, ax = ax, hist = True, norm_hist = True, kde = False, bins = nBins, color = 'blue', label = 'Samples distribution')
ax.plot(x, y, lw = 1, color = 'r', label = 'Theoretical distribution')
maxProbability = stats.norm.pdf(theoreticalSlopeExpectedValue, loc = theoreticalSlopeExpectedValue, scale = scale) + 9
ax.set_ylim(0, maxProbability)
plt.legend(loc = 'best')
plt.savefig('../figures/slope_distribution.pdf')
plt.close()

# distribution of estimated intercept
scale = np.sqrt(theoreticalInterceptVariance)
minValue = theoreticalInterceptExpectedValue - 3 * scale
maxValue = theoreticalInterceptExpectedValue + 3 * scale
x = np.linspace(minValue, maxValue, 100)
y = stats.norm.pdf(x, loc = theoreticalInterceptExpectedValue, scale = scale)

fig, ax = plt.subplots(1, 1, figsize = (5, 5))
ax = sns.distplot(interceptValuesList, ax = ax, hist = True, norm_hist = True, kde = False, bins = nBins, color = 'blue', label = 'Samples distribution')
ax.plot(x, y, lw = 1, color = 'r', label = 'Theoretical distribution')
maxProbability = stats.norm.pdf(theoreticalInterceptExpectedValue, loc = theoreticalInterceptExpectedValue, scale = scale) + 0.7
ax.set_ylim(0, maxProbability)
plt.legend(loc = 'best')
plt.savefig('../figures/intercept_distribution.pdf')
plt.close()
