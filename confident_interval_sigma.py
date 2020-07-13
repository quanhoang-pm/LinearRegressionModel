import matplotlib.pyplot as plt
import seaborn as sns

from samples_generator import ratiosList, CI_ratio, nTrials

sns.set()
fig, ax = plt.subplots(1, 1, figsize = (5, 5))
sns.lineplot(range(1, nTrials + 1), ratiosList, ax = ax, label = f'Proportion of correct {CI_ratio*100}% CIs', zorder = 5)
sns.lineplot([0, nTrials], [CI_ratio, CI_ratio], ax = ax, label = 'Expected proportion', lw = 2, zorder = 2)

# ax.figure.tight_layout()
ax.figure.subplots_adjust(left = 0.16)
ax.set_ylim(0.9, 1.002)
ax.set(xlabel = 'Number of samples', ylabel = 'Ratio')
plt.legend(loc = 'best')
plt.savefig('../figures/CI_sigma.pdf')
plt.close()
