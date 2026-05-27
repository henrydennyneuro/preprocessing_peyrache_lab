import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Hard-coded session ---
csv_path = r'D:\B3200\B3210\B3210-240909\B3210-240909_mean_wf.csv'
n_neurons_to_plot = 5
# --------------------------

df = pd.read_csv(csv_path, index_col=[0, 1])
df.index.names = ['neuron', 'time_s']
df.columns = df.columns.astype(int)

neurons   = df.index.get_level_values('neuron').unique()
last5     = neurons[-n_neurons_to_plot:]
time_ms   = df.loc[last5[0]].index.values * 1000

fig, axes = plt.subplots(n_neurons_to_plot, 1, sharex=True, figsize=(8, 10))

for ax, neuron_id in zip(axes, last5):
    wf = df.loc[neuron_id].values        # shape: (n_timepoints, n_channels)
    ax.plot(time_ms, wf, color='steelblue', alpha=0.25, linewidth=0.8)
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f'Neuron {neuron_id}')
    ax.axvline(0, color='k', linewidth=0.5, linestyle='--')

axes[-1].set_xlabel('Time (ms)')
plt.suptitle(f'Last {n_neurons_to_plot} neurons — all channels', y=1.01)
plt.tight_layout()
plt.show()
