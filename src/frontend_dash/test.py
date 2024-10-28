import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# Sample data generation
np.random.seed(42)
full_data = np.random.normal(loc=0, scale=1, size=1000)
sampled_data = np.random.choice(full_data, size=300, replace=False)

# KDE computation for both datasets
kde_full = gaussian_kde(full_data)
kde_sampled = gaussian_kde(sampled_data)

# Generate points for plotting
x_vals = np.linspace(min(full_data), max(full_data), 1000)
kde_full_vals = kde_full(x_vals)
kde_sampled_vals = kde_sampled(x_vals)

# Calculate overlap density (scaling sampled KDE by its proportion in full data)
sampled_proportion = len(sampled_data) / len(full_data)
overlap_density = kde_sampled_vals * sampled_proportion

# Plotting with Plotly
fig = go.Figure()

# Full data KDE line
fig.add_trace(go.Scatter(x=x_vals, y=kde_full_vals, mode='lines', name='Full Data Density',
                         line=dict(color='blue', width=2)))

# Sampled data KDE line (dotted)
fig.add_trace(go.Scatter(x=x_vals, y=kde_sampled_vals, mode='lines', name='Sampled Data Density',
                         line=dict(color='green', width=2, dash='dot')))

# Overlap density area
fig.add_trace(go.Scatter(x=x_vals, y=overlap_density, fill='tozeroy', name='Overlap Density',
                         line=dict(color='orange', width=2)))

# Figure layout
fig.update_layout(title="Density Distributions with Overlap",
                  xaxis_title="Value",
                  yaxis_title="Density",
                  showlegend=True)

fig.show()
