import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np

# Create figure and styling for plotting
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.set(xlabel='dimensions (m)', ylabel='log(dmax/dmin)', title='dmax/dmin vs. dimensionality')
line_styles = {0: 'ro-', 1: 'b^-', 2: 'gs-', 3: 'cv-'}

# Plot dmax/dmin ratio
# TODO: fill in valid test numbers
valid_test_numbers = [100, 250, 500, 1000]
for idx, num_samples in enumerate(valid_test_numbers):
    # TODO: Fill in a valid feature range -> dimensionality from 1 to 100 per project spec
    feature_range = range(1, 101)
    ratios = []
    for num_features in feature_range:
        # TODO: Generate synthetic data using make_classification
        X, _ = make_classification(n_samples=num_samples, n_features=num_features, n_informative=num_features,
                                   n_redundant=0, n_clusters_per_class=1, random_state=42)
        
        # TODO: Choose random query point from X
        query_point_idx = np.random.choice(X.shape[0])
        query_point = X[query_point_idx]
        
        # TODO: remove query pt from X so it isn't used in distance calculations
        remaining_points = np.delete(X, query_point_idx, axis=0)

        # TODO: Calculate distances
        distances = np.linalg.norm(remaining_points - query_point, axis=1)
        ratio = np.max(distances) / np.min(distances)
        ratios.append(ratio)

    ax.plot(feature_range, np.log(ratios), line_styles[idx], label=f'N={num_samples:,}')

plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()