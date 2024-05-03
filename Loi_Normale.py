import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats


# Definition of the distribution
mu = np.array([1, 2])
cov = np.array([[0.25, 0.3], [0.3, 0.75]])

#Part1: Distribution loi normale univarié

# Generate sample data
sample_data_uni = np.random.normal(loc=mu[0], scale=np.sqrt(cov[0, 0]), size=1000)
print(sample_data_uni)

# Plot histogram
plt.figure(figsize=(6, 4))
plt.hist(sample_data_uni, bins=30, density=True, alpha=0.6, color='b')

# Plot normal distribution PDF
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1000)
p = stats.norm.pdf(x, mu[0], np.sqrt(cov[0, 0]))
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu[0], np.sqrt(cov[0, 0]))
plt.title(title)

plt.show()


#Part2: Distribution loi normale bidimentionnelle

#part1

def fz(z):
    return (1 / (np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov)))) * np.exp(
        (-0.5) * (z - mu).dot(np.linalg.inv(cov)).dot((z - mu).T))


# Sample points
samples = np.random.multivariate_normal(mu, cov, size=1000)
xmin = np.min(samples[:, 0]) - 0.5
xmax = np.max(samples[:, 0]) + 0.5
ymin = np.min(samples[:, 1]) - 0.5
ymax = np.max(samples[:, 1]) + 0.5
xscale = np.arange(xmin, xmax, 0.01)
yscale = np.arange(ymin, ymax, 0.01)

# Probability density calculation
z = np.zeros((len(yscale), len(xscale)))
for i, x in enumerate(xscale):
    for j, y in enumerate(yscale):
        z[j, i] = fz(np.array([x, y]))


# Definition of an ellipse
def ellipse(p, color):
    plt.contour(xscale, yscale, z, levels=[(1 - p) / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov))], colors=color,
                linewidths=0.5)


# Display points and ellipses for p = 0.99 / 0.9 / 0.5 / 0.1
plt.figure(num='Densité')
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3)
ellipse(0.99, 'blue')
ellipse(0.9, 'green')
ellipse(0.5, 'red')
ellipse(0.1, 'orange')

# Create legend for ellipses
legend_patches = [
    mpatches.Patch(color='blue', label='p = 0.99'),
    mpatches.Patch(color='green', label='p = 0.9'),
    mpatches.Patch(color='red', label='p = 0.5'),
    mpatches.Patch(color='orange', label='p = 0.1')
]
plt.legend(handles=legend_patches)

plt.title("Sample points and density ellipses")


#part 2
# Estimated density function
def fz_estimated(z, mu_estimated, cov_estimated):
    return (1 / (np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov_estimated)))) * np.exp(
        (-0.5) * (z - mu_estimated).dot(np.linalg.inv(cov_estimated)).dot((z - mu_estimated).T))


# Definition of an ellipse based on the sample and covariance
def ellipse2(ax, x, y, z1, z2, p, color, cov1, cov2):
    # Ellipse with the real z and given covariance
    ax.contour(x, y, z1, levels=[(1 - p) / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov1))], colors=color,
               linewidths=0.5)

    # Ellipse with the estimated z and estimated covariance
    contour = ax.contour(x, y, z2, levels=[(1 - p) / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov2))], colors=color,
                         linewidths=0.5)
    for contour_path in contour.collections:
        contour_path.set_linestyle('dashed')

    # Create 2 invisible lines to be able to make the legend
    ax.plot([], [], color=color, linewidth=0.5, linestyle='dashed', label='Estimation')
    ax.plot([], [], color=color, linewidth=0.5, linestyle='solid', label='Real')


# Create the window
fig, axes = plt.subplots(2, 2, figsize=(8, 8), )

# Loop for each subplot
for i, ax in enumerate(axes.flatten()):

    # Definition of the different sample points
    n_points = [50, 200, 1000, 5000][i]

    # Calculate the points
    samples = np.random.multivariate_normal(mu, cov, n_points)
    xmin = np.min(samples[:, 0]) - 0.5
    xmax = np.max(samples[:, 0]) + 0.5
    ymin = np.min(samples[:, 1]) - 0.5
    ymax = np.max(samples[:, 1]) + 0.5
    xscale = np.arange(xmin, xmax, 0.01)
    yscale = np.arange(ymin, ymax, 0.01)

    # Estimation of the distribution parameters from the points
    mu_estimated = np.mean(samples, axis=0)
    cov_estimated = np.cov(samples.transpose())

    # Calculate probability densities
    z_real = np.zeros((len(yscale), len(xscale)))
    z_estimated = np.zeros((len(yscale), len(xscale)))
    for i, x in enumerate(xscale):
        for j, y in enumerate(yscale):
            z_real[j, i] = fz(np.array([x, y]))
            z_estimated[j, i] = fz_estimated(np.array([x, y]), mu_estimated, cov_estimated)

    # Display points and ellipses for p = 0.1 / 0.99
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3)
    ellipse2(ax, xscale, yscale, z_real, z_estimated, 0.1, 'blue', cov, cov_estimated)
    ellipse2(ax, xscale, yscale, z_real, z_estimated, 0.99, 'red', cov, cov_estimated)
    ax.set_title(f'Sample size {n_points}')

# Display legend, title and 2 windows
axes[0, 0].legend(loc='upper left')
plt.suptitle("Convergence of μ and Σ estimators")
plt.show()