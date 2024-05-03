# Simulation-loi-normale-avec-Python

Ce projet vise  à explorer la Loi Normale univariée et bidimensionnelle ainsi que les méthodes d'estimation associées, en se basant sur le code Python fourni. Le code utilise les bibliothèques NumPy, Matplotlib et SciPy pour la génération de données aléatoires, la représentation graphique et le calcul des statistiques.

## 1. Loi Normale Univariée :

La distribution univariée de la Loi Normale est illustrée dans la première partie du code. Cette distribution est caractérisée par deux paramètres : la moyenne (μ) et l'écart type (σ), définis respectivement comme mu[0] et np.sqrt(cov[0, 0]) dans le code.

Génération des données : Un échantillon de données est généré à partir de la Loi Normale univariée à l'aide de la fonction np.random.normal(). Les données sont ensuite représentées sous forme d'histogramme pour visualiser leur répartition.

Estimation de la densité de probabilité : La densité de probabilité de la Loi Normale est calculée et tracée sur le même graphique. Cette densité est représentée par une courbe lisse, montrant la forme caractéristique en cloche de la distribution.

## 2. Loi Normale Bidimensionnelle :

La distribution bidimensionnelle de la Loi Normale est explorée dans la seconde partie du code. Cette distribution implique une moyenne bidimensionnelle (μ) et une matrice de covariance (Σ), définies respectivement comme mu et cov.

Génération des données : Un ensemble de points est généré à partir de la Loi Normale bidimensionnelle à l'aide de np.random.multivariate_normal(). Ces points sont ensuite représentés sur un graphique en nuage de points.

Estimation de la densité de probabilité : La densité de probabilité de la Loi Normale bidimensionnelle est estimée à partir des échantillons de données. Cette estimation est effectuée en calculant la moyenne et la covariance des données échantillonnées, puis en utilisant ces estimations pour calculer la densité de probabilité.

Ellipses de densité : Les ellipses de densité sont tracées pour différentes valeurs de probabilité (p). Ces ellipses représentent les contours de densité de probabilité autour des points moyens, permettant de visualiser la dispersion des données.

## 3. Convergence de l'Estimation :

Le code démontre également comment l'estimation des paramètres de la distribution converge avec l'augmentation de la taille de l'échantillon. Pour différentes tailles d'échantillons (50, 200, 1000, 5000), les paramètres estimés (moyenne et covariance) convergent vers les valeurs réelles de la distribution.

Comparaison avec la réalité : Les ellipses de densité estimées sont comparées aux ellipses de densité réelles, permettant de visualiser la précision de l'estimation en fonction de la taille de l'échantillon.
