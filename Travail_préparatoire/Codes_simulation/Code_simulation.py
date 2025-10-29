import numpy as np

"""
=======================================================================
Fonctions importantes
=======================================================================
"""
def r_exp(D_exp, T, eta): # EN MILLIMÈTRES
    # Calculer r avec Stokes-Einstein
    k_b = 1.380649E-23
    r_exp = (1E3)*(k_b*T)/(6*np.pi*eta*(D_exp*10^(-6))) # r_exp en mm, D_exp originalement en mm^2 / s.
    return r_exp


def compute_msd(positions): # À valider
    """
    Input : positions (une matrice N par 2 contenant les coordonnées de tous les fits gaussiens)
            max_lag (saut maximal entre des frames subséquents, en entiers)
    Output : results_msd = [taus, msd] --> deux vecteur colonnes.
             taus : vecteur contenant les entiers des lag times, allant de 1 à max_lag. Axe x du graphique de MSD.
             msd : vecteur contenant les valeurs du msd, correspond aux valeurs y du graphique de MSD
    """
    N = positions.shape[0] # Nb de lignes --> de positions dans la marche aléatoire complète
    max_lag = N//4
    msd = np.zeros(max_lag) # Vecteur colonne

    for lag in range(1, max_lag + 1):  # lags go from 1 to max_lag inclusive
        diffs = positions[lag:, :] - positions[:-lag, :]  # shape: (N - lag, 2)
        sq = np.sum(diffs ** 2, axis=1)  # squared displacements
        msd[lag - 1] = np.mean(sq)

    taus = np.arange(1, max_lag + 1)
    return taus, msd



def fit_msd_linear(taus, msd, dt):
    """
    Fonction qui fait une régression linéaire de MSD(t) = 2*dim*D*t + C
    """

    dim = 2  # On évalue en xy
    t = taus * dt
    X = np.column_stack((t, np.ones_like(t)))  # Initialize matrix for least squares. Each row is [t_i, 1]
    beta, _, _, _ = np.linalg.lstsq(X, msd, rcond=None)  # beta is the least-squares estimate: beta[0] = slope, beta[1] = intercept
    pente = beta[0]  # intercept = beta[1]
    D_exp = pente / (2 * dim)

    # If more information about the linear fit is needed:
    # n = len(msd)
    # if n > 2:
    #     residuals = msd - X @ beta
    #     sigma2 = np.sum(residuals ** 2) / (n - 2)
    #     cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    # else:
    #     cov_beta = np.full((2, 2), np.nan)
    #
    # if not np.isnan(cov_beta).any():
    #     se_slope = np.sqrt(cov_beta[0, 0])
    #     se_D = se_slope / (2 * dim)
    # else:
    #     se_D = np.nan

    return D_exp

"""
=======================================================================
Début du code principal : 
=======================================================================
"""

# Écrire le code complet en python ici!!!
