import numpy as np

"""
=======================================================================
Fonctions importantes
=======================================================================
"""
def calcul_r_exp(D_exp, T, eta): # EN MILLIMÈTRES
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

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

def fit2D_gaussian(X, Y, Z): # À tester...
    """
    Perform a 2D Gaussian fit on an image.
    Généré par chatgpt.

    Parameters
    ----------
    X : 1D array
        x coordinates of the grid
    Y : 1D array
        y coordinates of the grid
    Z : 2D array
        Image intensity matrix, shape (len(Y), len(X))

    Returns
    -------
    popt : list
        Best-fit parameters [A, x0, y0, sx, sy, offset]
    """

    # Create meshgrid if needed (to get coordinate matrices)
    Xg, Yg = np.meshgrid(X, Y)

    # --- 1. Initial guesses ---
    max_idx = np.unravel_index(np.argmax(Z), Z.shape)
    y0_0 = Y[max_idx[0]]  # row index → Y position
    x0_0 = X[max_idx[1]]  # column index → X position
    A0 = Z[max_idx]       # peak amplitude
    offset0 = np.min(Z)

    # --- 2. Estimate sx0 and sy0 from FWHM along x and y profiles ---
    def fwhm_to_sigma(coord, profile):
        """Estimate standard deviation from FWHM using interpolation."""
        half_max = np.max(profile) / 2.0
        imax = np.argmax(profile)

        # Left crossing
        if imax > 0:
            try:
                f_left = interp1d(profile[:imax+1], coord[:imax+1], kind='linear')
                xl = f_left(half_max)
            except ValueError:
                xl = coord[0]
        else:
            xl = coord[0]

        # Right crossing
        if imax < len(profile) - 1:
            try:
                f_right = interp1d(profile[imax:], coord[imax:], kind='linear')
                xr = f_right(half_max)
            except ValueError:
                xr = coord[-1]
        else:
            xr = coord[-1]

        fwhm = abs(xr - xl)
        return fwhm / (2 * np.sqrt(2 * np.log(2)))  # σ = FWHM / 2.35482

    # x-profile at the peak row
    xprof = Z[max_idx[0], :]
    sx0 = fwhm_to_sigma(X, xprof)

    # y-profile at the peak column
    yprof = Z[:, max_idx[1]]
    sy0 = fwhm_to_sigma(Y, yprof)

    # --- 3. Flatten data for fitting ---
    xdata = np.vstack((Xg.ravel(), Yg.ravel()))
    zdata = Z.ravel()

    # --- 4. Define 2D Gaussian model ---
    def gauss2d(coords, A, x0, y0, sx, sy, offset):
        x, y = coords
        return A * np.exp(-(((x - x0) ** 2) / (2 * sx ** 2) + ((y - y0) ** 2) / (2 * sy ** 2))) + offset

    # --- 5. Initial parameter vector ---
    p0 = [A0, x0_0, y0_0, sx0, sy0, offset0]

    # --- 6. Fit using nonlinear least squares ---
    popt, _ = curve_fit(gauss2d, xdata, zdata, p0=p0, maxfev=10000)

    return popt


X_im = np.array([0,    0.1550,    0.3100,    0.4650,    0.6200])
Y_im = np.array([0,    0.1550,    0.3100,    0.4650,    0.6200])

image2D = np.array([[0,         0,         0,         0,         0],
         [0,    0.7473,    1.0000,         0,         0],
         [0,    0.8399,    0.9715,         0,         0],
         [0,         0,         0,         0,         0],
         [0,         0,         0,         0,         0]])

#param = fit2D_gaussian(X_im, Y_im, image2D)
#print(param)
#Le fit gaussien semble bien fonctionner


from scipy.special import j1
def f(x, y, x_try, y_try, NA, lmda):
    """
    Fonction initialement écrite par Émile en matlab, puis traduite en python par chatgpt.
    La PSF.
    Output:
        psf - valeur correspondant à la convolution de la PSF du mandat avec un delta de Dirac situé en (x,y)
    """
    r = np.sqrt((x_try - x)**2 + (y_try - y)**2)
    temp = (2 * np.pi * NA * r) / lmda

    # Éviter la division par zéro
    psf = np.zeros_like(temp)
    nonzero = temp != 0
    psf[nonzero] = (2 * j1(temp[nonzero]) / temp[nonzero])**2
    psf[~nonzero] = 1.0  # Limit as temp → 0

    return psf


def generate_random_number(x, y, N_photons, NA, lmda, pixel_camera):
    """
    Generate random photon positions distributed according to the PSF.
    Fonction initialement écrite par Émile en Matlab, puis traduite en python par chatgpt.

    Parameters
    ----------
    x, y : float
        Coordinates of the real particle.
    N_photons : int
        Number of photons to generate.
    NA : float
        Numerical aperture of the optical system.
    lmda : float
        Wavelength (in same units as x, y, pixel_camera).
    pixel_camera : float
        Pixel size of the camera (same units as x and y).

    Returns
    -------
    X_rand, Y_rand : ndarray
        Arrays of random photon coordinates.
    """

    # Définir la région où les pixels sont générés (par efficacité numérique)
    n_pixel = 10
    x_min = x - n_pixel * pixel_camera
    x_max = x + n_pixel * pixel_camera
    y_min = y - n_pixel * pixel_camera
    y_max = y + n_pixel * pixel_camera

    # Valeur max de la psf 
    f_max = f(x, y, x + 1e-12, y + 1e-12, NA, lmda) # Ici la PSF!
    print("f_max =", f_max)

    X_rand = np.zeros(N_photons)
    Y_rand = np.zeros(N_photons)

    n = 0
    count = 0

    # Rejection sampling
    while n < N_photons:
        # Tirages uniformes
        x_try = x_min + (x_max - x_min) * np.random.rand()
        y_try = y_min + (y_max - y_min) * np.random.rand()

        # Tirage vertical
        u = f_max * np.random.rand()

        # Acceptation  
        if u < f(x_try, y_try, x, y, NA, lmda):
            X_rand[n] = x_try
            Y_rand[n] = y_try
            n += 1

        count += 1

        # Si y'a pas assez de points acceptés, réduire la région d'exploration.
        if count == 1000 and n <= 50:
            n_pixel = int(np.ceil(0.5 * n_pixel))
            x_min = x - n_pixel * pixel_camera
            x_max = x + n_pixel * pixel_camera
            y_min = y - n_pixel * pixel_camera
            y_max = y + n_pixel * pixel_camera
            count = 0

    return X_rand, Y_rand


def real_image(x, y, X_im, Y_im, NA, lmda, N_photons, pixel_camera, n_pixel_camera):
    """
    Fonction initialement écrite par Émile, puis traduite en python par chatgpt.
    Paramètres sortie : image2D - Maillage 2D du nombre de photons par pixel
    Paramètres entrée :
        x et y - Coordonnées réelles de la particule.
        X_im et Y_im donnent les positions en x et en y sur
        le détecteur sous forme de maillage 2D.
        NA est l'ouverture numérique du système optique;
        lmda est la longueur d'onde du rayon incident;
        N_photons - Nombre de photons par image
    """

    # Génération des coordonnées des photons aléatoires
    X_rand, Y_rand = generate_random_number(x, y, N_photons, NA, lmda, pixel_camera)

    # Initialisation de l'image (matrice de photons par pixel)
    image2D = np.zeros((n_pixel_camera[1], n_pixel_camera[0]))

    for i in range(N_photons):
        x_i = X_rand[i]
        y_i = Y_rand[i]

        # Pixel associé à la position
        index_x = int(np.floor(x_i / pixel_camera + 0.5))
        index_y = int(np.floor(y_i / pixel_camera + 0.5))

        # Vérifie que l’indice est dans les limites de l’image
        if 0 <= index_x < n_pixel_camera[0] and 0 <= index_y < n_pixel_camera[1]:
            image2D[index_y, index_x] += 1

    # Normalisation et ajout d'un bruit de Poisson
    image_max = np.max(image2D)
    if image_max > 0:
        image2D = image2D / image_max

    # Ajout d’un bruit de Poisson (comme imnoise(image2D, "poisson"))
    image2D = np.random.poisson(image2D * np.max(image2D))

    return image2D

def brownien(x1, y1, D_real, delta_t):
    """
    Définition de la fonction "brownien" :
        Paramètres sortie : x2 et y2 les coordonnées réelles de la particule.
        Paramètres entrée : x1 et y1 les coordonnées initiales de la particule;
                            D_real le coefficient de diffusion théorique;
                            delta_t le temps entre chaque frame;
    """

    # Calculer les déplacements aléatoires en fonction du coefficient de diffusion
    sigma = np.sqrt(2 * D_real * delta_t)  # Écart type du mouvement brownien
    x2 = x1 + sigma * np.random.randn()    # Nouvelle position x
    y2 = y1 + sigma * np.random.randn()    # Nouvelle position y

    return x2, y2


"""
=======================================================================
Début du code principal : 
=======================================================================
"""

import matplotlib.pyplot as plt

# Paramètres de la particule
N_step = 50  # Nombre de pas de temps par marche aléatoire
N_marches = 10  # Nombre de marches aléatoires par valeur de paramètre
N_param = 10  # Nombre de valeurs du paramètre testées. i.e. pour r = 0.1, 1, 10 : N_param = 3
N_photons = 1000  # Nombre de photons reçu par la caméra par position de particules
r_real = 1E-3  # Taille réelle de la particule (en mm)

# Constantes physiques
k_b = 1.380649E-23  # Constante de Boltzmann (J/K)
T = 293  # Température absolue du fluide (K)
eta = 1E-3  # Viscosité dynamique du fluide
# La viscosité dynamique de l'eau est environ 0,001 Pa * s à 20°C.
D_real = (1E6) * (k_b * T) / (6 * np.pi * eta * (r_real * 10 ** (-3)))  # Coefficient de diffusion (mm^2/s)

# Paramètres de la caméra
delta_t = 0.1e-3  # Délai entre chaque frame (1 ms entre chaque frame?)
grossissement = 20  # Magnification du système optique
NA = 1.33  # Ouverture numérique
lmda = 500e-6  # Longueur d'onde captée (mm) --> 500 nm (vert)
pixel_camera = (1.55E-3) / 50  # Taille du pixel (mm)
n_pixels_camera = np.array([50, 50])  # Dimensions du détecteur (px)
# VRAIE VALEUR : [4056;3040] À REMETTRE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
pixel_objet = pixel_camera / grossissement  # Taille du pixel dans l'espace objet

# Maillage dans l'espace image
x_im = np.arange(0, n_pixels_camera[0]) * pixel_camera
y_im = np.arange(0, n_pixels_camera[1]) * pixel_camera
X_im, Y_im = np.meshgrid(x_im, y_im)

# Initialisation des positions
x_positions = np.zeros(N_step)
y_positions = np.zeros(N_step)
x_positions[0] = n_pixels_camera[0] / 2 * pixel_camera
y_positions[0] = n_pixels_camera[1] / 2 * pixel_camera

x_guess = np.zeros(N_step)
y_guess = np.zeros(N_step)

r_val_k = np.zeros(N_marches)
valeurs_k = np.zeros(N_param)
vec_r_tot = np.zeros(N_param)
vec_std_r = np.zeros(N_param)

for k in range(1, N_param + 1):  # Pour chaque valeur du paramètre testé
    # Opération sur le paramètre. Par exemple, pour le temps :
    delta_t_k = delta_t * k

    for j in range(N_marches):  # Pour toutes les marches, pour une valeur du paramètre testé
        for i in range(1, N_step):  # Pour une marche aléatoire complète
            # Calculer une position
            x_positions[i], y_positions[i] = brownien(x_positions[i - 1], y_positions[i - 1], D_real, delta_t_k)

            # Création d'une image réelle
            print("Commence real_image")
            image2D = real_image(x_positions[i], y_positions[i], X_im, Y_im, NA, lmda, N_photons, pixel_camera, n_pixels_camera)
            print("Finito real_image")

            # Affichage 3D (optionnel, équivalent à bar3(image2D))
            # plt.figure()
            # ax = plt.axes(projection='3d')
            # ax.plot_surface(X_im, Y_im, image2D, cmap='viridis')
            # plt.show()

            # Localisation de la particule
            param = fit2D_gaussian(x_im, y_im, image2D)
            print("param :", param)
            x_guess[i] = param[1]
            y_guess[i] = param[2]

        pos_guess = np.column_stack((x_guess, y_guess))
        res_msd = compute_msd(pos_guess)  # Par défaut : max_lag = N/4
        # res_msd = [taus, msd] des vecteurs colonnes.

        D_exp = fit_msd_linear(res_msd[0], res_msd[1], delta_t_k)  # Pour une marche aléatoire complète
        r_exp = calcul_r_exp(D_exp, T, eta)  # Pour une seule marche aléatoire complète
        r_val_k[j] = r_exp

    vec_r_tot[k - 1] = np.mean(r_val_k)
    vec_std_r[k - 1] = np.std(r_val_k)
    valeurs_k[k - 1] = delta_t_k  # Pour le temps

# Reste plus qu'à plot les résultats : vec_r_tot vs valeurs_k , avec incertitudes en y vec_std_r
plt.errorbar(valeurs_k, vec_r_tot, yerr=vec_std_r, fmt='o', linewidth=1.2, capsize=8)
plt.xlabel('delta_t (s)')
plt.ylabel('r_exp (mm)')
plt.title('Estimation de r = 1um, pour plusieurs delta_t')
plt.show()

