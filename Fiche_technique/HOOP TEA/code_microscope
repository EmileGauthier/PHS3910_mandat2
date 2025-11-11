import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max
from scipy.interpolate import interp1d
from scipy import stats

"""
=======================================================================
Fonctions importantes
=======================================================================
"""
def fit2D_gaussian(X, Y, Z):
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
    A0 = 47     # peak amplitude
    offset0 = 20

    # x-profile at the peak row
    xprof = Z[max_idx[0], :]
    sx0 = 5
    #sx0 = fwhm_to_sigma(X, xprof)

    # y-profile at the peak column
    yprof = Z[:, max_idx[1]]
    #sy0 = fwhm_to_sigma(Y, yprof)
    sy0 = 5

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

def compute_msd(positions):
    """
    Input : positions (une matrice N par 2 contenant les coordonnées de tous les fits gaussiens)
            max_lag (saut maximal entre des frames subséquents, en entiers)
    Output : results_msd = [taus, msd] --> deux vecteur colonnes.
             taus : vecteur contenant les entiers des lag times, allant de 1 à max_lag. Axe x du graphique de MSD.
             msd : vecteur contenant les valeurs du msd, correspond aux valeurs y du graphique de MSD
    """

    max_lag = 8
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
    slope = beta[0]  
    intercept = beta[1]
    D_exp = slope / (2 * dim)

    plot_msd_fit(taus,msd,dt,slope,intercept)
    incertitude = incertitude_pente(taus,msd,slope, intercept)  / (2 * dim)

    return D_exp, incertitude

def incertitude_pente(taus, msd, slope, intercept):
    N = len(taus)
    # Moyennes
    moy_t = np.mean(taus)
    
    # Variance résiduelle
    residuals = msd - (slope * taus + intercept)
    sigma_y2 = np.sum(residuals**2) / (N - 2)
    
    # Dénominateur de la pente
    Sxx = np.sum((taus - moy_t)**2)
    
    # Erreur-type sur la pente
    sigma_slope = np.sqrt(sigma_y2 / Sxx)
    return sigma_slope

def calcul_r_exp(D_exp, T, eta):
    # Calculer r avec Stokes-Einstein
    k_b = 1.380649E-23
    r_exp = (k_b*T)/(6*np.pi*eta*(D_exp)) # r_exp en m, D_exp en m^2 / s.
    return r_exp

def index_max_image(nom_fichier):
    # Fonction qui retourne les indices du maximum d'une image

    # Ouverture de l'image
    image2D = plt.imread(nom_fichier)

    # Conversion de l'image de RGBA à gris
    image2D = cv2.cvtColor(image2D, cv2.COLOR_RGBA2GRAY)

    # Pour réduire la qualité de l'image
    image2D = image2D[::1, ::1]

    N = 50

    coordinates = np.add(peak_local_max(image2D[N:-N, N:-N], exclude_border=False, min_distance=30,threshold_rel=0.5), N)
    
    return coordinates[0,:]

def plot_msd_fit(taus, msd, dt, slope, intercept):
    """
    Plot MSD points and their linear fit.
    
    Parameters
    ----------
    taus : array-like
        Time lags (indices or unitless).
    msd : array-like
        Mean square displacement values.
    dt : float
        Time step between frames.
    slope : float
        Slope from the linear fit.
    intercept : float
        Intercept from the linear fit.
    """
    t = taus * dt
    msd_fit = slope * t + intercept

    plt.figure(figsize=(6, 4))
    plt.plot(t, msd*1e12, 'o', label='Déplacement quadratique moyen (MSD)')
    plt.plot(t, msd_fit*1e12, '-', label=f'Régression linéaire')
    plt.xlabel('Temps (s)')
    plt.ylabel(r'MSD ($\mu\text{m}^2$)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

"""
=======================================================================
Début du code principal : 
=======================================================================
"""

# Constantes physiques
k_b = 1.380649e-23  # Constante de Boltzmann (J/K)
T = 293  # Température absolue du fluide (K)
eta = 1.01e-3  # Viscosité dynamique du fluide (Pa s)

# Paramètres de la caméra
delta_t = 0.65  # Délai entre chaque frame (s)
N_images = 156 # Nombre d'images 
pixel_camera = 0.377e-6 # Taille du pixel (m)
grossissement = 1.55e-6 / pixel_camera

# Paramètres de performance
N_pixel = 50 # Détermine la taille de la région d'intérêt dans les images. 

# Initialisation des array pour les positions estimées
x_guess = np.zeros(N_images)
y_guess = np.zeros(N_images)

# Position du maximum de la première image, pour zoom sur une région d'intérêt
index_max = index_max_image("C:\\Users\\gauth\\OneDrive\\Documents\\Polytechnique\\Polytechnique automne 2025\\Techniques expérimentales et instrumentation\\Mandat 2\\code_microscope\\timelapse_2028x1080_1fps\\img_0000.jpg")

for i in range(N_images):

    # Ouverture de l'image
    image2D = plt.imread("C:\\Users\\gauth\\OneDrive\\Documents\\Polytechnique\\Polytechnique automne 2025\\Techniques expérimentales et instrumentation\\Mandat 2\\code_microscope\\timelapse_2028x1080_1fps\\img_{:04}.jpg".format(i))

    # Affichage de l'image brut
    #plt.imshow(image2D)
    #plt.show()

    # Conversion de l'image de RGBA à gris
    image2D = cv2.cvtColor(image2D, cv2.COLOR_RGBA2GRAY)

    # Pour réduire la qualité de l'image (le chiffre après les deux :: est le facteur de réduction)
    image2D = image2D[::1, ::1]

    # Sélectionner la partie de l'image qui nous intéresse
    image2D = image2D[(index_max[0] - N_pixel):(index_max[0] + N_pixel),(index_max[1] - N_pixel):(index_max[1] + N_pixel)]

    # Affichage de l'image découpée
    #plt.imshow(image2D)
    #plt.show()

    # Maillage
    x_im = np.arange(0, len(image2D[:,0])) # On laisse les unités en pixel pour faciliter le fit gaussien
    y_im = np.arange(0, len(image2D[0,:]))
    X_im, Y_im = np.meshgrid(x_im, y_im)
    
    # Localisation de particules
    try:
        popt = fit2D_gaussian(x_im,y_im,image2D)
        x_guess[i] = popt[1]
        y_guess[i] = popt[2]
    except:
        x_guess[i] = x_guess[i-1]
        y_guess[i] = y_guess[i-1]

# Affichage de la trajectoire de la particule
plt.imshow(image2D)
plt.plot(y_guess,x_guess, color= 'red')
plt.show()

# Affichage du déplacement en fonction des images
#plt.plot(np.sqrt(np.power(x_guess,2) + np.power(y_guess,2)))
#plt.show()

# Calcul de la MSD
pos_guess = np.column_stack((x_guess*pixel_camera, y_guess*pixel_camera)) # On transforme les unités en mètre
res_msd = compute_msd(pos_guess)

# Fit linéaire de la MSD pour obtenir le coefficient D_exp et ensuite le rayon r_exp
D_exp, inc_D = fit_msd_linear(res_msd[0], res_msd[1], delta_t) 

print(f"La valeur du coefficient de diffusion est {D_exp} m^2/s et son incertitude est {inc_D}.")

r_exp = calcul_r_exp(D_exp, T, eta) 

inc_r = (inc_D/D_exp) * r_exp

print(f"Le rayon estimé de la particule est de {1e6*r_exp} um et son incertitude est {1e6*inc_r} um.")
