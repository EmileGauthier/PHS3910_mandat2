import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max

"""
=======================================================================
Fonctions importantes
=======================================================================
"""
def fit2D_multiple_gaussians(X_im,Y_im,image2D):
    # Fonction décrivant une gaussienne 2D
    def gaussian2d(xy, x0, y0, sigma_x, sigma_y, A, offset):
        (x, y) = xy
        return offset + A * np.exp(
            -(((x - x0)**2) / (2 * sigma_x**2) + ((y - y0)**2) / (2 * sigma_y**2))
        )

    # Fonction décrivant la somme de plusieurs gaussienne 2D
    def multi_gaussian2d(xy, *params):
        (x, y) = xy
        z = np.zeros_like(x, dtype = float)
        n = len(params) // 6  # 6 paramètres par gaussienne
        for i in range(n):
            x0, y0, sigma_x, sigma_y, A, offset = params[6*i:6*(i+1)]
            z += gaussian2d((x, y), x0, y0, sigma_x, sigma_y, A, offset)
        return z.ravel()

    # Identification des maximums locaux
    coordinates = peak_local_max(image2D, exclude_border=False, min_distance=30,threshold_rel=0.5)

    x_max = np.zeros(len(coordinates))
    y_max = np.zeros(len(coordinates))
    # Coordonnées xy des maximums locaux

    plt.imshow(np.transpose(image2D))
    for i in range(len(coordinates)):
        x_max[i] = X_im[int(coordinates[i][1])][int(coordinates[i][0])]
        y_max[i] = Y_im[int(coordinates[i][1])][int(coordinates[i][0])]
        plt.scatter(x_max[i]/pixel_camera,y_max[i]/pixel_camera,color='red')
    
    plt.show()

    initial_guess = []
    # Estimation initiale des paramètres
    for i in range(len(coordinates)):
        initial_guess = np.append(initial_guess,[x_max[i], y_max[i],10, 10, 100., 20])

    popt, pcov = curve_fit(multi_gaussian2d, (X_im, Y_im), np.transpose(image2D).ravel(), p0=initial_guess)

    return popt

def compute_msd(positions):
    """
    Input : positions (une matrice N par 2 contenant les coordonnées de tous les fits gaussiens)
            max_lag (saut maximal entre des frames subséquents, en entiers)
    Output : results_msd = [taus, msd] --> deux vecteur colonnes.
             taus : vecteur contenant les entiers des lag times, allant de 1 à max_lag. Axe x du graphique de MSD.
             msd : vecteur contenant les valeurs du msd, correspond aux valeurs y du graphique de MSD
    """

    max_lag = 5
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

    return D_exp

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
    
    return coordinates[1,:]

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
    plt.plot(t, msd*1e12, 'o', label='MSD data')
    plt.plot(t, msd_fit*1e12, '-', label=f'Linear fit, pente = {slope*1E12}')
    plt.xlabel('Time (s)')
    plt.ylabel('MSD')
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
k_b = 1.380649E-23  # Constante de Boltzmann (J/K)
T = 293  # Température absolue du fluide (K)
eta = 1.01e-3  # Viscosité dynamique du fluide (Pa s)

# Paramètres de la caméra
delta_t = 0.5  # Délai entre chaque frame (s)
N_images = 116 # Nombre d'images 
pixel_camera = 0.377e-6 # Taille du pixel (m)
grossissement = 1.55e-6 / pixel_camera

# Paramètres de performance
N_pixel = 30 # Détermine la taille de la région d'intérêt dans les images. 

# Initialisation des array pour les positions estimées
x_guess = np.zeros(N_images)
y_guess = np.zeros(N_images)

# Position du maximum de la première image, pour zoom sur une région d'intérêt
index_max = index_max_image("C:\\Users\\gauth\\OneDrive\\Documents\\Polytechnique\\Polytechnique automne 2025\\Techniques expérimentales et instrumentation\\Mandat 2\\code_microscope\\timelapse_1080p_2fps_10enchantillon2\\img_0000.jpg")

for i in range(N_images):

    # Ouverture de l'image
    image2D = plt.imread("C:\\Users\\gauth\\OneDrive\\Documents\\Polytechnique\\Polytechnique automne 2025\\Techniques expérimentales et instrumentation\\Mandat 2\\code_microscope\\timelapse_1080p_2fps_10enchantillon2\\img_{:04}.jpg".format(i))

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
    param = fit2D_multiple_gaussians(X_im, Y_im, image2D)

    if i == 0:
        # On prend la particule avec la plus grande intensité
        x_guess[i] = param[0]
        y_guess[i] = param[1]
    else: 
        # On garde la particule qui est le plus près de la position précédende
        dist = np.zeros(len(param[::6]))
        for j in range(len(param[::6])):
            dist[j] = np.sqrt((param[0 + 6*j] - x_guess[i-1])**2 + (param[1 + 6*j] - x_guess[i-1])**2)
        j_closer = np.argmin(dist)
        x_guess[i] = param[0 + 6*j_closer] 
        y_guess[i] = param[1 + 6*j_closer]

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
D_exp = fit_msd_linear(res_msd[0], res_msd[1], delta_t) 

print(f"La valeur du coefficient de diffusion est {D_exp} m^2/s.")

r_exp = calcul_r_exp(D_exp, T, eta) 

print(f"Le rayon estimé de la particule est de {1e6*r_exp} um.")
