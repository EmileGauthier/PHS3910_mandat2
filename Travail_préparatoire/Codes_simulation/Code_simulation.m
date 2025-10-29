% Paramètres de la particule
N_step = 50; % Nombre de pas de temps par marche aléatoire
N_marches = 1; % Nombre de marches aléatoires par valeur de paramètre.
N_param = 1; % Nombre de valeurs du paramètre testées. i.e. pour r = 0.1, 1, 10 : N_param = 3.
N_photons = 1000; % Nombre de photons reçu par la caméra par position de particules
r_real = 1E-6 ; % Taille réelle de la particule

% Constantes physiques
k_b = 1.380649E-23 ; % Constante de Boltzmann (J/K)
T = 293 ; % Température absolue du fluide (K)
eta = 1E-3 ; % Viscosité dynamique du fluide
% La viscosité dynamique de l'eau est environ 0,001 Pa * s à 20°C.
D_real = (k_b * T) / (6 * pi * eta * r_real) ; % Coefficient de diffusion (m^2/s)

% Paramètres de la caméra
delta_t = 0.1e-3 ; % Délai entre chaque frame (1 ms entre chaque frame?)
grossissement = 20; % Magnification du système optique
NA = 1.33; % Ouverture numérique
lmda = 500e-9; % Longueur d'onde captée (m)
pixel_camera = 1.55e-6; % Taille du pixel (m)
n_pixels_camera = [4056;3040]; % Dimensions du détecteur (pixel)
pixel_objet = pixel_camera / grossissement; % Taille du pixel dans l'espace objet

% Maillage dans l'espace image
x_im = 0:pixel_camera:(n_pixels_camera(1)-1)*pixel_camera; % Yo watch out pcq matlab commence à 1...
y_im = 0:pixel_camera:(n_pixels_camera(2)-1)*pixel_camera;
[X_im,Y_im] = meshgrid(x_im,y_im);

x_positions = zeros(1, N_step); % Initialiser le vecteur des positions x
y_positions = zeros(1, N_step); % Initialiser le vecteur des positions y
x_positions(1) = n_pixels_camera(1)/2 * pixel_camera; % Position initiale de la particule en x.
y_positions(1) = n_pixels_camera(2)/2 * pixel_camera; % Position initiale de la particule en y.

x_guess = zeros(1, N_step); % Initialiser le vecteur des positions x
y_guess = zeros(1, N_step); % Initialiser le vecteur des positions y


r_val_j = zeros(1, N_marches); % Initialiser le vecteur des résultats r pour chaque j.
valeurs_k = zeros(1, N_param); % Initialiser vecteur des valeurs du paramètres (axe x des résultats)
vec_r_tot = zeros(1, N_param); % Initialiser vecteur des valeurs de r pour chaque point (x (paramètres), y (r)) des résultats. 
vec_std_r = zeros(1, N_param); % Initialiser vecteur des incertitudes sur chaque valeur de r (incertitudes en y).


for k = 1:N_param % Pour chaque valeur du paramètre testé. 
    % Opération sur le paramètre. Par exemple, pour la taille : r_true = r_true * k

    for j = 1:N_marches % Pour toutes les marches, pour une valeur du paramètre testé.
    
        for i = 2:N_step % Pour une marche aléatoire complète
            % Calculer une position
            [x_positions(i), y_positions(i)] = brownien(x_positions(i-1), y_positions(i-1), D_real, delta_t);
        
            % Création d'une image réelle
            image2D = real_image(x_positions(i), y_positions(i),X_im, Y_im, NA, lmda, N_photons,pixel_camera,n_pixels_camera);
            
            bar3(image2D(1510:1530,2020:2040))
        
            % Localisation de la particule
            param = fit2DGaussian(X_im, Y_im, image2D);
            x_guess(i) = param(2);
            y_guess(i) = param(3);
        end
        
        pos_guess = [x_guess', y_guess'];
        res_msd = compute_msd(pos_guess); % Ajouter le paramètres max_lag si besoin, mais par défaut : N/4. 
        % res_msd = [taus, msd] des vecteurs colonne.
        
        D_exp = fit_msd_linear(res_msd(1), res_msd(2), delta_t); % Pour une marche aléatoire complète.
        r_exp = calcul_r_exp(D_exp, T, eta); % Encore pour une seule marche aléatoire complète.
        r_val_k(j) = r_exp;
    end
    vec_r_tot(k) = mean(r_val_k);
    vec_std_r(k) = std(r_val_k); 
    % Ajouter la valeur du paramètre par ex ici :
    % valeurs_k(k) = r_true; % Pour la taille
end

% Reste plus qu'à plot les résultats : vec_r_tot vs valeurs_k , avec incertitudes en y vec_std_r

