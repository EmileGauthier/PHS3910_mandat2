function [x2, y2] = brownien(x1, y1, D_real, delta_t)

% Définition de la fonction "brownien" : 
%   Paramètres sortie : x2 et y2 les coordonnées réelles de la particule.
%   Paramètres entrée : x1 et y1 les coordonnées initiales de la particule;
%                       D le coefficient de diffusion théorique;
%                       delta_t le temps entre chaque frame;
%                       

% X = randn retourne un scalaire à partir de la distribution normale


% Calculer les déplacements aléatoires en fonction du coefficient de diffusion
sigma = sqrt(2 * D_real * delta_t); % Écart type du mouvement brownien
x2 = x1 + sigma * randn; % Nouvelle position x
y2 = y1 + sigma * randn; % Nouvelle position y

end 