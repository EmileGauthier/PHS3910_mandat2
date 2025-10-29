% Étape 4 - Avec toutes les positions sur toutes les photos, calculer le
% MSD

function results_msd = compute_msd(positions, max_lag)
% fonction créée par chatgpt puis modifiée par l'équipe.
% Output : results_msd = [taus, msd], deux vecteur colonnes.

% taus : vecteur contenant les entiers des lag times, allant de 1 à max_lag
% (correspond à l'axe x du graphique de l'énoncé)
% msd : vecteur contenant les valeurs du msd, correspond aux valeurs y du
% graphique de l'énoncé.
% positions : matrice N par 2 (dans l'unité donnée, ex m ou um)
N = size(positions,1); % On a N points dans la trajectoire brownienne
if nargin < 2 || isempty(max_lag) % Si on donne pas de valeur de max_lag, on le fixe à N/4, pourrait etre jusqu'à floor(N/2). 
    max_lag = floor(N/4);
end
msd = zeros(max_lag,1); % Vecteur colonne

for lag = 1:max_lag % Sur sur les times lags (delta_t dans l'énoncé)
    diffs = positions(lag+1:end,:) - positions(1:end-lag,:); 
    % 1er terme : positions au temps futur, 2e terme : positions initiales associées. 
    % Diffs est une matrice (N - lag) x 2
    sq = sum(diffs.^2,2); % Déplacements quadratiques sommés en x et y. 
    msd(lag) = mean(sq);
end
taus = (1:max_lag)'; % Vecteur colonne

results_msd = [taus, msd];
end
