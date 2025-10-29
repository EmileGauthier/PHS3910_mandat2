function r_exp = calcul_r_exp(D_exp, T, eta)
% Calculer r avec Stokes-Einstein
    k_b = 1.380649E-23 ; % J/K
    r_exp = (k_b*T)/(6*pi*eta*D_exp);
end
