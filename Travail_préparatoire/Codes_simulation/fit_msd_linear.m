function D_exp = fit_msd_linear(taus, msd, dt)
% Fonction qui Fit (linéairement) MSD(t) = 2*dim*D*t + C
dim = 2; % Évalue en xy
t = taus * dt; 
X = [t ones(size(t))]; % Initialise la matrice pour les moindres carrés. Chaque ligne est [t_i, 1]
beta = X \ msd; % beta est l'estimé des moindres carrés, avec beta(1) la pente et beta(2) l'ordonnée à l'origine.
slope = beta(1); %intercept = beta(2);
D_exp = slope / (2*dim);
% Si on a besoin de plus d'informations sur le fit linéaire : 
%n = length(msd);
%if n > 2
    %residuals = msd - X*beta;
    %sigma2 = sum(residuals.^2) / (n - 2);
    %cov_beta = sigma2 * inv(X'*X);
%else
    %cov_beta = NaN(2,2);
%end
%if ~any(isnan(cov_beta(:)))
%    se_slope = sqrt(cov_beta(1,1));
%    se_D = se_slope / (2*dim);
%else
%    se_D = NaN;
%end
end
