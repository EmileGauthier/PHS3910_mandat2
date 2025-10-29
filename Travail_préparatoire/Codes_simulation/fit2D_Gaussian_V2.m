function params = fit2D_Gaussian_V2(X, Y, Z)
%FIT2DGAUSSIAN  Fit a 2D Gaussian by nonlinear least squares.
% Ce code a été initialement généré par chatgpt, mais fortement modifié par
% l'équipe.
%
%   params = fit2DGaussian(X, Y, Z)
%
% Inputs:
%   X, Y : coordinate matrices (as from meshgrid)
%   Z    : data matrix (same size as X,Y)
%
% Output:
%   params = [A, x0, y0, sx, sy, offset]
%     where the fitted model is:
%       Zfit = A * exp( -((X - x0).^2/(2*sx^2) + (Y - y0).^2/(2*sy^2)) ) + offset
%
% Requires Optimization Toolbox (for lsqcurvefit).

    % Initial guesses (reasonable defaults)
    A0       = max(Z(:)) - min(Z(:)); % Guess l'amplitude comme l'écart max d'intensité
    disp('A0');
    A0
    offset0  = min(Z(:)); % offset = bruit --> valeur min
    [max_val, linear_idx] = max(Z(:));     % Find maximum value and its linear index
    [row_max, col_max] = ind2sub(size(Z), linear_idx);
    x0_0     = X(col_max);
    disp('x0_0')
    x0_0
    y0_0     = Y(row_max);
    disp('y0_0')
    y0_0
    % Estimations initiales des std : 

    s0 = sigma_from_fwhm(X, Y, Z, x0_0, y0_0);
    sx0 = s0(1);
    sy0 = s0(2);

    % Mais on sait pour une gaussienne que fwhm = 2sqrt(2ln2) std_1D !!!
    % De ce que j'en comprends, tant qu'on a un bon guess initial, on a des
    % bonnes chances de converger vers la vraie solution (plutôt qu'un
    % minimum local), donc même s'il propose une autre méthode avec les
    % deuxième moments, ça devrait être chill. 

    p0 = [A0, x0_0, y0_0, sx0, sy0, offset0];

    % Mettre les données dans le format voulu pour lsqcurvefit
    xdata = [X(:), Y(:)];
    zdata = Z(:);

    % Define 2D Gaussian model
    gauss2d = @(p, xy) p(1) * exp( ...
        -((xy(:,1) - p(2)).^2 / (2*p(4)^2) + (xy(:,2) - p(3)).^2 / (2*p(5)^2)) ) + p(6);

    % Déclare ses variables p (paramètres) et xy (coordonnées où on évalue
    % la gaussienne 2D). 
    
    % Perform least-squares fit
    options = optimoptions('lsqcurvefit','Display','off'); % Turn off verbose output. You can change 'Display' for diagnostics.
    lb = [0, -inf, -inf, 0, 0, -inf]; % Lower bounds sur les paramètres : amplitude ≥ 0, widths ≥ 0. Centers unconstrained, offset unconstrained (can be negative).
    ub = [inf, inf, inf, inf, inf, inf]; % Bornes supérieures : on s'en fout!

    params = lsqcurvefit(gauss2d, p0, xdata, zdata, lb, ub, options);
end

function s0 = sigma_from_fwhm(X, Y, Z, xc, yc) % xc et yc les coordonnées du centre --> x0_0 et y0_0.
  % Z should have background subtracted if needed
  half = max(Z(:))/2;
  % x-profile (take nearest row to yc)
  %[~, iy] = min(abs(Y(:,1) - yc));
  [~, iy] = min(abs(Y(:) - yc));
  %disp('iy');
  %iy
  %xprof = X(iy,:);
  xprof = X(:)'; % vecteur ligne
  %disp('xprof') 
  %xprof
  zxp = Z(iy,:); % vecteur ligne
  %disp('zxp')
  %zxp
  sx = fwhm_to_sigma(xprof, zxp, half);
  disp('sx : ')
  sx

  % y-profile (take nearest column to xc)
  disp('about to start y profile')
  [~, ix] = min(abs(X(:) - xc));
  yprof = Y(:)'; % vecteur ligne
  %disp('yprof')
  %yprof
  zyp = Z(:,ix)'; %vecteur ligne
  %disp('zyp')
  %zyp
  sy = fwhm_to_sigma(yprof, zyp, half);
  disp('sy : ')
  sy
  s0 = [sx, sy]; % Jsp pourquoi ça output pas sans ça...
end

function sigma = fwhm_to_sigma(coord, profile, half)
  % interpolate to find half-maximum crossing points
  % ensure monotonic around peak by searching left/right from max
  [~, imax] = max(profile);
  disp('imax')
  %imax
  % left crossing
  %disp('profile(1:imax)')
  %profile(1:imax)
  %disp('coord')
  %coord
  %disp('coord(1:imax)')
  %coord(1:imax)
  %xl = interp1(profile(1:imax), coord(1:imax), half, 'linear'); % ERREUR ICI, il a besoin d'au moins 2 sample points...
  xl = interp1(coord(1:imax), profile(1:imax), half, 'linear');
  disp('xl : ')
  xl
  % right crossing
  %xr = interp1(profile(imax:end), coord(imax:end), half, 'linear');
  xr = interp1(coord(imax:end), profile(imax:end), half, 'linear');
  disp('xr : ')
  xr
  fwhm = xr - xl;
  sigma = fwhm / (2*sqrt(2*log(2))); % = fwhm / 2.35482
end
