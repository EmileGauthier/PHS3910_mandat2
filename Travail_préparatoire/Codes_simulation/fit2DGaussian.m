function params = fit2DGaussian(X, Y, Z)
%FIT2DGAUSSIAN  Fit a 2D Gaussian by nonlinear least squares.
% Ce code a été généré par chatgpt.
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
    offset0  = min(Z(:)); % offset = bruit --> valeur min
    [~, idx] = max(Z(:)); % estime le centre comme la position au pixel max, avec [~, idx] une notation compacte pour avoir juste l'indice.
    x0_0     = X(idx);
    y0_0     = Y(idx);
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
  [~, iy] = min(abs(Y(:,1) - yc));
  xprof = X(iy,:);
  zxp = Z(iy,:);
  sx = fwhm_to_sigma(xprof, zxp, half);

  % y-profile (take nearest column to xc)
  [~, ix] = min(abs(X(1,:) - xc));
  yprof = Y(:,ix);
  zyp = Z(:,ix);
  sy = fwhm_to_sigma(yprof, zyp, half);
  s0 = [sx, sy]; % Jsp pourquoi ça output pas sans ça...
end

function sigma = fwhm_to_sigma(coord, profile, half)
  % interpolate to find half-maximum crossing points
  % ensure monotonic around peak by searching left/right from max
  [~, imax] = max(profile);
  % left crossing
  xl = interp1(profile(1:imax), coord(1:imax), half, 'linear');
  % right crossing
  xr = interp1(profile(imax:end), coord(imax:end), half, 'linear');
  fwhm = xr - xl;
  sigma = fwhm / (2*sqrt(2*log(2))); % = fwhm / 2.35482
end