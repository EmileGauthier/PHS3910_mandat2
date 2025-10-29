function [X_rand,Y_rand] = generate_random_number(x, y, N_photons, NA, lmda, pixel_camera)
    
    n_pixel = 100; % Définie la région où on génère des pixels (efficacité numérique)
    x_min = x - n_pixel*pixel_camera; x_max = x + n_pixel*pixel_camera;
    y_min = y - n_pixel*pixel_camera; y_max = y + n_pixel*pixel_camera;

    f_max = f(x,y,x+eps,y+eps,NA,lmda);

    X_rand = zeros(N_photons,1);
    Y_rand = zeros(N_photons,1);
    n = 0;
    count = 0;
    while n < N_photons
        % Tirages uniformes
        x_try = x_min + (x_max - x_min) * rand();
        y_try = y_min + (y_max - y_min) * rand();
    
        % Tirage vertical
        u = f_max * rand();
    
        % Acceptation
        if u < f(x_try, y_try,x,y,NA,lmda)
            n = n + 1;
            X_rand(n) = x_try;
            Y_rand(n) = y_try;
        end
        count = count + 1;
        if count == 1000 && n <= 50 % La région explorée est trop grande
            n_pixel = ceil(0.5*n_pixel); % On réduit cette région
            x_min = x - n_pixel*pixel_camera; x_max = x + n_pixel*pixel_camera;
            y_min = y - n_pixel*pixel_camera; y_max = y + n_pixel*pixel_camera;
            count = 0;
        end
    end
end

function psf = f(x, y, x_try, y_try, NA, lmda)
%   Paramètres sorties : psf - Maillage 2D correspondant à la convolution
%   du delta Dirac à la position x,y avec la fonction psf.
    r = sqrt((x_try - x).^2 + (y_try - y).^2);
    temp = (2 .* pi .* NA .* r)./lmda;
    psf = (2*besselj(1,temp) ./ temp).^2;

end