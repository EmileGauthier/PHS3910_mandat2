function image2D = real_image(x, y, X_im, Y_im, NA, lmda, N_photons, pixel_camera,n_pixel_camera)
 
%   Paramètres sortie : psf - Maillage 2D du nombre de photons par pixel
%   Paramètres entrée : x et y - Coordonnées réelles de la particule.
%                       X_im et Y_im donne les positions en x et en y sur
%                       le détecteur sous forme de maillage 2D.
%                       NA est l'ouverture numérique du système optique;
%                       lmda est la longueur d'onde du rayon incident;
%                       N_photons - Nombre de photons par image
%                       

    [X_rand,Y_rand] = generate_random_number(x, y, N_photons, NA, lmda, pixel_camera);
    
    image2D = zeros(n_pixel_camera(2),n_pixel_camera(1));

    for i = 1:N_photons

        x_i = X_rand(i);
        y_i = Y_rand(i);

        % Pixel associé à la position
        index_x = floor(x_i/pixel_camera + (1/2));
        index_y = floor(y_i/pixel_camera + (1/2));
        
        image2D(index_y,index_x) = image2D(index_y,index_x) + 1;
    end
    
    % Normalisation et ajout d'un bruit de poisson
    image_max = max(image2D,[],"all");
    image2D = image2D / (image_max);
    %image2D = imnoise(image2D,"poisson");
end