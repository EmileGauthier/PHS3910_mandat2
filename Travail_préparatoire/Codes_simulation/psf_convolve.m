function psf = psf_convolve(x, y, X_im, Y_im, NA, lmda)
%   Paramètres sorties : psf - Maillage 2D correspondant à la convolution
%   du delta Dirac à la position x,y avec la fonction psf.
    r = sqrt((X_im - x).^2 + (Y_im - y).^2);
    temp = (2 .* pi .* NA .* r)./lmda;
    psf = (2*besselj(1,temp) ./ temp).^2;

end