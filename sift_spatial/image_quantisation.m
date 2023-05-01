function [quantised_image] = image_quantisation(image, quantisation)
%IMAGE_QUANTISATION Summary of this function goes here
%   image = image matrix
%   quantisation = quantisation level of colour values
double_image = double(image);
quantised_image = double_image/255;
quantised_image = round(quantised_image * (quantisation-1)) + 1;
end

