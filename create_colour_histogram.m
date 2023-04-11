function [colour_hist] = create_colour_histogram(quantised_img, quantise_level)
%CREATE_COLOUR_HISTOGRAM Summary of this function goes here
%   quantised_img = image matrix with quantised colour values 
%   quantise_level = quantisation level for initialisation
%%
img_size = size(quantised_img);
% Initialise (q x q x q) matrix for colour histogram of image
colour_hist = zeros(quantise_level, quantise_level, quantise_level);
% Reshape image from (N x M x 3) to (N*M x 3) to reduce number of for loops
tmp = reshape(quantised_img, [img_size(1) * img_size(2), 3]);
% Get colour histogram of image
for x = 1:size(tmp, 1)
    r = tmp(x, 1);
    g = tmp(x, 2);
    b = tmp(x, 3);
    colour_hist(r, g, b) = colour_hist(r, g, b) + 1;
end
% A less efficient method would look like this
%     for x = 1:size(quantised_img, 1)
%         for y = 1:size(quantised_img, 2)
%             r = quantised_img(x, y, 1);
%             g = quantised_img(x, y, 2);
%             b = quantised_img(x, y, 3);
%             colour_hist(r, g, b) = colour_hist(r, g, b) + 1;
%         end
%     end
end

