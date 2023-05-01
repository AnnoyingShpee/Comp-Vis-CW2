function [features] = get_colour_histograms(img_paths, quantisation, colour_space)
%GET_COLOUR_HISTOGRAM Summary of this function goes here
%   img_paths = Cell array of test image paths
%   quantisation = Quantisation level of image colour values
%   colour_space = Type of colour space of image colour values
features = zeros(size(img_paths,1), quantisation^3);
% USE THREADS FOR PARALLEL RUNNING
parfor i = 1:length(img_paths)
%     colour_hist = zeros(quantisation, quantisation, quantisation);
    img = imread(img_paths{i});
    % Convert image RGB values to a different different colour space if
    % colour_space is not "rgb"
    switch lower(colour_space)
        case "hsv"
            img = rgb2hsv(img);
        case "lab"
            img = rgb2xyz(img);
        case "ycbcr"
            img = rgb2ycbcr(img);
        case "yiq"
            img = rgb2ntsc(img);
    end
    % Quantise the image colour values
    quantised_img = image_quantisation(img, quantisation);
    % Get colour histogram
    colour_hist = create_colour_histogram(quantised_img, quantisation);
    % Reshape colour histogram to be (1 x M)
    tmp = colour_hist(:);
    % Store feature vector
    features(i,:) = tmp;
end
end

