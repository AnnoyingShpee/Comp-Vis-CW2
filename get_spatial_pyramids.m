function image_feats = get_spatial_pyramids(image_paths, vocab_size, num_layers, step, bin_size, colour_type)
%GET_SPATIAL_PYRAMIDS Summary of this function goes here
%   image_paths = N x 1 size cell of image path strings
%   num_layers = Number of spatial pyramid layers. 0 <= num_layers <= n
%   step = For 'step' in vl_dsift().
%   bin_size = For 'size' in vl_dsift().
%   colour_type = SIFT extraction with or without colour.

% load('vocab.mat');
% vocab = vocab';
total_cells = sum((2.^(0:num_layers-1)).^2);
image_feats = zeros(length(image_paths), vocab_size*total_cells);
switch lower(colour_type)
    case "grayscale"
        parfor i = 1:length(image_paths)
            img = imread(image_paths{i});
            if size(img, 3) > 1
                img = rgb2gray(img);
            end
            img = single(img);
            layers = create_pyramid_layer(img, vocab_size, num_layers, step, bin_size, colour_type)
            image_feats(i,:) = layers;
        end
    case "rgb"
        parfor i = 1:length(image_paths)
            img = imread(image_paths{i});
            if size(img, 3) == 1
                [rgb, ~] = gray2ind(img,256);
                img = cat(3,rgb, rgb, rgb);
            end
            img = single(img);
            layers = create_pyramid_layer(img, vocab_size, num_layers, step, bin_size, colour_type);
            image_feats(i,:) = layers;
        end
    case "rgb_phow"
        parfor i = 1:length(image_paths)
            img = imread(image_paths{i});
            if size(img, 3) == 1
                [rgb, ~] = gray2ind(img,256);
                img = cat(3,rgb, rgb, rgb);
            end
            img = single(img);
            layers = create_pyramid_layer(img, vocab_size, num_layers, step, bin_size, colour_type);
            image_feats(i,:) = layers;
        end            
end
end


