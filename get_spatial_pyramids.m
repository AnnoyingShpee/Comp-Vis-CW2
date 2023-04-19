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
%             layer = [];
%             for L = 0:num_layers-1
%                 % Weight of each layer in the spatial pyramid
%                 if L == 0
%                     kernel_weight = 1/(2^num_layers);
%                 else
%                     kernel_weight = 1/(2^(num_layers - L + 1));
%                 end
%                 % Dimension size of each layers
%                 cells_per_dimension = 2^L;
%                 rows_per_cell = floor(size(img, 1) / cells_per_dimension);
%                 cols_per_cell = floor(size(img, 2) / cells_per_dimension);
%                 row_indices = floor(1:img_size(1)/cells_per_dimension:img_size(1));
%                 col_indices = floor(1:img_size(2)/cells_per_dimension:img_size(2));
%                 % Get histogram of visual words for each section
%                 hists = [];
%                 for r = 1:length(row_indices)
%                     for c = 1:length(col_indices)
%                         if r == length(row_indices) 
%                             row_end = img_size(1);
%                         else
%                             row_end = row_indices(r)+rows_per_cell-1;
%                         end
%                         if c == length(col_indices)
%                             col_end = img_size(2);
%                         else
%                             col_end = col_indices(c)+cols_per_cell-1;
%                         end
%                         cell = img(row_indices(r):row_end, col_indices(c):col_end);
%                         [~, SIFT_features] = vl_dsift(cell, 'fast', 'step', step, 'size', bin_size);
%                         distance = vl_alldist2(single(SIFT_features), vocab);
%                         SIFT_hist = create_sift_histogram(distance, vocab_size);
%                         hists = [hists, SIFT_hist];
%                     end
%                 end
%                 % Weight the pyramid layer and concatenate to feature
%                 % vector
%                 layer = [layer, hists * kernel_weight];
% %                 layer = [layer, hists];
%             end
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
%             layer = [];
%             for L = 0:num_layers-1
%                 % Weight of each layer in the spatial pyramid
%                 if L == 0
%                     kernel_weight = 1/(2^num_layers);
%                 else
%                     kernel_weight = 1/(2^(num_layers - L + 1));
%                 end
%                 % Dimension size of each layers
%                 cells_per_dimension = 2^L;
%                 rows_per_cell = floor(size(img, 1) / cells_per_dimension);
%                 cols_per_cell = floor(size(img, 2) / cells_per_dimension);
%                 row_indices = floor(1:img_size(1)/cells_per_dimension:img_size(1));
%                 col_indices = floor(1:img_size(2)/cells_per_dimension:img_size(2));
%                 % Get histogram of visual words for each section
%                 hists = [];
%                 for r = 1:length(row_indices)
%                     for c = 1:length(col_indices)
%                         if r == length(row_indices) 
%                             row_end = img_size(1);
%                         else
%                             row_end = row_indices(r)+rows_per_cell-1;
%                         end
%                         if c == length(col_indices)
%                             col_end = img_size(2);
%                         else
%                             col_end = col_indices(c)+cols_per_cell-1;
%                         end
%                         cell = img(row_indices(r):row_end, col_indices(c):col_end, :);
%                         [~, R_SIFT_features] = vl_dsift(cell(:,:,1), 'fast', 'step', step, 'size', bin_size);
%                         [~, G_SIFT_features] = vl_dsift(cell(:,:,2), 'fast', 'step', step, 'size', bin_size);
%                         [~, B_SIFT_features] = vl_dsift(cell(:,:,3), 'fast', 'step', step, 'size', bin_size);
%                         SIFT_features = [R_SIFT_features; G_SIFT_features; B_SIFT_features];
%                         distance = vl_alldist2(single(SIFT_features), vocab);
%                         hist = create_sift_histogram(distance, vocab_size);
%                         hists = [hists, hist];
%                     end
%                 end
%                 % Weight the pyramid layer and concatenate to feature
%                 % vector
%                 layer = [layer, hists * kernel_weight];
% %                 layer = [layer, hists];
%             end
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
%             layer = [];
%             for L = 0:num_layers-1
%                 % Weight of each layer in the spatial pyramid
%                 if L == 0
%                     kernel_weight = 1/(2^num_layers);
%                 else
%                     kernel_weight = 1/(2^(num_layers - L + 1));
%                 end
%                 % Dimension size of each layers
%                 cells_per_dimension = 2^L;
%                 rows_per_cell = floor(size(img, 1) / cells_per_dimension);
%                 cols_per_cell = floor(size(img, 2) / cells_per_dimension);
%                 row_indices = floor(1:img_size(1)/cells_per_dimension:img_size(1));
%                 col_indices = floor(1:img_size(2)/cells_per_dimension:img_size(2));
%                 % Get histogram of visual words for each section
%                 hists = [];
%                 for r = 1:length(row_indices)
%                     for c = 1:length(col_indices)
%                         if r == length(row_indices) 
%                             row_end = img_size(1);
%                         else
%                             row_end = row_indices(r)+rows_per_cell-1;
%                         end
%                         if c == length(col_indices)
%                             col_end = img_size(2);
%                         else
%                             col_end = col_indices(c)+cols_per_cell-1;
%                         end
%                         cell = img(row_indices(r):row_end, col_indices(c):col_end, :);
%                         [~, SIFT_features] = vl_phow(single(cell), 'step', step, 'sizes', bin_size);
%                         distance = vl_alldist2(single(SIFT_features), vocab);
%                         histogram = create_sift_histogram(distance, vocab_size);
%                         histogram = histogram / sum(histogram);
%                         hists = [hists, histogram];
%                     end
%                 end
%                 % Weight the pyramid layer and concatenate to feature
%                 % vector
% %                 hists = hists / sum(hists);
%                 layer = [layer, hists * kernel_weight];
%                 layer = [layer, hists];
            image_feats(i,:) = layers;
        end
%             image_feats(i,:) = layer / sum(layer);
            
end
end


