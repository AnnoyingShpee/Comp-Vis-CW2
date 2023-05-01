function layers = create_pyramid_layer(img, vocab_size, num_layers, step, bin_size, colour_type, vocab)
%CREATE_PYRAMID_LAYERS Summary of this function goes here
% Creates the spatial pyramid kernels by sectioning the image into quarters
% and calculating the histogram from each kernel.
load('vocab.mat');
vocab = vocab';
img_size = size(img);
layers = [];
% +1 to include the final layer as the original image layer is put into
% the first cell already.
for L = 0:num_layers-1
    % Weight of each layer in the spatial pyramid
    if L == 0
        kernel_weight = 1/(2^num_layers);
    else
        kernel_weight = 1/(2^(num_layers - L + 1));
    end
    % Dimension size of each layers
    cells_per_dimension = 2^L;
    rows_per_cell = floor(img_size(1) / cells_per_dimension);
    cols_per_cell = floor(img_size(2) / cells_per_dimension);
    row_indices = floor(1:img_size(1)/cells_per_dimension:img_size(1));
    col_indices = floor(1:img_size(2)/cells_per_dimension:img_size(2));
    % Get histogram of visual words for each section
    hists = [];
    for r = 1:length(row_indices)
        for c = 1:length(col_indices)
            if r == length(row_indices) 
                row_end = img_size(1);
            else
                row_end = row_indices(r)+rows_per_cell-1;
            end
            if c == length(col_indices)
                col_end = img_size(2);
            else
                col_end = col_indices(c)+cols_per_cell-1;
            end
            switch lower(colour_type)
                case "grayscale"
                    cell = img(row_indices(r):row_end, col_indices(c):col_end);
                    [~, SIFT_features] = vl_dsift(cell, 'fast', 'step', step, 'size', bin_size);
                    distance = vl_alldist2(single(SIFT_features), vocab);
                    SIFT_hist = create_sift_histogram(distance, vocab_size);
                    hists = [hists, SIFT_hist];
                case "rgb"
                    cell = img(row_indices(r):row_end, col_indices(c):col_end, :);
                    [~, R_SIFT_features] = vl_dsift(cell(:,:,1), 'fast', 'step', step, 'size', bin_size);
                    [~, G_SIFT_features] = vl_dsift(cell(:,:,2), 'fast', 'step', step, 'size', bin_size);
                    [~, B_SIFT_features] = vl_dsift(cell(:,:,3), 'fast', 'step', step, 'size', bin_size);
                    SIFT_features = [R_SIFT_features; G_SIFT_features; B_SIFT_features];
                    distance = vl_alldist2(single(SIFT_features), vocab);
                    SIFT_hist = create_sift_histogram(distance, vocab_size);
                    hists = [hists, SIFT_hist];
                case "rgb_phow"
                    cell = img(row_indices(r):row_end, col_indices(c):col_end, :);
                    [~, SIFT_features] = vl_phow(cell, 'step', step, 'sizes', bin_size, 'color', 'rgb');
                    distance = vl_alldist2(single(SIFT_features), vocab);
                    SIFT_hist = create_sift_histogram(distance, vocab_size);
                    hists = [hists, SIFT_hist];
            end
        end
    end
    % Weight the pyramid layer and concatenate to feature
    % vector
    layers = [layers, hists * kernel_weight];
%     layers = layers / sum(layers);
end

end

