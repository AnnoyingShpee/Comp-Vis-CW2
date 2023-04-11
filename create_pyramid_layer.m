function layers = create_pyramid_layer(img, num_layers, step, bin_size)
%CREATE_PYRAMID_LAYERS Summary of this function goes here
% Creates the spatial pyramid kernels by sectioning the image into quarters
% and calculating the histogram from each kernel.

% +1 to include the final layer as the original image layer is put into
% the first cell already.
layers = cell(num_layers+1, 1);
img_size = size(img);
layers{1} = img;

for L = 1:num_layers+1
    layer = create_pyramid_layer(prev_layer, step, bin_size);
    layers{L+1} = layer; 
    if  numel(layers{L}) == 2^(num_layers + num_layers)
        
    end
end
% Takes into account of odd-number dimension sizes
sub_row = floor(img_size(1) / 2);
sub_col = floor(img_size(2) / 2);
% Split image into sections. 
top_left = img(1:sub_row, 1:sub_col);
top_right = img(1:sub_row, sub_col+1:end);
bottom_left = img(sub_row+1:end, 1:sub_col);
bottom_right = img(sub_row+1:end, sub_col+1:end);

top_left_SIFT = create_sift_features(top_left, step, bin_size);
top_right_SIFT = create_sift_features(top_right, step, bin_size);
bottom_left_SIFT = create_sift_features(bottom_left, step, bin_size);
bottom_right_SIFT = create_sift_features(bottom_right, step, bin_size);

img_sections = [
    top_left_SIFT    ,    top_right_SIFT   ;
    bottom_left_SIFT ,    bottom_right_SIFT
];

end

