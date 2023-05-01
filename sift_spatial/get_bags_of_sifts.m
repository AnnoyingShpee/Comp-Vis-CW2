% Implementated according to the starter code prepared by James Hays, Brown
% University 
% Michal Mackiewicz, UEA

function image_feats = get_bags_of_sifts(image_paths, vocab_size, step, bin_size, colour_type)
% image_paths is an N x 1 cell array of strings where each string is an
% image path on the file system.

% This function assumes that 'vocab.mat' exists and contains an N x 128
% matrix 'vocab' where each row is a kmeans centroid or a visual word. This
% matrix is saved to disk rather than passed in a parameter to avoid
% recomputing the vocabulary every time at significant expense.

% image_feats is an N x d matrix, where d is the dimensionality of the
% feature representation. In this case, d will equal the number of clusters
% or equivalently the number of entries in each image's histogram.

% You will want to construct SIFT features here in the same way you
% did in build_vocabulary.m (except for possibly changing the sampling
% rate) and then assign each local feature to its nearest cluster center
% and build a histogram indicating how many times each cluster was used.
% Don't forget to normalize the histogram, or else a larger image with more
% SIFT features will look very different from a smaller version of the same
% image.

%{
Useful functions:
[locations, SIFT_features] = vl_dsift(img) 
 http://www.vlfeat.org/matlab/vl_dsift.html
 locations is a 2 x n list list of locations, which can be used for extra
  credit if you are constructing a "spatial pyramid".
 SIFT_features is a 128 x N matrix of SIFT features
  note: there are step, bin size, and smoothing parameters you can
  manipulate for vl_dsift(). We recommend debugging with the 'fast'
  parameter. This approximate version of SIFT is about 20 times faster to
  compute. Also, be sure not to use the default value of step size. It will
  be very slow and you'll see relatively little performance gain from
  extremely dense sampling. You are welcome to use your own SIFT feature
  code! It will probably be slower, though.

D = vl_alldist2(X,Y) 
   http://www.vlfeat.org/matlab/vl_alldist2.html
    returns the pairwise distance matrix D of the columns of X and Y. 
    D(i,j) = sum (X(:,i) - Y(:,j)).^2
    Note that vl_feat represents points as columns vs this code (and Matlab
    in general) represents points as rows. So you probably want to use the
    transpose operator '  You can use this to figure out the closest
    cluster center for every SIFT feature. You could easily code this
    yourself, but vl_alldist2 tends to be much faster.
%}
load('vocab.mat');
vocab = vocab';
image_feats = zeros(length(image_paths), vocab_size);
switch colour_type
    case "grayscale"
        parfor i = 1:length(image_paths)
            img = imread(image_paths{i});
            % vl_dsift only accepts images that are of class SINGLE and grayscale.
            % Hence, image must be converted using single() and checked if it is
            % colour to become grayscale.
            if size(img, 3) > 1
                img = rgb2gray(img);     
            end
            [~, SIFT_features] = vl_dsift(single(img), 'fast', 'step', step, 'size', bin_size);
            SIFT_features = single(SIFT_features);
            % Get the distances between each SIFT feature and vocab. 
            % Returns a S x K matrix of distances where 
            % S = Number of columns in SIFT_features
            % K = Number of columns in vocab
            distance = vl_alldist2(SIFT_features, vocab);
            histogram = create_sift_histogram(distance, vocab_size);
            % Normalise the histogram and store into matrix
            image_feats(i, :) = histogram / sum(histogram);
        end
    case "rgb"
        parfor i = 1:length(image_paths)
            img = imread(image_paths{i});
            if size(img, 3) == 1
                [rgb, ~] = gray2ind(img,256);
                img = cat(3,rgb, rgb, rgb);
            end
            [~, R_SIFT_features] = vl_dsift(single(img(:,:,1)), 'fast', 'step', step, 'size', bin_size);
            [~, G_SIFT_features] = vl_dsift(single(img(:,:,2)), 'fast', 'step', step, 'size', bin_size);
            [~, B_SIFT_features] = vl_dsift(single(img(:,:,3)), 'fast', 'step', step, 'size', bin_size);
            SIFT_features = [R_SIFT_features; G_SIFT_features; B_SIFT_features];
            % Get the distances between each SIFT feature and vocab. 
            % Returns a S x K matrix of distances where 
            % S = Number of columns in SIFT_features
            % K = Number of columns in vocab
            distance = vl_alldist2(single(SIFT_features), vocab);
            histogram = create_sift_histogram(distance, vocab_size);
            % Normalise the histogram and store into matrix
            image_feats(i, :) = histogram / sum(histogram); 
        end
    case "rgb_phow"
        parfor i = 1:length(image_paths)
            img = imread(image_paths{i});
            if size(img, 3) == 1
                [rgb, ~] = gray2ind(img,256);
                img = cat(3,rgb, rgb, rgb);
            end
            [~, SIFT_features] = vl_phow(single(img), 'step', step, 'sizes', bin_size, 'color', 'rgb');
            distance = vl_alldist2(single(SIFT_features), vocab);
            histogram = create_sift_histogram(distance, vocab_size);
            image_feats(i, :) = histogram / sum(histogram);
        end
end

end