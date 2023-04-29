% Michal Mackiewicz, UEA 
% This code has been adapted from the code 
% prepared by James Hays, Brown University
tic
%% Step 0: Set up parameters, vlfeat, category list, and image paths.
% FEATURE = 'tiny image';
% FEATURE = 'colour histogram';
% FEATURE = 'bag of sift';
FEATURE = 'spatial pyramids';

CLASSIFIER = 'nearest neighbor';
% CLASSIFIER = 'support vector machine';

% To test parameters
VOCAB_SIZES = 50:50:300;
Ks = 1:2:15;
LAMBDAS = 10.^(-5:1);
METRIC = "euclidean";
FEATURE_COLOUR = "grayscale";  % grayscale, rgb, rgb_phow
STEP = 4;
BIN_SIZE = 4;
NUM_LAYERS = 3;

% Set up paths to VLFeat functions. 
% See http://www.vlfeat.org/matlab/matlab.html for VLFeat Matlab documentation
% This should work on 32 and 64 bit versions of Windows, MacOS, and Linux
%run('vlfeat/toolbox/vl_setup')

data_path = '../data/';

%This is the list of categories / directories to use. The categories are
%somewhat sorted by similarity so that the confusion matrix looks more
%structured (indoor and then urban and then rural).
categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'House', ...
       'Industrial', 'Stadium', 'Underwater', 'TallBuilding', 'Street', ...
       'Highway', 'Field', 'Coast', 'Mountain', 'Forest'};
   
%This list of shortened category names is used later for visualization.
abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Hou', 'Ind', 'Sta', ...
    'Und', 'Bld', 'Str', 'HW', 'Fld', 'Cst', 'Mnt', 'For'};
    
%number of training examples per category to use. Max is 100. For
%simplicity, we assume this is the number of test cases per category, as
%well.
num_train_per_cat = 100; 

%This function returns cell arrays containing the file path for each train
%and test image, as well as cell arrays with the label of each train and
%test image. By default all four of these arrays will be 1500x1 where each
%entry is a char array (or string).
fprintf('Getting paths and labels for all train and test data\n')
[train_image_paths, test_image_paths, train_labels, test_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat);
%   train_image_paths  1500x1   cell      
%   test_image_paths   1500x1   cell           
%   train_labels       1500x1   cell         
%   test_labels        1500x1   cell          

%% Step 1: Represent each image with the appropriate feature
% Each function to construct features should return an N x d matrix, where
% N is the number of paths passed to the function and d is the 
% dimensionality of each image representation. See the starter code for
% each function for more details.

fprintf('Using %s representation for images\n', FEATURE)

knn_parameters = cell(9, length(VOCAB_SIZES));
knn_parameters{1, 2} = 50;
knn_parameters{1, 3} = 100;
knn_parameters{1, 4} = 150;
knn_parameters{1, 5} = 200;
knn_parameters{1, 6} = 250;
knn_parameters{1, 7} = 300;

knn_parameters{1, 1} = "vocab_size";
knn_parameters{2, 1} = "k1";
knn_parameters{3, 1} = "k3";
knn_parameters{4, 1} = "k5";
knn_parameters{5, 1} = "k7";
knn_parameters{6, 1} = "k9";
knn_parameters{7, 1} = "k11";
knn_parameters{8, 1} = "k13";
knn_parameters{9, 1} = "k15";

svm_parameters = cell(8, length(VOCAB_SIZES));
svm_parameters{1, 2} = 50;
svm_parameters{1, 3} = 100;
svm_parameters{1, 4} = 150;
svm_parameters{1, 5} = 200;
svm_parameters{1, 6} = 250;
svm_parameters{1, 7} = 300;

svm_parameters{1, 1} = "vocab_size";
svm_parameters{2, 1} = "0.00001";
svm_parameters{3, 1} = "0.0001";
svm_parameters{4, 1} = "0.001";
svm_parameters{5, 1} = "0.01";
svm_parameters{6, 1} = "0.1";
svm_parameters{7, 1} = "1";
svm_parameters{8, 1} = "10";

for vs = 1:length(VOCAB_SIZES)
    vocab_size = VOCAB_SIZES(vs);
    switch lower(FEATURE)    
        case 'bag of sift'
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab = build_vocabulary(train_image_paths, vocab_size, STEP, BIN_SIZE, FEATURE_COLOUR); %Also allow for different sift parameters
            save('vocab.mat', 'vocab');
            train_image_feats = get_bags_of_sifts(train_image_paths, vocab_size, STEP, BIN_SIZE, FEATURE_COLOUR); %Allow for different sift parameters
            test_image_feats  = get_bags_of_sifts(test_image_paths, vocab_size, STEP, BIN_SIZE, FEATURE_COLOUR); 
            save('image_feats.mat', 'train_image_feats', 'test_image_feats')
        case 'spatial pyramids'
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab = build_vocabulary(train_image_paths, vocab_size, STEP, BIN_SIZE, FEATURE_COLOUR); %Also allow for different sift parameters
            save('vocab.mat', 'vocab');
            % YOU CODE spatial pyramids method
            train_image_feats = get_spatial_pyramids(train_image_paths, vocab_size, NUM_LAYERS, STEP, BIN_SIZE, FEATURE_COLOUR);
            test_image_feats = get_spatial_pyramids(test_image_paths, vocab_size, NUM_LAYERS, STEP, BIN_SIZE, FEATURE_COLOUR);
            save('image_feats.mat', 'train_image_feats', 'test_image_feats')
    end
    switch lower(CLASSIFIER)
        case 'nearest neighbor'
            for n = 1:length(Ks)
                k = Ks(n);
                predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, k, METRIC);
                accuracy = create_results_webpage( ...
                                train_image_paths, ...
                                test_image_paths, ...
                                train_labels, ...
                                test_labels, ...
                                categories, ...
                                abbr_categories, ...
                                predicted_categories);
                knn_parameters{1+n, 1+vs} = accuracy;
            end
        case 'support vector machine'
            for n = 1:length(LAMBDAS)
                LAMBDA = LAMBDAS(n);
                predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats, LAMBDA);
                accuracy = create_results_webpage( ...
                                train_image_paths, ...
                                test_image_paths, ...
                                train_labels, ...
                                test_labels, ...
                                categories, ...
                                abbr_categories, ...
                                predicted_categories);
                svm_parameters{1+n, 1+vs} = accuracy;
            end
    end
end
toc