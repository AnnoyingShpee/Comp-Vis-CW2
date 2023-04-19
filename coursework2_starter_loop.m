% Michal Mackiewicz, UEA 
% This code has been adapted from the code 
% prepared by James Hays, Brown University
tic
%% Step 0: Set up parameters, vlfeat, category list, and image paths.
% FEATURE = 'tiny image';
% FEATURE = 'colour histogram';
% FEATURE = 'bag of sift';
FEATURE = 'spatial pyramids';

% CLASSIFIER = 'nearest neighbor';
CLASSIFIER = 'support vector machine';

% QUANTISATION = 16; % 8, 16, 32, 64
% COLOUR_SPACE = "rgb";
% IMG_SIZE = 16; % 8, 16, 32, 64
% K = 5;
METRIC = "euclidean";
% STEP = 10;
% % Note: Default value of size in vl_dsift is 3
% BIN_SIZE = 16;
% VOCAB_SIZE = 50;
% NUM_LAYERS = 2;
% FEATURE_COLOUR_TYPE = "grayscale"; % grayscale, rgb
% LAMBDA = 0.00001; % 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10

% To test parameters
FEATURE_COLOURS = ["grayscale", "rgb"];
STEPS = 5:5:50;
BIN_SIZES = [8, 16, 32, 64];
VOCAB_SIZES = 50:50:300;
LAYERS = [1, 2, 3];
Ks = 1:2:15;
LAMBDAS = 10.^(-5:2);

FEATURE_COLOUR = "grayscale";
STEP = 4;
BIN_SIZE = 4;
LAMBDA = 0.00001;
LAYER = 3;

% best_accuracy = 0;
% best_parameters = cell(7, 2);
% best_parameters{1, 1} = "FEATURE_COLOUR_TYPE";
% best_parameters{2, 1} = "STEP";
% best_parameters{3, 1} = "BIN_SIZE";
% best_parameters{4, 1} = "VOCAB_SIZE";
% best_parameters{5, 1} = "LAYERS";
% switch CLASSIFIER
%     case "nearest neighbor"
%         best_parameters{6, 1} = "K";
%     case "support vector machine"
%         best_parameters{6, 1} = "LAMBDA";
% end
% best_parameters{7, 1} = "BEST_ACCURACY";
% 
% best_parameters{1, 2} = [];
% best_parameters{2, 2} = [];
% best_parameters{3, 2} = [];
% best_parameters{4, 2} = [];
% best_parameters{5, 2} = [];
% best_parameters{6, 2} = [];
% best_parameters{7, 2} = [];
% 
% worst_accuracy = 100;
% worst_parameters = cell(7, 2);
% worst_parameters{1, 1} = "FEATURE_COLOUR_TYPE";
% worst_parameters{2, 1} = "STEP";
% worst_parameters{3, 1} = "BIN_SIZE";
% worst_parameters{4, 1} = "VOCAB_SIZE";
% worst_parameters{5, 1} = "LAYERS";
% switch CLASSIFIER
%     case "nearest neighbor"
%         worst_parameters{6, 1} = "K";
%     case "support vector machine"
%         worst_parameters{6, 1} = "LAMBDA";
% end
% worst_parameters{7, 1} = "WORST_ACCURACY";
% 
% worst_parameters{1, 2} = [];
% worst_parameters{2, 2} = [];
% worst_parameters{3, 2} = [];
% worst_parameters{4, 2} = [];
% worst_parameters{5, 2} = [];
% worst_parameters{6, 2} = [];
% worst_parameters{7, 2} = [];
% 
% mid_parameters = cell(7, 2);
% mid_parameters{1, 1} = "FEATURE_COLOUR_TYPE";
% mid_parameters{2, 1} = "STEP";
% mid_parameters{3, 1} = "BIN_SIZE";
% mid_parameters{4, 1} = "VOCAB_SIZE";
% mid_parameters{5, 1} = "LAYERS";
% switch CLASSIFIER
%     case "nearest neighbor"
%         mid_parameters{6, 1} = "K";
%     case "support vector machine"
%         mid_parameters{6, 1} = "LAMBDA";
% end
% mid_parameters{7, 1} = "ACCURACY";
% 
% mid_parameters{1, 2} = [];
% mid_parameters{2, 2} = [];
% mid_parameters{3, 2} = [];
% mid_parameters{4, 2} = [];
% mid_parameters{5, 2} = [];
% mid_parameters{6, 2} = [];
% mid_parameters{7, 2} = [];

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

parameters = cell(2, 2);
parameters{1, 1} = "vocab_size";
parameters{2, 1} = "accuracy";
parameters{1, 2} = [];
parameters{2, 2} = [];

for vs = 1: length(VOCAB_SIZES)
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
            predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, K, METRIC);
            accuracy = create_results_webpage( ...
                            train_image_paths, ...
                            test_image_paths, ...
                            train_labels, ...
                            test_labels, ...
                            categories, ...
                            abbr_categories, ...
                            predicted_categories);
            parameters{1, 2} = [parameters{1, 2}, vocab_size];
            parameters{2, 2} = [parameters{2, 2}, accuracy];
        case 'support vector machine'
            predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats, LAMBDA);
            accuracy = create_results_webpage( ...
                            train_image_paths, ...
                            test_image_paths, ...
                            train_labels, ...
                            test_labels, ...
                            categories, ...
                            abbr_categories, ...
                            predicted_categories);
            parameters{1, 2} = [parameters{1, 2}, vocab_size];
            parameters{2, 2} = [parameters{2, 2}, accuracy];
    end
end


% for fc = 1:length(FEATURE_COLOURS)
%     FEATURE_COLOUR_TYPE = FEATURE_COLOURS(fc);
%     for vs = 1:length(VOCAB_SIZES)
%         vocab_size = VOCAB_SIZES(vs);
%         for ss = 1:length(STEPS)
%             STEP = STEPS(ss);
%             for bs = 1:length(BIN_SIZES)
%                 BIN_SIZE = BIN_SIZES(bs);
%                 switch lower(FEATURE)    
%                     case 'bag of sift'
%                         % YOU CODE build_vocabulary.m
%                         if  prev_feature_colour ~= FEATURE_COLOUR_TYPE && prev_vocab_size ~= vocab_size && prev_step ~= STEP && prev_bin_size ~= BIN_SIZE
%                             fprintf('No existing dictionary found. Computing one from training images\n')
%                             prev_feature_colour = FEATURE_COLOUR_TYPE;
%                             prev_vocab_size = vocab_size;
%                             prev_step = STEP;
%                             prev_bin_size = BIN_SIZE;
%                             vocab = build_vocabulary(train_image_paths, vocab_size, STEP, BIN_SIZE, FEATURE_COLOUR_TYPE); %Also allow for different sift parameters
%                             save('vocab.mat', 'vocab');
%                         end
%                         
%                         % YOU CODE get_bags_of_sifts.m
%                         train_image_feats = get_bags_of_sifts(train_image_paths, STEP, BIN_SIZE, FEATURE_COLOUR_TYPE); %Allow for different sift parameters
%                         test_image_feats  = get_bags_of_sifts(test_image_paths, STEP, BIN_SIZE, FEATURE_COLOUR_TYPE); 
%                     case 'spatial pyramids'
%                         if prev_feature_colour ~= FEATURE_COLOUR_TYPE && prev_vocab_size ~= vocab_size && prev_step ~= STEP && prev_bin_size ~= BIN_SIZE
%                             fprintf('No existing dictionary found. Computing one from training images\n')
%                             prev_feature_colour = FEATURE_COLOUR_TYPE;
%                             prev_vocab_size = vocab_size;
%                             prev_step = STEP;
%                             prev_bin_size = BIN_SIZE;
%                             vocab = build_vocabulary(train_image_paths, vocab_size, STEP, BIN_SIZE, FEATURE_COLOUR_TYPE); %Also allow for different sift parameters
%                             save('vocab.mat', 'vocab');
%                         end
%                         % YOU CODE spatial pyramids method
%                         for nl = 1:length(LAYERS)
%                             NUM_LAYERS = LAYERS(nl);
%                             train_image_feats = get_spatial_pyramids(train_image_paths, NUM_LAYERS, STEP, BIN_SIZE, FEATURE_COLOUR_TYPE);
%                             test_image_feats = get_spatial_pyramids(test_image_paths, NUM_LAYERS, STEP, BIN_SIZE, FEATURE_COLOUR_TYPE);
%                         end
%                 end
%                 switch lower(CLASSIFIER)    
%                     case 'nearest neighbor'
%                     %Here, you need to reimplement nearest_neighbor_classify. My P-code
%                     %implementation has k=1 set. You need to allow for varying this
%                     %parameter.
%                         
%                     %This function will predict the category for every test image by finding
%                     %the training image with most similar features. Instead of 1 nearest
%                     %neighbor, you can vote based on k nearest neighbors which will increase
%                     %performance (although you need to pick a reasonable value for k).
%                     
%                     % image_feats is an N x d matrix, where d is the dimensionality of the
%                     %  feature representation.
%                     % train_labels is an N x 1 cell array, where each entry is a string
%                     %  indicating the ground truth category for each training image.
%                     % test_image_feats is an M x d matrix, where d is the dimensionality of the
%                     %  feature representation. You can assume M = N unless you've modified the
%                     %  starter code.
%                     % predicted_categories is an M x 1 cell array, where each entry is a string
%                     %  indicating the predicted category for each test image.
%                     % Useful functions: pdist2 (Matlab) and vl_alldist2 (from vlFeat toolbox)
%                         for k_index = 1:length(Ks)
%                             K = Ks(k_index);
%                             predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, K, METRIC);
%                             accuracy = create_results_webpage( ...
%                                     train_image_paths, ...
%                                     test_image_paths, ...
%                                     train_labels, ...
%                                     test_labels, ...
%                                     categories, ...
%                                     abbr_categories, ...
%                                     predicted_categories)
%                             if accuracy > 0.4
%                                 best_parameters{1, 2} = FEATURE_COLOUR_TYPE;
%                                 best_parameters{2, 2} = STEP;
%                                 best_parameters{3, 2} = BIN_SIZE;
%                                 best_parameters{4, 2} = vocab_size;
%                                 best_parameters{5, 2} = NUM_LAYERS;
%                                 best_parameters{6, 2} = K;
%                                 best_parameters{7, 2} = accuracy;
%                             end
%                             if accuracy < 0.2
%                                 worst_parameters{1, 2} = FEATURE_COLOUR_TYPE;
%                                 worst_parameters{2, 2} = STEP;
%                                 worst_parameters{3, 2} = BIN_SIZE;
%                                 worst_parameters{4, 2} = vocab_size;
%                                 worst_parameters{5, 2} = NUM_LAYERS;
%                                 worst_parameters{6, 2} = K;
%                                 worst_parameters{7, 2} = accuracy;
%                             end
%                             if accuracy >= 0.3 && accuracy <= 0.4
%                                 mid_parameters{1, 2} = [mid_parameters{1,2}, FEATURE_COLOUR_TYPE];
%                                 mid_parameters{2, 2} = [mid_parameters{2,2}, STEP];
%                                 mid_parameters{3, 2} = [mid_parameters{3,2}, BIN_SIZE];
%                                 mid_parameters{4, 2} = [mid_parameters{4,2}, vocab_size];
%                                 mid_parameters{5, 2} = [mid_parameters{5,2}, NUM_LAYERS];
%                                 mid_parameters{6, 2} = [mid_parameters{6,2}, K];
%                                 mid_parameters{7, 2} = [mid_parameters{7,2}, accuracy];
%                             end
%                         end
%                     case 'support vector machine'
%                         for l = 1:length(LAMBDAS)
%                             LAMBDA = LAMBDAS(l);
%                             predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats, LAMBDA);
%                             accuracy = create_results_webpage( ...
%                                     train_image_paths, ...
%                                     test_image_paths, ...
%                                     train_labels, ...
%                                     test_labels, ...
%                                     categories, ...
%                                     abbr_categories, ...
%                                     predicted_categories)
%                             if accuracy > 0.5
%                                 best_parameters{1, 2} = [best_parameters{1,2}, FEATURE_COLOUR_TYPE];
%                                 best_parameters{2, 2} = [best_parameters{2,2}, STEP];
%                                 best_parameters{3, 2} = [best_parameters{3,2}, BIN_SIZE];
%                                 best_parameters{4, 2} = [best_parameters{4,2}, vocab_size];
%                                 best_parameters{5, 2} = [best_parameters{5,2}, NUM_LAYERS];
%                                 best_parameters{6, 2} = [best_parameters{6,2}, LAMBDA];
%                                 best_parameters{7, 2} = [best_parameters{7,2}, accuracy];
%                             end
%                             if accuracy < 0.2
%                                 worst_parameters{1, 2} = [worst_parameters{1,2}, FEATURE_COLOUR_TYPE];
%                                 worst_parameters{2, 2} = [worst_parameters{2,2}, STEP];
%                                 worst_parameters{3, 2} = [worst_parameters{3,2}, BIN_SIZE];
%                                 worst_parameters{4, 2} = [worst_parameters{4,2}, vocab_size];
%                                 worst_parameters{5, 2} = [worst_parameters{5,2}, NUM_LAYERS];
%                                 worst_parameters{6, 2} = [worst_parameters{6,2}, LAMBDA];
%                                 worst_parameters{7, 2} = [worst_parameters{7,2}, accuracy];
%                             end
%                             if accuracy >= 0.3 && accuracy <= 0.4
%                                 mid_parameters{1, 2} = [mid_parameters{1,2}, FEATURE_COLOUR_TYPE];
%                                 mid_parameters{2, 2} = [mid_parameters{2,2}, STEP];
%                                 mid_parameters{3, 2} = [mid_parameters{3,2}, BIN_SIZE];
%                                 mid_parameters{4, 2} = [mid_parameters{4,2}, vocab_size];
%                                 mid_parameters{5, 2} = [mid_parameters{5,2}, NUM_LAYERS];
%                                 mid_parameters{6, 2} = [mid_parameters{6,2}, LAMBDA];
%                                 mid_parameters{7, 2} = [mid_parameters{7,2}, accuracy];
%                             end
%                         end
%                 end
%             end
%         end
%     end
% end

toc