%% Step 0: Initialise dataset

img_size = [64 64];
model_name = 'beta_net';
num_chanels = 3;


imds1 = imageDatastore('../data/train',  ...
                       'IncludeSubfolders', true, ...
                       'LabelSource', 'foldernames');
imds2 = imageDatastore('../data/test',  ...
                       'IncludeSubfolders', true, ...
                       'LabelSource', 'foldernames');
                   
imds1.Labels = categorical(lower(cellstr(imds1.Labels)));
imds2.Labels = categorical(lower(cellstr(imds2.Labels)));

imds = imageDatastore(cat(1,imds1.Files,imds2.Files));
imds.Labels = cat(1,imds1.Labels,imds2.Labels);

[trainDataDS, rest] = splitEachLabel(imds, 0.8, 'randomized');
[validateDataDS, testDataDS] = splitEachLabel(rest, 0.5, 'randomized');

%% Step 1: Data augmentation

% imageAugmenter = imageDataAugmenter( ...
%     'RandRotation',[-20 20], ...
%     'RandXShear', [0 45], ...
%     'RandYShear', [0 45], ...
%     'RandXReflection', 1, ...
%     'RandYReflection', 1, ...
%     'RandXTranslation',[-3 3], ...
%     'RandYTranslation',[-3 3]);

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3]);

trainData = augmentedImageDatastore(img_size, trainDataDS, 'DataAugmentation',imageAugmenter);
% trainData = augmentedImageDatastore(img_size, trainDataDS);
validateData = augmentedImageDatastore(img_size, validateDataDS);
testData = augmentedImageDatastore(img_size, testDataDS);


%% Step 2: Define CNN layers

layers = [
    imageInputLayer(horzcat(img_size, num_chanels))
    
    convolution2dLayer(3, 16, 'Padding', 1, 'Name', 'Conv1')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 32, 'Padding', 1, 'Name', 'Conv2')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 64, 'Padding', 1, 'Name', 'Conv3')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 128, 'Padding', 1, 'Name', 'Conv4')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 512, 'Padding', 1, 'Name', 'Conv5')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(4096)
    reluLayer
    dropoutLayer(0.5)
    
    fullyConnectedLayer(4096)
    reluLayer
    dropoutLayer(0.5)
    
    fullyConnectedLayer(15)
    softmaxLayer
    classificationLayer
];


%% Step 4: Set training parameters

options = trainingOptions('sgdm', ...
    'MaxEpochs', 35, ...
    'MiniBatchSize', 64, ... 
    'ValidationData', validateData, ...
    'ValidationFrequency', 15, ...
    'ValidationPatience', 10, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', 'parallel', ...
    'Plots', 'training-progress');

%% Step 5: Start training

net = trainNetwork(trainData, layers, options);
save([model_name '.mat'], 'net')
             

%% Step 6: Generate Confusion matrix and report
% 
% predicted_categories = cellstr(classify(net, testData));
% categories = cellstr(countEachLabel(imds).Label);
% abbr_categories = {'Bed', 'Cst', 'Fld', 'For', 'HW', 'Hou', 'Ind', ...
%     'Kit', 'Liv', 'Mnt', 'Sta', 'Sto', 'Str', 'Bld', 'Und'};
% 
% create_results_webpage(trainDataDS.Files, testDataDS.Files, ...
%                        cellstr(trainDataDS.Labels), cellstr(testDataDS.Labels), ...
%                        categories, abbr_categories, predicted_categories);


%% Step 7: Test on a single file

% % load([model_name '.mat'], 'net')
% % testImg = imread('../data/test/bedroom/sun_abllxrmlmfgdbepz.jpg');
% % testImg = imread('../data/test/forest/sun_agwkzxvlvdxfvjje.jpg');
% % testImg = imread('../data/test/stadium/sun_aadjflxtadgqciqj.jpg');
% % testImg = imread('../data/test/underwater/sun_aalvvifbfqogovsr.jpg');
% % testImg = imread('../data/test/mountain/sun_aawnncfvjepzpmly.jpg');
% inputSize = net.Layers(1).InputSize;
% testImg = imresize(testImg, inputSize(1:2));
% 
% predictedLabel = classify(net, testImg);
% disp(predictedLabel);
% imshow(testImg);


%% Step 8: Plot Model Schematic

% net.plot();


%% Step 9: Visual feature Maps from different layers

% img = imread('../data/test/stadium/sun_aadjflxtadgqciqj.jpg');
% layer_name = 'Conv5';
% 
% act = activations(net, img, layer_name);
% 
% figure;
% montage(act, 'Size', [8 8]);
% title(sprintf('%s activations', layer_name));

