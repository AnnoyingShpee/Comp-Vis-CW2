%% Step 0: Initialise dataset

img_size = [128 128];
model_name = 'alpha_net';
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

% imageAugmenter = imageDataAugmenter( ...
%     'RandRotation',[-20,20], ...
%     'RandXTranslation',[-3 3], ...
%     'RandYTranslation',[-3 3]);

% trainData = augmentedImageDatastore(img_size, trainDataDS, 'DataAugmentation',imageAugmenter);
trainData = augmentedImageDatastore(img_size, trainDataDS);
validateData = augmentedImageDatastore(img_size, validateDataDS);
testData = augmentedImageDatastore(img_size, testDataDS);


%% Step 2: Define CNN layers

layers = [
    imageInputLayer(horzcat(img_size, num_chanels), 'Name', 'Input')
    
    convolution2dLayer(11, 96, 'Name', 'Conv1', 'Stride', 4, 'Padding', 2)
    batchNormalizationLayer('Name', 'BatchNorm1')
    reluLayer('Name', 'ReLU1') 
    maxPooling2dLayer(3, 'Name', 'MaxPool1', 'Stride', 2)
    
    convolution2dLayer(5, 256, 'Name', 'Conv2', 'Stride', 1, 'Padding', 2)
    batchNormalizationLayer('Name', 'BatchNorm2')
    reluLayer('Name', 'ReLU2')
    maxPooling2dLayer(3, 'Name', 'MaxPool2', 'Stride', 2)
    
    convolution2dLayer(3, 384, 'Name', 'Conv3', 'Stride', 1, 'Padding', 1)
    reluLayer('Name', 'ReLU3')
    
    convolution2dLayer(3, 384, 'Name', 'Conv4', 'Stride', 1, 'Padding', 1)
    reluLayer('Name', 'ReLU4')
    
    convolution2dLayer(3, 256, 'Name', 'Conv5', 'Stride', 1, 'Padding', 1)
    reluLayer('Name', 'ReLU5') 
    maxPooling2dLayer(3, 'Name', 'MaxPool3', 'Stride', 2)
    
    fullyConnectedLayer(4096, 'Name', 'FC1')
    reluLayer('Name', 'ReLU6')
    dropoutLayer(0.5, 'Name', 'Dropout1')
    
    fullyConnectedLayer(4096, 'Name', 'FC2')
    reluLayer('Name', 'ReLU7')
    dropoutLayer(0.5, 'Name', 'Dropout2')
    
    fullyConnectedLayer(15, 'Name', 'FC3')
    softmaxLayer('Name', 'Softmax')
    classificationLayer('Name', 'Output')
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









