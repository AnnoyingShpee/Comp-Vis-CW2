function [predictions] = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, K, dist_measure)
%NEAREST_NEIGHBOR_CLASSIFY Summary of this function goes here
%   train_image_feats = training image feature vectors   (N x M)
%   train_labels = training image labels  (N x 1)
%   test_image_feats = test image feature vectors   (n x m)
%   K = Number of nearest neighbous
%   dist_measure =  Type of measure to calculate distance between a test
%                   image feature and train image feature

test_size = size(test_image_feats);
% Initialise vector of empty strings 
predictions = strings(test_size(1), 1);
parfor i = 1:test_size(1)
    % Get a feature vector
    test_feature_vector = test_image_feats(i,:);
    % Calculate the distance between each test feature vector and train
    % feature vectors
    distances = pdist2(train_image_feats, test_feature_vector, dist_measure);
    %% Get the K nearest neighbours
    % Sort the distances by ascending order
    % sort_dist and sort_index are both size (1 x N)
    [sort_dist, sort_index] = sort(distances(:,1));
    % Get the first K elements in sort_index. These are the index of the
    % nearest neighbours in train_labels
    k_nearest_index = sort_index(1:K);
    % Initialise vector of empty strings for the test image labels
    k_labels = strings(K, 1);
    % Store the K labels 
    for k = 1:K
        k_labels(k,:) = train_labels{k_nearest_index(k)};
    end
    %% Assign a label to the test image feature vector
    % Group the labels and count them
    % count is the vector of number of elements in a group
    % element is the group value
    [count, element] = groupcounts(k_labels(:,1));
    % max() returns [highest_count, index] 
    % highest_count = the highest value in a vector 
    % index = index of the highest value in the vector
    [highest_count, index] = max(count);
    % Get a vector of indexes that have the same highest value 
    multi_highest_indexes = find(count==highest_count);
    % If there is only one index, just assign the label to the test image
    if size(multi_highest_indexes, 1) == 1
        predictions(i,:) = element(index);
    % Else, calculate the total distance of each label and assign the test
    % image with a label with the smallest total distance
    else
        k_dist = sort_dist(1:K);
        sums = zeros(size(multi_highest_indexes, 1), 1);
        sums_labels = strings(size(multi_highest_indexes, 1), 1);
        for s = 1:size(multi_highest_indexes, 1)
            matches = ismember(k_labels(:,1), element(multi_highest_indexes(s)));
            sums(s, 1) = sum(k_dist(matches,1));
            sums_labels(s, 1) = element(multi_highest_indexes(s));
        end
        [~, index] = min(sums, [], 1);
        predictions(i, :) = sums_labels(index);
    end
end
end

