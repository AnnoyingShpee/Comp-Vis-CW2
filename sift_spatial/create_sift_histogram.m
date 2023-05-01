function hist = create_sift_histogram(distance, vocab_size)
%CREATE_SIFT_HISTOGRAM Summary of this function goes here
%   Detailed explanation goes here
% Finds the minimum value of each row which returns [M, I] where 
% M = Value of the smallest value in that row
% I = Column index of the smallest value
[~, indexes] = min(distance, [], 2);
% Returns a histogram of vocab clusters used. 
% 1:vocab_size+1 is the bins from 1 to vocab column size
hist = histcounts(indexes, 1:vocab_size+1);
end

