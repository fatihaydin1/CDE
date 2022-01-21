%% Compute the Information Gain per features, i.e. Compute IG = H(Y) - H(Y|X)
function [ gains ] = InformationGain( features, outcome )
    
    % Remove the test instances from the data set
    features(isundefined(outcome),:) = [];
    outcome(isundefined(outcome)) = [];

    % Compute the entropy of the class 
    outcomeEntropy = H_Theorem(outcome);
    
    d = size(features, 2);
    
    % Compute entropy per features.
    gains = zeros(d, 1);
    for i = 1 : d
        gains(i) = outcomeEntropy - entropyOfAttribute(features(:,i), outcome);
    end
end



%% Compute the entropy of the class, i.e. Compute H(Y)
function [ H ] = H_Theorem( p )
    % Create a frequency table
    % The first column in 'T' contains the unique string values in 'p'.
    % The second is the number of instances of each value.
    % The last column contains the percentage of each value.
    T = tabulate(p);
    
    % Get a vector including the number of instances per class.
    m = cell2mat(T(:,2));
    
    % Compute probabilities per class.
    probPerClass = m ./ sum(m);
    
    % Ignore bins with 0 probability such as 0*log(0)=0.
    probPerClass = probPerClass(probPerClass > 0);
    
    % Compute Boltzmann's H Theorem.
    H = -sum(probPerClass .* log2(probPerClass));
end



%% Compute the entropy per attributes, i.e. Compute H(Y|X)
function [ eoa ] = entropyOfAttribute( X, outcome )
    eoa = 0;
    % It partitions the values in 'X' into bins, and is an array of the
    % same size as 'X' whose elements are the 'BIN' indices for the
    % corresponding elements in 'X'.
    [~, ~, BIN] = histcounts(X, 'BinMethod', 'sturges');
    
    for c = unique(BIN)'
        idx = (BIN == c);
        p = sum(idx) / length(X);
        if p > 0
            eoa = eoa + p * H_Theorem(outcome(idx));
        end
    end  
end


