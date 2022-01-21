%% Demo for the CDE algorithm
clc;
clear;

if ~exist('../results', 'dir')
   mkdir('../results');
end

% Select a data set
dataset = load('../data/ClimateModel');
fns = fieldnames(dataset);
[ X, Y ] = divideTable( dataset.(fns{1}) );

% Set the parameters of CDE
param.DistanceMetric = 'cityblock';
param.NumOfNeighbors = 5;
param.WeightingFunc = 'adaptive';

% Perform an experiment with 10-fold cross-validation
Accuracy = demo1( X, Y, 'KNN', param );
writematrix(Accuracy, '../results/demo1.txt', 'Delimiter', 'tab', 'WriteMode', 'overwrite');
%writeAccuracyToTextFile('demo1.txt', 'ClimateModel', Accuracy);

% Find the embeddings of input data
mappedX1 = demo2( X, Y, param );
writematrix(mappedX1, '../results/demo2.txt', 'Delimiter', 'tab', 'WriteMode', 'overwrite');

% Find the embeddings of input data
mappedX2 = demo3( X, Y );
writematrix(mappedX2, '../results/demo3.txt', 'Delimiter', 'tab', 'WriteMode', 'overwrite');

clear dataset;
clear fns;
clear param;



%% Perform an experiment with 10-fold cross-validation
function [ acc ] = demo1( X, Y, classifier, param )

    predictions = repmat(Y, 1, 2);
    indices = crossvalind('Kfold', Y, 10);
    
    for i = 1:10
        fprintf('%d',i);
        test = (indices == i);
        train = ~test;
                
        param.TestIndices = test;
        mappedX = CDE(X, Y, param);
        
        trainX = mappedX(train,:);
        testX = mappedX(test,:);
        trainY = Y(train,:);

        switch classifier
            case 'CART'
                Mdl = fitctree(trainX, trainY);
            case 'KNN'
                Mdl = fitcknn(trainX, trainY);
            case 'NB'
                % "normal", "mn", "kernel", "mvmn".
                Mdl = fitcnb(trainX, trainY, 'DistributionNames', 'normal');
            case 'SVM'
                % "linear", "gaussian", "rbf", "polynomial"
                t = templateSVM('Standardize', true, 'KernelFunction', 'linear');
                Mdl = fitcecoc(trainX, trainY, 'Learners', t);
        end
        % Predict the output of an identified model
        predictions(test, 2) = predict(Mdl, testX);
    end
    acc = sum(predictions(:,1) == predictions(:,2))*100/length(Y);
end



%% Find the embeddings of input data
function [ mappedX ] = demo2( X, Y, param )

    mappedX = CDE(X, Y, param);
end



%% Find the embeddings of input data
function [ mappedX ] = demo3( X, Y )

    param.DistanceMetric = 'infogain';
    param.NumOfNeighbors = 3;
    param.WeightingFunc = 'lnln';
    
    mappedX = CDE(X, Y, param);
end



%% Separate the dataset into the input matrix and the output vector
function [ X, Y ] = divideTable( DATASET )

    if istable(DATASET)
        X = table2array(DATASET(:,1:end-1));        
        Y = categorical(DATASET.Class);
    else
        error('The parameter must be a table, not a %s.', class(DATASET));
    end
end


