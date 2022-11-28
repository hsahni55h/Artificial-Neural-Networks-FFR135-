% Classification 

clear;
clc;

%Loading the given data set
xTest2 = loadmnist2();

%given validation set
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadMNIST(3);

figure
imshow(xTest2(:,:,:,9998));

% Visuallise random set of imgs
figure;
perm = randperm(10000,20);
for i = 1:20
subplot(4,5,i);
imshow(xTest2(:,:,:,perm(i)));
end

%get size of images. Important for input layer
size(xTest2(:,:,:,1))

layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%Training Options
options = trainingOptions('sgdm', ...
'InitialLearnRate',0.01, ...
'MaxEpochs',20, ...
'Shuffle','every-epoch', ...
'MiniBatchSize', 256, ...
'ValidationData',{xValid, tValid}, ...
'ValidationFrequency',30, ...
'Verbose',false, ...
'Plots','training-progress');


%Network
net = trainNetwork(xTrain, tTrain,layers,options);

%Clasification of given data
YPred = classify(net,xTest2);
writematrix(YPred,'classifications.csv')