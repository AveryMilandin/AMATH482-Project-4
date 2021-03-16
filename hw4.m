clear all; close all;

gunzip('train-images-idx3-ubyte.gz');
gunzip('train-labels-idx1-ubyte.gz');
gunzip('t10k-images-idx3-ubyte.gz');
gunzip('t10k-labels-idx1-ubyte.gz');
%% parse data

[images, labels] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');

[testImages, testLabels] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');

%% reshape
reshapedImages = im2double(reshape(images, [28*28, length(images)]));
reshapedTestImages = im2double(reshape(testImages, [28*28, length(testImages)]));

%% Create Singular Value Plot
sVals = diag(S);
plot([1:50],sVals(1:50), '.', 'MarkerSize', 10);
xlabel('Index')
ylabel('Value')
title('First 50 Singular Values')

%% Determine easiest to classify and hardest to classify
easiest = [];
hardest = [];
lowestAccuracy = 100;
highestAccuracy = 0;
allAccuracy = [];

for j = 0:9
    for k = (j+1):9
        [U,S,V,threshold,w] = digit_trainer(j,k,images,labels);
        [digit1, digit2] = extract_data(j,k,testImages, testLabels);
        Test_wave = dc_wavelet([digit1, digit2]);
        TestMat = U'*Test_wave; % PCA projection
        pval = w'*TestMat;
        ResVec = (pval > threshold);
        solutions = [zeros(1, length(digit1)) ones(1, length(digit2))];
        incorrect = abs(ResVec - solutions);
        accuracy = 100 - sum(incorrect)/length(solutions)*100;
        if (accuracy > highestAccuracy)
            highestAccuracy = accuracy;
            easiest = [j,k];
        end
        if (accuracy < lowestAccuracy)
            lowestAccuracy = accuracy;
            hardest = [j,k];
        end
        allAccuracy = [allAccuracy; [j, k, accuracy]]
    end
end

%% Train Model to distinguish between 1, 0, and 8
[U,S,V,w,mFirst,mSecond,mThird] = three_digit_trainer(0, 1, 8, images, labels);
%% Classify test data
[Zeros, Ones, eights] = extract_data_3(0, 1,8,testImages, testLabels);
Test_wave = dc_wavelet([Zeros, Ones, eights]);
TestMat = U'*Test_wave; % PCA projection
pval = w'*TestMat;
ResVec = zeros(1, length(pval));
solutions = [zeros(1, length(Zeros)) ones(1, length(Ones)) 2*ones(1,length(eights))];
for j=1:length(pval)
    diff1 = abs(pval(j)-mFirst);
    diff2 = abs(pval(j)-mSecond);
    diff3 = abs(pval(j)-mThird);
    minVal = min([diff1,diff2,diff3]);
    if (minVal == diff1)
        ResVec(j) = 0;
    elseif (minVal == diff2)
        ResVec(j) = 1;
    else 
        ResVec(j) = 2;
    end
end
incorrectZeros = (ResVec(1:length(Zeros)) ~= solutions(1:length(Zeros)));
incorrectOnes = (ResVec(length(Zeros)+1:length(Zeros)+length(Ones)) ~= solutions(length(Zeros)+1:length(Zeros)+length(Ones)));
incorrectEights = (ResVec(length(Zeros)+length(Ones)+1:length(Zeros)+length(Ones)+length(eights)) ~= solutions(length(Zeros)+length(Ones)+1:length(Zeros)+length(Ones)+length(eights)));
zerosAccuracy = 100 - sum(incorrectZeros)/length(Zeros)*100;
onesAccuracy = 100 - sum(incorrectOnes)/length(Ones)*100;
eightsAccuracy = 100 - sum(incorrectEights)/length(eights)*100;
overallAccuracy = 100 - (sum(incorrectZeros) + sum(incorrectOnes) + sum(incorrectEights))/(length(Zeros) + length(Ones) + length(eights))*100

%% Create Binary Classification Tree
data = array2table([reshapedImages' labels]);
tree=fitctree(data, 'Var785');

%% Use binary tree to predict test images
testData = array2table(reshapedTestImages');
predicted = predict(tree, testData);
incorrect = (predicted ~= testLabels);
accuracy = 100 - sum(incorrect)/length(testLabels)*100;

%% Create and use binary tree to classify easiest and hardest digit pairs
[trainZeros, trainOnes] = extract_data(0,1,images, labels);
data = array2table([[trainZeros, trainOnes]; [zeros(1,length(trainZeros)), ones(1,length(trainOnes))]]');
BTree01 = fitctree(data, 'Var785');
%%
[trainThrees, trainFives] = extract_data(3,5,images, labels);
data = array2table([[trainThrees, trainFives]; [3*ones(1,length(trainThrees)), 5*ones(1,length(trainFives))]]');
BTree35 = fitctree(data, 'Var785');
%%
[Zeros, Ones] = extract_data(0,1,testImages, testLabels);
testData = array2table([Zeros Ones]');
predicted = predict(BTree01, testData);
zeroOneLabels = [zeros(length(Zeros), 1); ones(length(Ones), 1)];
incorrect = (predicted ~= zeroOneLabels);
accuracyEasy = 100 - sum(incorrect)/length(incorrect)*100
%%
[threes, fives] = extract_data(3,5,testImages, testLabels);
testData = array2table([threes fives]');
predicted = predict(BTree35, testData);
threeFiveLabels = [3*ones(length(threes), 1); 5*ones(length(fives), 1)];
incorrect = (predicted ~= threeFiveLabels);
accuracyHard = 100 - sum(incorrect)/length(incorrect)*100

%% Create SVM classifier
data = array2table([reshapedImages' labels]);
model = fitcecoc(data, 'Var785');

%% Use SVM classifier to predict test images
testData = array2table(reshapedTestImages');
predicted = predict(model, testData);
incorrect = (predicted ~= testLabels);
accuracy = 100 - sum(incorrect)/length(testLabels)*100;

%% Create and use SVM classifier to classify easiest and hardest digit pairs
[trainZeros, trainOnes] = extract_data(0,1,images, labels);
data = array2table([[trainZeros, trainOnes]; [zeros(1,length(trainZeros)), ones(1,length(trainOnes))]]');
SVM01 = fitcsvm(data, 'Var785');
%%
[trainThrees, trainFives] = extract_data(3,5,images, labels);
data = array2table([[trainThrees, trainFives]; [3*ones(1,length(trainThrees)), 5*ones(1,length(trainFives))]]');
SVM35 = fitcsvm(data, 'Var785');
%%
[Zeros, Ones] = extract_data(0,1,testImages, testLabels);
testData = array2table([Zeros Ones]');
predicted = predict(SVM01, testData);
zeroOneLabels = [zeros(length(Zeros), 1); ones(length(Ones), 1)];
incorrect = (predicted ~= zeroOneLabels);
accuracyEasy = 100 - sum(incorrect)/length(incorrect)*100
%%
[threes, fives] = extract_data(3,5,testImages, testLabels);
testData = array2table([threes fives]');
predicted = predict(SVM35, testData);
threeFiveLabels = [3*ones(length(threes), 1); 5*ones(length(fives), 1)];
incorrect = (predicted ~= threeFiveLabels);
accuracyHard = 100 - sum(incorrect)/length(incorrect)*100

%% Trainer function
function [U,S,V,threshold,w] = digit_trainer(dOne, dTwo, images, labels)
[firstDigit, secondDigit] = extract_data(dOne, dTwo, images, labels);
digits = dc_wavelet([firstDigit, secondDigit]);
features = 30;
[U,S,V] = svd(digits, 'econ');
U = U(:,1:features);
numFirst = size(firstDigit, 2);
numSecond = size(secondDigit, 2);
digitsProj = S*V';
firstProj = digitsProj(1:features, 1:numFirst);
secondProj = digitsProj(1:features, numFirst+1:numFirst + numSecond);
mFirst = mean(firstProj,2);
mSecond = mean(secondProj,2);
Sw = 0; % within class variances
for k = 1:numFirst
    Sw = Sw + (firstProj(:,k) - mFirst)*(firstProj(:,k) - mFirst)';
end
for k = 1:numSecond
    Sw =  Sw + (secondProj(:,k) - mSecond)*(secondProj(:,k) - mSecond)';
end
Sb = (mFirst-mSecond)*(mFirst-mSecond)'; % between class
[V2,D] = eig(Sb,Sw);
[lambda,ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);
vFirst = w'*firstProj;
vSecond = w'*secondProj;
if mean(vFirst) > mean(vSecond)
    w = -w;
    vFirst = -vFirst;
    vSecond = -vSecond;
end
sortFirst = sort(vFirst);
sortSecond = sort(vSecond);
t1 = length(sortFirst);
t2 = 1;
while sortFirst(t1) > sortSecond(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold = (sortFirst(t1) + sortSecond(t2))/2;
end

%% Trainer function for 3 digits
function [U,S,V,w,mFirst,mSecond,mThird] = three_digit_trainer(dOne, dTwo, dThree, images, labels)
[firstDigit, secondDigit, thirdDigit] = extract_data_3(dOne, dTwo, dThree, images, labels);
digits = dc_wavelet([firstDigit, secondDigit, thirdDigit]);
features = 30;
[U,S,V] = svd(digits, 'econ');
U = U(:,1:features);
numFirst = size(firstDigit, 2);
numSecond = size(secondDigit, 2);
numThird = size(thirdDigit, 2);
digitsProj = S*V';
firstProj = digitsProj(1:features, 1:numFirst);
secondProj = digitsProj(1:features, numFirst+1:numFirst + numSecond);
thirdProj = digitsProj(1:features, numFirst + numSecond+1:numFirst + numSecond + numThird);
mFirst = mean(firstProj,2);
mSecond = mean(secondProj,2);
mThird = mean(thirdProj, 2);
Sw = 0; % within class variances
for k = 1:numFirst
    Sw = Sw + (firstProj(:,k) - mFirst)*(firstProj(:,k) - mFirst)';
end
for k = 1:numSecond
    Sw =  Sw + (secondProj(:,k) - mSecond)*(secondProj(:,k) - mSecond)';
end
for k = 1:numThird
    Sw =  Sw + (thirdProj(:,k) - mThird)*(thirdProj(:,k) - mThird)';
end
Sb = (mFirst-mSecond)*(mFirst-mSecond)' + (mFirst-mThird)*(mFirst-mThird)' + (mSecond-mThird)*(mSecond-mThird)'; % between class
[V2,D] = eig(Sb,Sw);
[lambda,ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);
vFirst = w'*firstProj;
vSecond = w'*secondProj;
vThird = w'*thirdProj;
mFirst = mean(vFirst);
mSecond = mean(vSecond);
mThird = mean(vThird);
end
%% data extraction function

function [mat1, mat2] = extract_data(val1, val2, mat, labels)
    reshapedMat = reshape(mat, 28*28, length(mat));
    mat1 = [];
    mat2 = [];
    for j = 1: length(reshapedMat)
        if (labels(j) == val1)
            mat1 = [mat1, reshapedMat(:,j)];
        elseif (labels(j) == val2)
            mat2 = [mat2, reshapedMat(:,j)];
        end
    end
    mat1 = im2double(mat1);
    mat2 = im2double(mat2);
end

%% 3 digit data extraction function

function [mat1, mat2, mat3] = extract_data_3(val1, val2, val3, mat, labels)
    reshapedMat = reshape(mat, 28*28, length(mat));
    mat1 = [];
    mat2 = [];
    mat3 = [];
    for j = 1: length(reshapedMat)
        if (labels(j) == val1)
            mat1 = [mat1, reshapedMat(:,j)];
        elseif (labels(j) == val2)
            mat2 = [mat2, reshapedMat(:,j)];
        elseif (labels(j) == val3)
            mat3 = [mat3, reshapedMat(:,j)];
        end
    end
    mat1 = im2double(mat1);
    mat2 = im2double(mat2);
    mat3 = im2double(mat3);
end
