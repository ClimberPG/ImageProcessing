%%
%% Template for PCA-based face recognition
%%

fprintf('Loading data...\n');
load('ORL_32x32.mat'); % matrix with face images (fea) and labels (gnd)
load('train_test_orl.mat'); % training and test indices (trainIdx, testIdx)
fea = double(fea / 255);
image_dims = [32,32];

% display_faces(fea,10,10);
% title('Face data');
% pause;

% partition the data into training and test subset
n_train = size(trainIdx,1);
n_test = size(testIdx,1);
train_data = fea(trainIdx,:);
train_label = gnd(trainIdx,:);
test_data = fea(testIdx,:);
test_label = gnd(testIdx,:);

% Calculate the average face
mean_train = mean(train_data);
shifted_train = train_data - mean_train;
shifted_test = test_data - mean_train;

fprintf('Running PCA...\n');
% find principal components (use pca function)
components = pca(shifted_train, 'Economy', false);

% figure;
% display_faces(components,10,10); 
% title ('Top principal components');

num_eigenfaces = 100;
% low-dim coefficients for training data (projection onto components)
train_data_pca = shifted_train * components(1:num_eigenfaces, :)';
% high-dimensional faces reconstructed from the low-dim coefficients
train_data_reconstructed = train_data_pca * components(1:num_eigenfaces, :);

% display_faces(train_data_reconstructed,10,10);
% title('Reconstructed 100 train data');
% pause;

fprintf('Projecting test data...\n');
% low-dim coefficients for test data
test_data_pca = shifted_test * components(1:num_eigenfaces, :)';
% high-dimensional reconstructed test faces
test_data_reconstructed = test_data_pca * components(1:num_eigenfaces, :);

% classification of face images

fprintf('Running nearest-neighbor classifier...\n');

[nn_ind, estimated_label] = classifyNN(test_data_pca, train_data_pca, train_label);
% output of nearest-neighbor classifier:
% nearest neighbor training indices for each training point and 
% estimated labels (corresponding to labels of the nearest neighbors)
    
fprintf('Classification rate: %f\n', sum(estimated_label == test_label)/n_test);

% display complete test results (for debugging)

for batch = 1:10
    clf;
    for i = 1:12
        test_ind = (batch-1)*12+i;
        subplot(4,12,i);
        imshow(reshape(test_data(test_ind,:),[32 32]),[]);
        if i == 6
            title('Orig. test img.');
        end
        subplot(4,12,i+12);
        imshow(reshape(test_data_reconstructed(test_ind,:),[32 32]),[]);
        if i == 6
            title('Low-dim test img.');
        end
        subplot(4,12,i+24);
        imshow(reshape(train_data_reconstructed(nn_ind(test_ind),:),[32 32]),[]);
        if i == 6
            title('Low-dim nearest neighbor');
        end
        subplot(4,12,i+36);
        imshow(reshape(train_data(nn_ind(test_ind),:),[32 32]),[]);
        if i == 6
            title('Orig. nearest neighbor');
        end
        if estimated_label(test_ind)~=test_label(test_ind)
            xlabel('incorrect');
        end
    end
    pause;
end

