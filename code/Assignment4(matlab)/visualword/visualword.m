run('~/src/vlfeat/vlfeat-0.9.21/toolbox/vl_setup')

% basic config
conf.calDir = '/Users/bill/DS/CV/Assignment4/visual word/101_ObjectCategories/';
conf.numTrain = 15 ;
conf.numTest = 15 ;
conf.numClasses = 102 ;
conf.numWords = 50 ;

% setup data
classes = dir(conf.calDir) ;
classes = classes([classes.isdir]) ;
classes = {classes(3:conf.numClasses+2).name} ;

images = {} ;
imageClass = {} ;
for ci = 1:length(classes)
    ims = dir(fullfile(conf.calDir, classes{ci}, '*.jpg'))' ;
    ims = vl_colsubset(ims, conf.numTrain + conf.numTest) ;
    ims = cellfun(@(x)fullfile(classes{ci},x),{ims.name},'UniformOutput',false) ;
    images = {images{:}, ims{:}} ;
    imageClass{end+1} = ci * ones(1,length(ims)) ;
end
selTrain = find(mod(0:length(images)-1, conf.numTrain+conf.numTest) < conf.numTrain) ;
selTest = setdiff(1:length(images), selTrain) ;
imageClass = cat(2, imageClass{:}) ;

% train the vocabulary
selTrainFeats = vl_colsubset(selTrain, 30) ;
descrs = {} ;

parfor ii = 1:length(selTrainFeats)
    img = imread(fullfile(conf.calDir, images{selTrainFeats(ii)})) ;
    if (size(img, 3) == 3)
        img = single(double(rgb2gray(img)));
    else
        img = single(double(img));
    end
    [~, descrs{ii}] = vl_sift(img) ;
end

descrs = vl_colsubset(cat(2, descrs{:}), 10e4) ;
descrs = single(descrs) ;

% cluster the descriptors by k-means
vocab = vl_kmeans(descrs, conf.numWords);

% Test on selTestFeats
selTestFeats = vl_colsubset(selTest, 2);
matches = {};
scores = {};

for ii = 1:length(selTestFeats)
    img = imread(fullfile(conf.calDir, images{selTestFeats(ii)})) ;
    if (size(img, 3) == 3)
        img = single(double(rgb2gray(img)));
    else
        img = single(double(img));
    end
    [~, d] = vl_sift(img);
    d = single(d);
    
    for j = 1:length(vocab)
        [matches{ii}, scores{ii}] = vl_ubcmatch(d, vocab);
    end
end

% Compute and plot the visual word histograms
dayun1 = imread('dayun1.jpg');
dayun2 = imread('dayun2.jpg');
lab = imread('lab.jpg');

dayun1 = single(rgb2gray(dayun1)) ;
dayun2 = single(rgb2gray(dayun2)) ;
lab = single(rgb2gray(lab)) ;

% compute the SIFT frames and descriptors
[f1, d1] = vl_sift(dayun1);
[f2, d2] = vl_sift(dayun2);
[f3, d3] = vl_sift(lab);

% visual word histogram
d1 = single(d1);
h1 = feature_histogram(vocab, d1);

d2 = single(d2);
h2 = feature_histogram(vocab, d2);

d3 = single(d3);
h3 = feature_histogram(vocab, d3);

bar(h1);
ylim([0, 0.1]);
bar(h2);
ylim([0, 0.1]);
bar(h3);
ylim([0, 0.1]);
