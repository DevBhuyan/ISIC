base_dir = fullfile('D:/Curious Dev B/PROJECT STAGE - II/ISIC 2017 for segmentation (python)/');
train_dir = fullfile(base_dir, 'Train');
test_dir = fullfile(base_dir, 'Test');

train_img_dir = fullfile(train_dir, 'IMAGES');
train_px_dir = fullfile(train_dir, 'MASKS');
test_img_dir = fullfile(test_dir, 'IMAGES');
test_px_dir = fullfile(test_dir, 'MASKS');

training_images = imageDatastore(train_img_dir);
testing_images = imageDatastore(test_img_dir);

classNames = ["nonLesion", "lesion"];
pxlabels = [0 1];

training_masks = pixelLabelDatastore(train_px_dir, classNames, pxlabels);
testing_masks = pixelLabelDatastore(test_px_dir, classNames, pxlabels);

training_data = combine(training_images, training_masks);
testing_data = combine(testing_images, testing_masks);

pool = parpool('local');

imageSize = [224 224 3];
numClasses = 2;
layers = unetLayers(imageSize, numClasses, 'EncoderDepth', 3);

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 2, ...
    'MiniBatchSize', 16, ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency', 5, ...
    'ExecutionEnvironment','parallel');

trainedUNet = trainNetwork(training_data, layers, options);

predictions = predict(trainedUNet, testing_data);
    
delete(pool);