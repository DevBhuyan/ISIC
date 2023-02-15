function [IoU, F1Score] = evaluateMetrics(YPred, YTrue)
    % YPred: predicted labels
    % YTrue: true labels

    % convert binary masks to logical
    YPred = logical(YPred > 0.5);
    YTrue = logical(YTrue);

    % calculate the number of true positive, false positive, and false negative pixels
    TP = sum(YPred(YTrue));
    FP = sum(YPred(~YTrue));
    FN = sum(~YPred(YTrue));

    % calculate IoU score
    IoU = TP / (TP + FP + FN);

    % calculate F1 score
    Precision = TP / (TP + FP);
    Recall = TP / (TP + FN);
    F1Score = 2 * Precision * Recall / (Precision + Recall);
end