function C = customreader(filename)
    img = imread(filename);
    C = zeros(size(img));
    for i = 1:size(img, 1)
        for j = 1:size(img, 2)
            if img(i, j) == 0
                C(i, j) = "nonLesion";
            else
                C(i, j) = "lesion";
            end
        end
    end
end