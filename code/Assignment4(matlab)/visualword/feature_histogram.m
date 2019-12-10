function h = feature_histogram(codebook, features)

    h = zeros(1, size(codebook, 2));

    for i = 1 : size(features, 2)
        min = 1e8;
        ind = 1;
        for j = 1 : size(codebook, 2)
            feature = features(:, i);
            word = codebook(:, j);
            s = sum((word - feature).^2);
            if s < min
                ind = j;
                min = s;
            end
        end
        h(ind) = h(ind) + 1;
    end
    
    h = h / sum(h);
   
end