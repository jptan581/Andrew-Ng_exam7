%% Machine Learning Online Class
%  Exercise 7 | Principle Component Analysis and K-Means Clustering
%     pca.m
%     projectData.m
%     recoverData.m
%     computeCentroids.m
%     findClosestCentroids.m
%     kMeansInitCentroids.m
%% Initialization
clear ; close all; clc
%% ================== Part 1: Load Example Dataset  ===================
fprintf('Visualizing example dataset for PCA.\n\n');
load ('ex7data1.mat');
plot(X(:, 1), X(:, 2), 'bo');
axis([0.5 6.5 2 8]); axis square;
%% =============== Part 2: Principal Component Analysis ===============
fprintf('\nRunning PCA on example dataset.\n\n');
[X_norm, mu, sigma] = featureNormalize(X);
[U, S] = pca(X_norm);
hold on;
drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);
hold off;

fprintf('Top eigenvector: \n');
fprintf(' U(:,1) = %f %f \n', U(1,1), U(2,1));
fprintf('\n(you should expect to see -0.707107 -0.707107)\n');
%% =================== Part 3: Dimension Reduction ===================
fprintf('\nDimension reduction on example dataset.\n\n');

plot(X_norm(:, 1), X_norm(:, 2), 'bo');
axis([-4 3 -4 3]); axis square
K = 1;
Z = projectData(X_norm, U, K);
fprintf('Projection of the first example: %f\n', Z(1));
fprintf('\n(this value should be about 1.481274)\n\n');

X_rec  = recoverData(Z, U, K);
fprintf('Approximation of the first example: %f %f\n', X_rec(1, 1), X_rec(1, 2));
fprintf('\n(this value should be about  -1.047419 -1.047419)\n\n');
hold on;
plot(X_rec(:, 1), X_rec(:, 2), 'ro');
for i = 1:size(X_norm, 1)
    drawLine(X_norm(i,:), X_rec(i,:), '--k', 'LineWidth', 1);
end
hold off
%% =============== Part 4: Loading and Visualizing Face Data =============
fprintf('\nLoading face dataset.\n\n');
load ('ex7faces.mat')
displayData(X(1:100, :));
%% =========== Part 5: PCA on Face Data: Eigenfaces  ===================
fprintf(['\nRunning PCA on face dataset.\n' ...
         '(this mght take a minute or two ...)\n\n']);
[X_norm, mu, sigma] = featureNormalize(X);
[U, S] = pca(X_norm);
displayData(U(:, 1:36)');
%% ============= Part 6: Dimension Reduction for Faces =================
fprintf('\nDimension reduction for face dataset.\n\n');
K = 100;
Z = projectData(X_norm, U, K);
fprintf('The projected data Z has a size of: ')
fprintf('%d ', size(Z));
%% ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
fprintf('\nVisualizing the projected (reduced dimension) faces.\n\n');
K = 100;
X_rec  = recoverData(Z, U, K);
subplot(1, 2, 1);
displayData(X_norm(1:100,:));
title('Original faces');
axis square;
subplot(1, 2, 2);
displayData(X_rec(1:100,:));
title('Recovered faces');
axis square;
%% === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
close all; close all; clc
A = double(imread('bird_small.png'));
A = A / 255;
img_size = size(A);
X = reshape(A, img_size(1) * img_size(2), 3);
K = 16; 
max_iters = 10;
initial_centroids = kMeansInitCentroids(X, K);
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);
sel = floor(rand(1000, 1) * size(X, 1)) + 1;
palette = hsv(K);
colors = palette(idx(sel), :);
figure;
scatter3(X(sel, 1), X(sel, 2), X(sel, 3), 10, colors);
title('Pixel dataset plotted in 3D. Color shows centroid memberships');
%% === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
[X_norm, mu, sigma] = featureNormalize(X);
[U, S] = pca(X_norm);
Z = projectData(X_norm, U, 2);
figure;
plotDataPoints(Z(sel, :), idx(sel), K);
title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');
