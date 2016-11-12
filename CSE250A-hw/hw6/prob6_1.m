% Problem 6.1 - Viterbi Algorithm
% Evan Gravelle, Spring 2016
clear; clc; close all

load('emissionMatrix.mat')
load('initialStateDistribution.mat')
load('observations.mat')
load('transitionMatrix.mat')

% Initialize
transitionMatrix(:,27) = [];
A = transitionMatrix;
B = emissionMatrix;
O = observations;
T = length(O);
n = length(A);
max_P = zeros(1,T);
P = zeros(n,T);
P0 = initialStateDistribution;

tree = zeros(n,T);
path = zeros(1,T);

% Calculate the first probability vector
P(:,1) = (A*P0).*B(:,O(1)+1)/sum((A*P0).*B(:,O(1)+1));

% Using probabilities, construct a matrix containing indices of connected
% node
for t = 2:T
    for j = 1:n % j is the index of the 2nd column
        best_val = 0;
        for i = 1:n % i is the index of the 1st column
            new = P(i,t-1)*A(i,j);
            if new > best_val
                best_val = new;
                tree(j,t) = i;
                P(j,t) = new*B(j,O(t)+1);
            end
        end
    end
    P(:,t) = P(:,t)/sum(P(:,t));
end

% Construct path backwards
[~,path(T)] = max(P(:,T));
for t = T-1:-1:1
    path(t) = tree(path(t+1),t+1);
end
plot(path)
title('State transitions')
xlabel('t')
ylabel('State')

% print out message converted into a string
% 97 is a on ASCII table
str = char(path(1)+96);
for t = 2:T
    if path(t) ~= path(t-1)
        str = [str char(path(t)+96)];
    end
end
disp(['The hidden message is: ' str])


