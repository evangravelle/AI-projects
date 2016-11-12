% Prob % Prob 5.3f
% Evan Gravelle, Spring 2016
clear; clc; close all

num_iter = 10;
x1 = zeros(1,num_iter);
x2 = zeros(1,num_iter);
x1(1) = 2;
x2(1) = -3;
for i = 1:num_iter
    x1(i+1) = x1(i) - (exp(x1(i))-exp(-x1(i)))/(exp(x1(i))+exp(-x1(i)));
    x2(i+1) = x2(i) - (exp(x2(i))-exp(-x2(i)))/(exp(x2(i))+exp(-x2(i)));
end

hold on
plot(x1)
plot(x2)
title('5.3f')
