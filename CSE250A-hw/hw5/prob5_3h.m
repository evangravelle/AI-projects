% Prob 5.3h
% Evan Gravelle, Spring 2016
clear; clc; close all

g = @(x) 0;
for k = 1:10
    g = @(x) g(x) + 0.1*log(cosh(x+1/sqrt(k^2+1)));
end

ezplot(g)
title('5.3h')
