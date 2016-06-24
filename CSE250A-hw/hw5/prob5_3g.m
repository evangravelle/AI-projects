% Prob 5.3g
clear; clc; close all

x = linspace(.01,2,100);
y = exp(2*x) - exp(-2*x) - 8*x;

plot(x,y)