% Prob 5.3c
clear; clc; close all

f = @(x) log(cosh(x));
df = @(x) (exp(x)-exp(-x))/(exp(x)+exp(-x));
Q1 = @(x) f(2) + df(2)*(x-2) + 0.5*(x-2)^2;
Q2 = @(x) f(-3) + df(-3)*(x+3) + 0.5*(x+3)^2;

hold on
ezplot(f)
% ezplot(df)
ezplot(Q1)
ezplot(Q2)
title('5.3c')