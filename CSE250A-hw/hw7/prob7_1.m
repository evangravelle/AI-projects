% Problem 7.1 script
% Evan Gravelle, Spring 2016
clear;clc;close all

% Calculates state value function
gamma = .75;
I = eye(3);
P = [1/3 2/3 0
     2/3 1/3 0
     0   2/3 1/3];
R = [9 -6 3]';

V = (I - gamma*P)\R
