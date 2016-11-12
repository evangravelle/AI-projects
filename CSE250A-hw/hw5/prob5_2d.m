% 5.2d
% Evan Gravelle, Spring 2016
clear; clc; close all

load('X.mat')
load('Y.mat')

X = hw5X1;
Y = hw5Y;
T = size(X,1);
n = size(X,2);
num_iter = 256;

p = zeros(num_iter+2,n);
M = zeros(num_iter+1,1);
L = zeros(num_iter+1,1);
Py0x = zeros(T,1);
Py1x = zeros(T,1);

p(1,:) = 2/n*ones(1,n);
count = sum(X,1);
disp('  it       M           L')
for it = 1:num_iter+1
    
    L(it) = 0;
    for t = 1:T
        
        Py0x(t) = 1;
        for i = 1:n
            Py0x(t) = Py0x(t)*(1-p(it,i))^X(t,i);
        end
        Py1x(t) = 1 - Py0x(t);
        
        for i = 1:n
            p(it+1,i) = p(it+1,i) + (1/count(i))*Y(t)*X(t,i)*p(it,i)/Py1x(t);
        end
        
        if ((Py1x(t) >= 0.5 && Y(t) == 0) || (Py1x(t) <= 0.5 && Y(t) == 1))
            M(it) = M(it) + 1;
        end
        
        if Y(t) == 1
            L(it) = L(it) + (1/T)*log(Py1x(t));
        else
            L(it) = L(it) + (1/T)*log(Py0x(t));
        end
    end
    
    if ((floor(log2(it-1)) == log2(it-1)) || (it-1 == 0))
        disp(sprintf('%4d%8d%12.4f',it-1,M(it),L(it)))
    end
end
