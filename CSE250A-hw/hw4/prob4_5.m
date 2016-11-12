% Problem 4.5
% Evan Gravelle, Spring 2016
clear;clc;close all

load('C:\Users\evan\Desktop\AI\hw4\nasdaq00.mat')
load('C:\Users\evan\Desktop\AI\hw4\nasdaq01.mat')
x = nasdaq00;
y = nasdaq01;
% Construct the gradient and Hessian

H = zeros(4,4);
for i = 1:4
    for j = 1:4
        for t = 5:length(x)
            H(i,j) = H(i,j) - x(t-i)*x(t-j);
        end
    end
end

% Newton's Method (not strictly necesary, given a quadratic cost)
num_iter = 10;
a = zeros(4,num_iter+1);
L = zeros(num_iter+1,1);
p1 = log(1/sqrt(2*pi));
for k = 1:num_iter
    grad = zeros(1,4);
    for i = 1:4
        for t = 5:length(x)
            grad(i) = grad(i) + (x(t) - a(1,k)*x(t-1) - ...
              a(2,k)*x(t-2) - a(3,k)*x(t-3) - a(4,k)*x(t-4))*x(t-i);
        end
    end
    
    for t = 5:length(x)
        L(k) = L(k) + p1 - 0.5*(x(t) - a(1,k)*x(t-1) - ...
          a(2,k)*x(t-2) - a(3,k)*x(t-3) - a(4,k)*x(t-4))^2;
    end
    
    a(:,k+1) = a(:,k) - H\(grad');
end

for t = 5:length(x)
    L(num_iter+1) = L(num_iter+1) + p1 - 0.5*(x(t) - a(1,num_iter+1)*x(t-1) - ...
      a(2,num_iter+1)*x(t-2) - a(3,num_iter+1)*x(t-3) - a(4,num_iter+1)*x(t-4))^2;
end

a_final = a(:,end);
disp('The coefficients which maximize L are:')
disp(['a1 = ' num2str(a_final(1))])
disp(['a2 = ' num2str(a_final(2))])
disp(['a3 = ' num2str(a_final(3))])
disp(['a4 = ' num2str(a_final(4))])
disp(' ')

figure(1)
hold on
plot(a(1,:),'--')
plot(a(2,:),'-')
plot(a(3,:),'--')
plot(a(4,:),'-')
title('Coefficients using Newtons Method')
xlabel('Iteration')
ylabel('Value')

% Performance evaluation
xy = [x;y];
pred = zeros(size(xy));
pred(1:4) = xy(1:4);
for t = 5:length(xy)
    pred(t) = a_final(1)*pred(t-1) + a_final(2)*pred(t-2) + ...
      a_final(3)*pred(t-3) + a_final(4)*pred(t-4);
end
MSE_00 = 1/length(x)*sum((x-pred(1:length(x))).^2);
MSE_01 = 1/length(y)*sum((y-pred(length(x)+1:end)).^2);
disp('The mean squared error for 2000 is: ')
disp(MSE_00)
disp('The mean squared error for 2001 is: ')
disp(MSE_01)

figure(2)
hold on
plot(xy)
plot(pred,'--')

disp('I would recommend this model for short term stock market prediction, ')
disp('because the MSE actually decreases in 2001')

