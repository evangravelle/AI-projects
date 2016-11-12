% Problem 3.6, maximum likelihood estimates
% Evan Gravelle, Spring 2016
clear;clc;close all

load('C:\Users\evan\Desktop\AI\hw3\bigram.mat')
load('C:\Users\evan\Desktop\AI\hw3\unigram.mat')
load('C:\Users\evan\Desktop\AI\hw3\vocab.mat')

% Unigram distribution
unigram_sum = sum(unigram);
unigram_likelihood = zeros(length(unigram));
disp('All words beginning with B with corresponding probabilities:')
for i = 1:length(unigram)
    unigram_likelihood(i) = unigram(i)/unigram_sum;
    current_word = char(vocab(i));
    current_line = '            ';
    if current_word(1) == 'B'
        current_line(1:length(current_word)) = current_word;
        disp([current_line num2str(unigram_likelihood(i))]);
    end
end
disp(' ')

% Bigram distribution, row is second word, column is first
% ONE is the 17th index
bigram_likelihood = zeros(length(unigram));
bigram_count = zeros(length(unigram));
disp('Ten most likely words to follow the word ONE, with corresponding probabilities:')
for i = 1:size(bigram,1)
    bigram_count(bigram(i,2),bigram(i,1)) = bigram(i,3);
end
bigram_sum = sum(bigram_count,1);
for i = 1:length(bigram_sum)
    % If there is no instance of any word following a given word, assume a
    % uniform distribution
    if bigram_sum(i) == 0;
        bigram_likelihood(:,i) = 1/length(unigram)*ones(length(unigram),1);
    else
        bigram_likelihood(:,i) = bigram_count(:,i)/bigram_sum(i);
    end
end
[~,sort_ind] = sort(bigram_likelihood(:,17),'descend');
for i = 1:10
    current_word = char(vocab(sort_ind(i)));
    current_line = '            ';
    current_line(1:length(current_word)) = current_word;
    disp([current_line num2str(bigram_likelihood(sort_ind(i),17))])
end
disp(' ')

disp('Log-likelihoods of the following sentence:')
disp('The stock market fell by one hundred points last week.')
sentence1 = {'THE', 'STOCK', 'MARKET', 'FELL', 'BY', ...
    'ONE', 'HUNDRED', 'POINTS', 'LAST', 'WEEK'};
sentence1_ind = zeros(length(sentence1));
for i = 1:length(sentence1)
    for j = 1:length(unigram)
        if strcmp(char(vocab(j)),char(sentence1(i)))
            sentence1_ind(i) = j;
        end
    end
end
L_u = 0;
for i = 1:length(sentence1)
    L_u = L_u + log(unigram_likelihood(sentence1_ind(i)));
end
disp(['L_u = ' num2str(L_u)])
L_b = 0;
for i = 1:length(sentence1)
    if i == 1
        L_b = L_b + log(bigram_likelihood(sentence1_ind(i),2));
    else
        L_b = L_b + log(bigram_likelihood(sentence1_ind(i),sentence1_ind(i-1)));
    end
end
disp(['L_b = ' num2str(L_b)])
disp('The bigram model yields a higher log-likelihood, which')
disp('we expect because it seems like a familiar combination of words!')
disp(' ')

disp('Log-likelihoods of the following sentence:')
disp('The fourteen officials sold fire insurance.')
sentence2 = {'THE', 'FOURTEEN', 'OFFICIALS', 'SOLD', 'FIRE', 'INSURANCE'};
sentence2_ind = zeros(length(sentence2));
for i = 1:length(sentence2)
    for j = 1:length(unigram)
        if strcmp(char(vocab(j)),char(sentence2(i)))
            sentence2_ind(i) = j;
        end
    end
end
L_u = 0;
for i = 1:length(sentence2)
    L_u = L_u + log(unigram_likelihood(sentence2_ind(i)));
end
disp(['L_u = ' num2str(L_u)])
L_b = 0;
for i = 1:length(sentence2)
    if i == 1
        L_b = L_b + log(bigram_likelihood(sentence2_ind(i),2));
        if L_b == -Inf
            disp(['P(' char(vocab(sentence2_ind(i))) '|' char(vocab(2)) ') = 0'])
        end
    else
        L_b = L_b + log(bigram_likelihood(sentence2_ind(i),sentence2_ind(i-1)));
        if L_b == -Inf
            disp(['P(' char(vocab(sentence2_ind(i))) '|' char(vocab(sentence2_ind(i-1))) ') = 0'])
        end
    end    
end
disp(['L_b = ' num2str(L_b)])
disp(['Any instance of an ordered pair of words that never appears ' ...
   'makes L_b = -Inf.']);
disp(' ')

disp('Mixture model of the following sentence:')
disp('The fourteen officials sold fire insurance.')
lambda = linspace(0,1,100);
L_m = zeros(length(lambda),1);
for h = 1:length(lambda)
    for i = 1:length(sentence2)
        if i == 1
            L_m(h) = L_m(h) + log((1-lambda(h))*unigram_likelihood(sentence2_ind(i)) + ...
                lambda(h)*bigram_likelihood(sentence2_ind(i),2));
            if L_m(h) == -Inf
                disp(['P(' char(vocab(sentence2_ind(i))) '|' char(vocab(2)) ') = 0'])
            end
        else
            L_m(h) = L_m(h) + log((1-lambda(h))*unigram_likelihood(sentence2_ind(i)) + ...
                lambda(h)*bigram_likelihood(sentence2_ind(i),sentence2_ind(i-1)));
            if L_m(h) == -Inf
                disp(['P(' char(vocab(sentence2_ind(i))) '|' char(vocab(sentence2_ind(i-1))) ') = 0'])
            end
        end
    end
end

plot(lambda,L_m)
xlabel('Lambda')
ylabel('L_m')
title('Mixture Model')
[~,lambda_ind] = max(L_m);
disp(['Lambda = ' num2str(lambda(lambda_ind),2) ' is optimal with respect to maximizing L_m.'])
