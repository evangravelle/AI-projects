% Hangman
% Evan Gravelle, Spring 2016
clear;clc;close all

load('word.mat');
load('count.mat');

[count_sorted,index_sorted] = sort(count);
num_words = length(count);
grand_total = sum(count);
words_sorted = textdata(index_sorted);
words_to_print = 10;

% Sanity check, prints 5 most frequent 5 letter words and 5 least frequent
disp('Least common words:');
for i = 1:words_to_print
    disp(char(words_sorted(i)));
    % disp(count_sorted(i)/total);
end
disp(' ');
disp('Most common words:');
for i = words_to_print:-1:1
    disp(char(words_sorted(end-words_to_print+i)));
    %disp(count_sorted(end-words_to_print+i)/total);
end

% Guessed letters, unknown is @
correct = '@U@@@';
incorrect = 'AEIOS';

% disp(' ');
% disp('Words with counts that satisfy the constraints: ');
char_count = zeros(1,26);
total = 0;
for word_ind = 1:num_words
    current_word = char(words_sorted(word_ind));
    
    % Does the current word satisfy all conditions?
    feasible = 1;
    for letter = 1:5
        % current_word(letter)
        % [correct incorrect]
        if ((current_word(letter) ~= correct(letter) && ...
          correct(letter) ~= '@') || ...
          (ismember(current_word(letter),[incorrect correct]) && ...
          correct(letter) == '@'))
            feasible = 0;
            break;
        end
    end
    
    % If current word is feasible, add count for each new letter and for
    % total
    new_letters = [];
    if feasible
        % disp([current_word char(9) int2str(count_sorted(word_ind))])
        total = total + count_sorted(word_ind);
        for letter = 1:5
            if correct(letter) == char(64) && ~ismember(current_word(letter),new_letters)
                new_letters = [new_letters current_word(letter)];
            end
        end
    end

    char_to_add = double(new_letters) - 64;
    char_count(char_to_add) = char_count(char_to_add) + count_sorted(word_ind);
    
end

[best_count,best_guess] = max(char_count);
% best_count
% total
% disp(' ');
% disp(['Best guess is ' char(best_guess + 64) ' with probability']);
% disp(best_count/total)
