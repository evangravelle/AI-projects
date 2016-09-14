# Some string practice


def avoids(_word, forbidden):
    for letter in _word:
        if letter in forbidden:
            return False
    return True


def is_abecedarian(_word):
    old_letter = _word[0]
    for letter in _word[1:]:
        if letter <= old_letter:
            return False
        old_letter = letter
    return True

fin = open('words.txt')
count = 0.
total = 0.
for line in fin:
    word = line.strip()
    # if avoids(word, 'aeoiu'):
    if is_abecedarian(word):
        print word
        count += 1
    total += 1

print count / total
