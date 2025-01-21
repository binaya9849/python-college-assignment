def print_with_commas(sentence):
    for char in sentence:
        if char == " ":
            print(",", end="")
        else:
            print(char, end="")
    print() 

def print_words_on_new_lines(sentence):
    current_word = ""
    for char in sentence:
        if char == " ":
            if current_word:  
                print(current_word)
                current_word = ""
        else:
            current_word += char
    if current_word: 
        print(current_word)

sentence = input("Please enter a sentence: ")
print_with_commas(sentence)
print_words_on_new_lines(sentence)
