from datasets import load_from_disk




if __name__ == "__main__":
    ds = load_from_disk('././dataset/')
    x = ds['train']['targets'][0][0]
    persian_alphabet_map = {
        'ا': 1, 'ب': 2, 'پ': 3, 'ت': 4, 'ث': 5, 'ج': 6, 'چ': 7, 'ح': 8, 'خ': 9,
        'د': 10, 'ذ': 11, 'ر': 12, 'ز': 13, 'ژ': 14, 'س': 15, 'ش': 16, 'ص': 17, 
        'ض': 18, 'ط': 19, 'ظ': 20, 'ع': 21, 'غ': 22, 'ف': 23, 'ق': 24, 'ک': 25, 
        'گ': 26, 'ل': 27, 'م': 28, 'ن': 29, 'و': 30, 'ه': 31, 'ی': 32, ' ': 33
    }
    numerical_embedding = []
    for char in x:
        # i += 1
        if char in persian_alphabet_map.keys():
            numerical_embedding.append(persian_alphabet_map[char])
        else:
            numerical_embedding.append(0)    
        # print(char, '\n')
    # print(x)    
    print(numerical_embedding)    