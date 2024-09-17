import numpy as np




def softmax(attention):
    return (np.exp(attention) / np.sum(np.exp([attention]), axis=-1).T)




if __name__ == "__main__":
    attention = np.random.randn(4, 4)
    print("--attention befor masking--")
    print(attention)
    mask = np.tri(4, 4)
    mask[mask == 0] = -np.inf
    mask[mask == 1] = 0
    print("--attention mask--")
    print(mask)
    attention = attention + mask
    print("--attention after masking--")
    print(attention)
    attention = softmax(attention)
    print("--attention after applying softmax--")
    print(attention)