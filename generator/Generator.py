import random
import numpy as np
import generator.Reader as reader

def generateLyrics(corpus, maxlen, vocabular_size, char_ind, ind_char, model):
    generated = ""
    start_index = random.randint(0, len(corpus) - maxlen - 1)
    sent = corpus[start_index:start_index + maxlen]
    generated += sent
    print(">>Generating lyrics<<")
    for i in range(1900):
        x_sample = generated[i : i + maxlen]
        x = np.zeros((1, maxlen, vocabular_size))
        for j in range(maxlen):
            x[0, j, char_ind[x_sample[j]]] = 1

        probs = model.predict(x)
        probs = np.reshape(probs, probs.shape[1])
        ix = np.random.choice(range(vocabular_size), p=probs.ravel())
        generated += ind_char[ix]

    print(">>Lyrics generated<<")
    reader.saveResult(generated)