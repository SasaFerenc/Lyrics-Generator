import generator.Reader as reader
import generator.LSTM as lstm
import generator.Generator as gen
import numpy as np

def main():
    lyrics = reader.readFile()

    print(">>Creating corpus<<")
    corpus = ''
    for i in range(len(lyrics)):
        corpus += lyrics[i]
    print(">>Corpus created<<")

    print(">>Creating char to index and index to char<<")
    vocabular = list(set(corpus))
    char_ind = {c:i for i, c in enumerate(vocabular)}
    ind_char = {i:c for i, c in enumerate(vocabular)}
    print(">>Char to index and index to char created<<")

    maxlen = 40
    vocabular_size = len(vocabular)
    sentences = []
    next_char = []

    print(">>Creating sentences<<")
    for i in range(len(corpus) - maxlen - 1):
        sentences.append(corpus[i:i+maxlen])
        next_char.append(corpus[i+maxlen])
    print(">>Sentences created<<")

    print(str(len(sentences)) + " = " + str(maxlen) + " = " + str(vocabular_size) )
    x = np.zeros((len(sentences), maxlen, vocabular_size))
    y = np.zeros((len(sentences), vocabular_size))

    print(">>Creating 0-1 matrix<<")
    for i in range(len(sentences)):
        print("Matrix: " + str(i) + " / " + str(len(sentences)))
        y[i, char_ind[next_char[i]]] = 1
        for j in range(maxlen):
            x[i, j, char_ind[sentences[i][j]]] = 1
    print(">>Matrix created<<")

    model = lstm.createLSTM(x, y, maxlen, vocabular_size)
    reader.saveModel(model)
    gen.generateLyrics(corpus, maxlen, vocabular_size, char_ind, ind_char, model)

if __name__ == "__main__":
    main()