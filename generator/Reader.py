import pandas as pd
import math as math

lyricsPath_csv = r'E:\Projects\PyCharm\NeuralNetwork\Project\resources\lyrics.csv'
model_weights_path = r'E:\Projects\PyCharm\NeuralNetwork\Project\resources\model_weights.h5'
model_json_path = r'E:\Projects\PyCharm\NeuralNetwork\Project\resources\model_json.json'
lyricsPath_txt = r'E:\Projects\PyCharm\NeuralNetwork\Project\resources\lyrics_cleaned.txt'
result_128 = r'E:\Projects\PyCharm\NeuralNetwork\Project\results\result128.txt'

def readFile():
    print(">>Reading dataset<<")
    file = pd.read_csv(lyricsPath_csv)['lyrics']

    lyrics = list()
    for i in file.values:
        if not (isinstance(i, float) and math.isnan(i)):
            lyrics.append(i)

    print(">>Reading finished<<")
    return lyrics[:250]


def saveModel(model):
    print(">>SAVING MODEL<<")
    model_json = model.to_json()
    with open(model_json_path, "w") as file:
        file.write(model_json)

    model.save_weights(model_weights_path)
    print(">>MODEL SAVED<<")


def saveResult(result):
    print(">>SAVING RESULT<<")

    with open(result_128, "w") as file:
        file.write(result)

    print(">>RESULT SAVED<<")