import json
from utils import *


def precision(TP, FP):
    if TP==0 and FP==0:
        return 1.0
    return TP/(TP+FP)

def recall(TP, FN):
    if TP==0 and FN==0:
        return 1.0
    return TP/(TP+FN)

def F1Score(precision, recall):
    if precision==0 and recall==0:
        return 0.0
    return 2*precision*recall/(precision+recall)

def calculate_metrics(characters, res_characters):
    TP = 0
    FP = 0
    FN = 0
    for char in res_characters:
        foundMatch = False
        for anot_char in characters:
            if char in anot_char or anot_char in char:
                TP+=1
                characters.remove(anot_char)
                foundMatch = True
                break
        if not foundMatch:
            FN+=1
    FP = len(characters)
    return TP, FP, FN


def main():
    annotations_path = "../data/annotations/AesopFablesCharacterSentiment.json"
    #Uncomment path variable based on what you want to test
    #Stanza Ner
    #res_path = "../results/stanza/ner/"

    #Spacy Ner
    #res_path = "../results/spacy/ner/"

    #Stanza Ner+Coref
    res_path = "../results/stanza/ner_coref/"

    #Spacy Ner+Coref
    #res_path = "../results/spacy/ner_coref/"

    with open(annotations_path) as f:
        data = json.load(f)
    stories = data["stories"]

    TP=0
    FP=0
    FN=0
    #Open every story in calculate metrics based on annotations
    for story in stories:
        characters = story['characters']
        with open(res_path+story['title'].replace(" ","_")+".json") as f:
            res_characters = json.load(f)
            res_characters = res_characters['characters']
        
        characters = [str(i).lower().replace("'s","") for i in characters]
        characters = remove_articles(characters)
        tp, fp, fn = calculate_metrics(characters, res_characters)
        TP+=tp
        FP+=fp
        FN+=fn
    #Calculates scores based on global TP, FP and FN scores.
    prec = precision(TP,FP)
    rec = recall(TP,FN)
    print(f'All true positives: {TP}')
    print(f'All false positives: {FP}')
    print(f'All false negatives: {FN}')
    print(f'Precision: {prec}')
    print(f'Recall: {rec}')
    print(f'F1 Score: {F1Score(prec,rec)}')
        


if __name__ == "__main__":
    main()