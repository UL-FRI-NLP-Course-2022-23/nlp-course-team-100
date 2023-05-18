from afinn import Afinn
from transformers import pipeline
import stanza
import numpy as np
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from scipy.special import softmax
import json
import os
from pprint import pprint
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
import itertools

class SentimentPrediction: 

    def __init__(self):
        pass

    def score_sentiment_sentences(self,sentences):
        return np.round(np.mean([self.score_sentiment_sentence(sentence) for sentence in sentences]))
    
    def score_sentiment_sentence(self,sentence):
        pass

class SentimentAfinn(SentimentPrediction): 

    def __init__(self):
        super().__init__()
        self.afn = Afinn(language="en")    

    def score_sentiment_sentence(self, sentence):
        return self.afn.score(sentence)
    
    def __str__(self):
        return "SentimentAfinn"

class SentimentRoBERTa(SentimentPrediction): 

    def __init__(self):
        super().__init__()        
        #self.sentiment_analysis = pipeline("sentiment-analysis",model="bowipawan/bert-sentimental")
        self.sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
        self.label_converter = {
            "positive":1,
            "negative":-1,
            "neutral":0,
        }

    def score_sentiment_sentence(self, sentence):
        inference = self.sentiment_analysis(sentence)        
        if len(inference) > 1:
            raise Exception("More than one sentence was provided")        
        return self.label_converter[inference[0]["label"]]

    def __str__(self):
        return "SentimentRoBERTa"

class SentimentStanza(SentimentPrediction): 

    def __init__(self):
        super().__init__()        
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,sentiment')

    def score_sentiment_sentence(self, sentence):
        #print("sentence: " + str(sentence))
        inference = self.pipeline(sentence)            
        #if len(inference.sentences) > 1:
        #    raise Exception("More than one sentence was provided")        
        stanza_label_converter = {
            0:-1,
            1: 0, 
            2: 1
        }
        return np.mean([stanza_label_converter[i.sentiment] for i in inference.sentences])
        

    def __str__(self):
        return "SentimentStanza"
    

class SentimentSpacy(SentimentPrediction): 

    def __init__(self):
        super().__init__()        
        self.pipeline = spacy.load('en_core_web_sm')
        self.pipeline.add_pipe('spacytextblob')

    def score_sentiment_sentence(self, sentence):
        inference = self.pipeline(sentence)            
        return inference._.blob.polarity     

    def __str__(self):
        return "SentimentSpacy"

class SentimentAnsamble(SentimentPrediction):

    def __init__(self):
        super().__init__()        
        self.models = [
            SentimentRoBERTa(),
            SentimentAfinn(),
            SentimentStanza(),
            SentimentSpacy()
        ]
        self.weights = softmax([1]*len(self.models))
        self.stories = {}

    def score_sentiment_sentence(self, sentence):
        scores = [model.score_sentiment_sentence(sentence) for model in self.models]
        return np.average(scores, weights=self.weights)
    
    def add_story(self, story):
        if not story["number"] in self.stories:
            self.stories[story["number"]] = story

    def add_stories(self, stories):
        for story in stories:
            self.add_story(story)
        
    def eval_weights_based_on_stories(self):
        results = {
            "all":0,
            "correct": {
                model: 0 for model in self.models
            }                        
        }
        for story_number in self.stories:
            story = self.stories[story_number]    
            if not "character_sentences" in story:
                continue
            
            character_sentences = story["character_sentences"]
            for character in story["characters"]:
                if not character in character_sentences:
                    continue
                results["all"] += 1
                scores = {  
                    model: model.score_sentiment_sentences(character_sentences[character]) for model in self.models
                }

                for model in scores:
                    if scores[model] == story["character_sentiment"][character]:
                        results[model] += 1


        self.weights = softmax([results["all"]/results["correct"][model] for model in self.models])

    def __str__(self):
        return "SentimentAnsamble"

def test_the_wolf_and_kid_story():
    story = {
        "number": "01",
        "title": "THE WOLF AND THE KID",
        "story": [
            "There was once a little Kid whose growing horns made him think he was a grown-up Billy Goat and able to take care of himself.",
            "So one evening when the flock started home from the pasture and his mother called, the Kid paid no heed and kept right on nibbling the tender grass.",
            "A little later when he lifted his head, the flock was gone.",
            "He was all alone.",
            "The sun was sinking.",
            "Long shadows came creeping over the ground.",
            "A chilly little wind came creeping with them making scary noises in the grass.",
            "The Kid shivered as he thought of the terrible Wolf.",
            "Then he started wildly over the field, bleating for his mother.",
            "But not half-way, near a clump of trees, there was the Wolf!",
            "The Kid knew there was little hope for him.",
            "Please, Mr. Wolf, he said trembling, I know you are going to eat me.",
            "But first please pipe me a tune, for I want to dance and be merry as long as I can.",
            "The Wolf liked the idea of a little music before eating, so he struck up a merry tune and the Kid leaped and frisked gaily.",
            "Meanwhile, the flock was moving slowly homeward.",
            "In the still evening air the Wolf's piping carried far.",
            "The Shepherd Dogs pricked up their ears.",
            "They recognized the song the Wolf sings before a feast, and in a moment they were racing back to the pasture.",
            "The Wolf's song ended suddenly, and as he ran, with the Dogs at his heels, he called himself a fool for turning piper to please a Kid, when he should have stuck to his butcher's trade."
        ],
        "moral": "Do not let anything turn you from your purpose.",
        "characters": [
            "The Kid", 
            "The Wolf", 
            "The Mother", 
            "The Dogs"
        ],
        "character_sentiment": {
            "The Kid": 0,
            "The Wolf": -1,
            "The Mother": 0,
            "The Dogs": 1
        }
    }

    character_sentences = {
        "The Kid":  [
            "There was once a little Kid whose growing horns made him think he was a grown-up Billy Goat and able to take care of himself.",
            "So one evening when the flock started home from the pasture and his mother called, the Kid paid no heed and kept right on nibbling the tender grass.",
            "A little later when he lifted his head, the flock was gone.",
            "He was all alone.",
            "The Kid shivered as he thought of the terrible Wolf.",
            "Then he started wildly over the field, bleating for his mother.",
            "The Kid knew there was little hope for him.",
            "Please, Mr. Wolf, he said trembling, I know you are going to eat me.",
            "The Wolf liked the idea of a little music before eating, so he struck up a merry tune and the Kid leaped and frisked gaily.",
            "The Wolf's song ended suddenly, and as he ran, with the Dogs at his heels, he called himself a fool for turning piper to please a Kid, when he should have stuck to his butcher's trade."

        ],
        "The Wolf":  [
            "The Kid shivered as he thought of the terrible Wolf.",
            "Please, Mr. Wolf, he said trembling, I know you are going to eat me.",
            "But first please pipe me a tune, for I want to dance and be merry as long as I can.",
            "The Wolf liked the idea of a little music before eating, so he struck up a merry tune and the Kid leaped and frisked gaily.",
            "In the still evening air the Wolf's piping carried far.",
            "They recognized the song the Wolf sings before a feast, and in a moment they were racing back to the pasture.",
            "The Wolf's song ended suddenly, and as he ran, with the Dogs at his heels, he called himself a fool for turning piper to please a Kid, when he should have stuck to his butcher's trade."

        ],
        "The Mother":  [
            "So one evening when the flock started home from the pasture and his mother called, the Kid paid no heed and kept right on nibbling the tender grass."
        ],
        "The Dogs": [
            "The Shepherd Dogs pricked up their ears.",
            "They recognized the song the Wolf sings before a feast, and in a moment they were racing back to the pasture.",
            "The Wolf's song ended suddenly, and as he ran, with the Dogs at his heels, he called himself a fool for turning piper to please a Kid, when he should have stuck to his butcher's trade."
        ]
    }

    models = [
        SentimentRoBERTa(),
        SentimentAfinn(),
        SentimentStanza(),
        SentimentSpacy(),
        SentimentAnsamble(),        
    ]

    for character in character_sentences:
        scores = {
            model: model.score_sentiment_sentences(character_sentences[character]) for model in models
        }   
        print("")
        print("character: " + str(character))
        for model in models:
            print(str(model) + ", score: " + str(scores[model])  + ", truth: " + str(story["character_sentiment"][character]))

def load_story_coref(story, path_to_coref):
    
    file = "{}.json".format(story['title'].replace(' ', '_'))
    path_to_file = os.path.join(path_to_coref, file)
    with open(path_to_file) as json_file:
        story_coref = json.load(json_file)
    return story_coref

def load_story_ner(story, path_to_ner):
    
    file = "{}.json".format(story['title'].replace(' ', '_'))
    path_to_file = os.path.join(path_to_ner, file)
    with open(path_to_file) as json_file:
        story_ner = json.load(json_file)
    return story_ner


def load_data(path_to_annotations, path_to_coref, path_to_ner):

    with open(path_to_annotations) as json_file:
        data = json.load(json_file)
    
    for idx, _ in tqdm(enumerate(data['stories'])):
        data['stories'][idx]['story_coref'] = load_story_coref(data['stories'][idx], path_to_coref)
        data['stories'][idx]['story_ner'] = load_story_ner(data['stories'][idx], path_to_ner)
        data['stories'][idx]['character_sentences'] = extract_characters_sentences(data['stories'][idx])
        
    return data


def extract_character_sentences(story, character):
    sentences = story['story_coref']['coref'].split(".")
    return list(filter(lambda sentence: character.title() in sentence, sentences))

def extract_characters_sentences(story):
    return {
        character: extract_character_sentences(story, character)
        for character in story['story_ner']['characters']
    }
    
def predict_sentiment_on_data(data):
    models = [
        #SentimentRoBERTa(),
        SentimentAfinn(),
        SentimentStanza(),
        SentimentSpacy(),
        #SentimentAnsamble(),        
    ]
    for idx, _ in enumerate(data['stories']):
        #if idx > 10:
        #    break
        data['stories'][idx]['scores'] = {}
        character_sentences = data['stories'][idx]['character_sentences']
        for character in character_sentences:
            
            #print("")
            #character_truth = 'The {}'.format(character.title())            
            #if not character_truth in data['stories'][idx]["character_sentiment"]:
            #    continue
            #print("character: " + str(character) + ", character_truth: " + str(character_truth))
            #pprint(character_sentences[character])
            scores = {
                str(model): model.score_sentiment_sentences(character_sentences[character]) for model in models
            }   
            data['stories'][idx]['scores'][character] = scores
            #for model in models:
            #    print(str(model) + ", score: " + str(scores[model])  + ", truth: " + str(data['stories'][idx]["character_sentiment"][character_truth]))
        
    return data

def get_sentiment_stats(data):
    stats = {
        "character_sentiment_stats":{
            "positive":0,
            "negative":0,
            "neutral":0,
        },
        "animals_stats":{}
    }

    sentiment_to_label_converter = {
        1:"positive",
        -1:"negative",
        0:"neutral"
    }

    for idx, story in enumerate(data['stories']):                                        
        for character in story["character_sentiment"]:
            label = sentiment_to_label_converter[story["character_sentiment"][character]]
            stats["character_sentiment_stats"][label] += 1                        
            if not character in stats["animals_stats"]:
                stats["animals_stats"][character] = {
                    "positive":0,
                    "negative":0,
                    "neutral":0,
                    "appeared":0
                }

            stats["animals_stats"][character][label] += 1
            stats["animals_stats"][character]["appeared"] += 1

    n = 5
    least_apperances = 5
    keys = list(stats["animals_stats"].keys())
    print("Number of all characters: {}".format(len(keys)))    
    keys = [key for key in keys if stats["animals_stats"][key]["appeared"] > least_apperances]
    
    print("{} of characters appeared at least {} times".format(len(keys), least_apperances))    

    keys.sort(key=lambda key: stats["animals_stats"][key]["positive"]/stats["animals_stats"][key]["appeared"])
    positive_sorted = deepcopy(keys)
    
    keys.sort(key=lambda key: stats["animals_stats"][key]["negative"]/stats["animals_stats"][key]["appeared"])
    negative_sorted = deepcopy(keys)

    keys.sort(key=lambda key: stats["animals_stats"][key]["neutral"]/stats["animals_stats"][key]["appeared"])
    neutral_sorted = deepcopy(keys)

    keys.sort(key=lambda key: stats["animals_stats"][key]["appeared"])
    appeared_sorted = deepcopy(keys)

    top_n_characters = {
        "most":{
            "positive": list(reversed(positive_sorted[-n:])),
            "negative": list(reversed(negative_sorted[-n:])),
            "neutral": list(reversed(neutral_sorted[-n:])),
            "appeared": list(reversed(appeared_sorted[-n:]))
        },
        "least":{
            "positive": positive_sorted[:n],
            "negative": negative_sorted[:n],
            "neutral": neutral_sorted[:n],
            "appeared": appeared_sorted[:n]            
        }
        
    } 

    plt.figure(figsize=(14,6))
    
    sentiment = list(stats["character_sentiment_stats"].keys())
    occurences = list(stats["character_sentiment_stats"].values())    
    plt.bar(sentiment,occurences, color = ["g","r","b"])
    plt.title("Character sentiment class distribution")
    plt.savefig("character_sentiment_stats.png")
    plt.clf()
    #for _type  in ["most", "least"]:
    for _type  in ["most", "least"]:
        data = [
            [stats["animals_stats"][character]["positive"]/stats["animals_stats"][character]["appeared"] for character in top_n_characters[_type]["positive"]],
            [stats["animals_stats"][character]["negative"]/stats["animals_stats"][character]["appeared"] for character in top_n_characters[_type]["negative"]],
            [stats["animals_stats"][character]["neutral"] /stats["animals_stats"][character]["appeared"] for character in top_n_characters[_type]["neutral"]]
        ]    
        X = np.arange(n)        
        plt.barh(X + 0.00, data[0], color = 'g', height = 0.25, label="positive")
        plt.barh(X + 0.25, data[1], color = 'r', height = 0.25, label="negative")
        plt.barh(X + 0.50, data[2], color = 'b', height = 0.25, label="neutral")
        labels = [
            [
                "{} ({})".format(top_n_characters[_type]["positive"][i], stats["animals_stats"][top_n_characters[_type]["positive"][i]]["appeared"]), 
                "{} ({})".format(top_n_characters[_type]["negative"][i], stats["animals_stats"][top_n_characters[_type]["positive"][i]]["appeared"]), 
                "{} ({})".format(top_n_characters[_type]["neutral"][i], stats["animals_stats"][top_n_characters[_type]["positive"][i]]["appeared"]) 
            ] for i in range(0,n)
        ]
        labels = list(itertools.chain.from_iterable(labels))
        locs = [[i + 0.0, i+0.25, i+0.50] for i in range(0,n)]
        locs = list(itertools.chain.from_iterable(locs))

        plt.yticks(locs, labels)  # Set text labels.
        plt.legend()
        plt.title("Sentiment class distribution with respect to characters (top {} per class)".format(n))
        #plt.show()
        plt.savefig("top_n_characters_{}.png".format(_type))
        plt.clf()


    

def predict_sentiment(data):
    data = predict_sentiment_on_data(data)
    store_path = '../results/sentiment/sentiment_prediction.json'
    if os.path.exists(store_path):
        os.remove(store_path)
    
    with open(store_path, 'w') as f:
        f.write(json.dumps(data,indent=4))

def main():
    data = load_data(
        path_to_annotations='../data/annotations/AesopFablesCharacterSentiment.json',
        path_to_coref='../results/spacy/coref',
        path_to_ner='../results/spacy/ner',
    )
    get_sentiment_stats(data)
    #predict_sentiment(data)
    

if __name__ == "__main__":
    
    main()
    
    
    
    #test_the_wolf_and_kid_story()