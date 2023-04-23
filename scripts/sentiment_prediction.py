from afinn import Afinn
from transformers import pipeline
import stanza
import numpy as np
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from scipy.special import softmax

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

class SentimentBert(SentimentPrediction): 

    def __init__(self):
        super().__init__()        
        self.sentiment_analysis = pipeline("sentiment-analysis",model="bowipawan/bert-sentimental")
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
        return "SentimentBert"

class SentimentStanza(SentimentPrediction): 

    def __init__(self):
        super().__init__()        
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,sentiment')

    def score_sentiment_sentence(self, sentence):
        inference = self.pipeline(sentence)            
        if len(inference.sentences) > 1:
            raise Exception("More than one sentence was provided")        
        return inference.sentences[0].sentiment

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
            SentimentBert(),
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
        SentimentBert(),
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

if __name__ == "__main__":
    test_the_wolf_and_kid_story()