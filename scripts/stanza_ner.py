import json
from utils import *
import stanza
from allennlp.predictors.predictor import Predictor

nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
def stanza_ner(stories, coref=False):
    for story in stories:
        concat_story = concatenate_sentences(story["story"])
        if coref:
            concat_story=allen_coref(concat_story)
            filename = "../results/stanza/ner_coref/"+story["title"].replace(" ","_")+".json"
        else:
            filename = "../results/stanza/ner/"+story["title"].replace(" ","_")+".json"
        ner = nlp(concat_story)
        named_entities = [i.text for i in ner.ents if i.type =='PERSON']
        named_entities = [str(i).lower().replace("'s","") for i in named_entities]
        named_entities = remove_articles(named_entities)
        named_entities = get_unique_entities(named_entities)
        write_characters_to_json(named_entities,filename)

def allen_coref(concat_story):
    model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
    predictor = Predictor.from_path(model_url)
    prediction = predictor.predict(document=concat_story)
    res = predictor.coref_resolved(concat_story)
    return res

def main():
    corpus_path = "../data/corpus/AesopFables.json"
    with open(corpus_path) as f:
        data = json.load(f)
    stories = data["stories"]
    stanza_ner(stories,True)
        


if __name__ == "__main__":
    main()