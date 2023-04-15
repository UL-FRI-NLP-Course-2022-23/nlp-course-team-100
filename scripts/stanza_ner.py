import json
from utils import *
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
def stanza_ner(stories):
    i=0
    for story in stories:
        if i==1:
            break
        concat_story = concatenate_sentences(story["story"])
        ner = nlp(concat_story)
        named_entities = [i.text for i in ner.ents if i.type =='PERSON']
        named_entities = [str(i).lower().replace("'s","") for i in named_entities]
        named_entities = remove_articles(named_entities)
        named_entities = get_unique_entities(named_entities)
        print(named_entities)
        i+=1


def main():
    corpus_path = "../data/corpus/AesopFables.json"
    with open(corpus_path) as f:
        data = json.load(f)
    stories = data["stories"]
    stanza_ner(stories)
        


if __name__ == "__main__":
    main()