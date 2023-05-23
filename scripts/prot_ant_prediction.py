import json
from utils import *
import stanza
from collections import Counter

nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

def stanza_ner_prediction(stories):
    count_protagonists=0
    count_antagonists=0
    for story in stories:
        concat_story = concatenate_sentences(story["story"])
        filename = "../results/prot_ant_prediction/"+story["title"].replace(" ","_")+".json"
        ner = nlp(concat_story)
        named_entities = [i.text for i in ner.ents if i.type =='PERSON']
        named_entities = [str(i).lower().replace("'s","") for i in named_entities]
        named_entities = remove_articles(named_entities)
        unique_entities = get_unique_entities(named_entities)
        sentiments = story['scores']['All sentences']
        #print(named_entities)
        #print(unique_entities)
        #print(sentiments)
        counter = Counter(named_entities).most_common()
        occurrences_dict = dict(counter)
        protagonist = ""
        antagonist = ""
        prot_found = False
        for key, value in occurrences_dict.items():
            if key in sentiments and sentiments[key]['CNN'] >= 0.0:
                protagonist = key
                count_protagonists+=1
                prot_found=True
            if not prot_found and key in sentiments and sentiments[key]['CNN'] < 0.0:
                antagonist = key
                count_antagonists+=1
        data = {"character_counts": occurrences_dict , "protagonist": protagonist, "antagonist": antagonist}
        with open(filename, 'w') as f:
            json.dump(data, f)
        #print(f'Protagonist: {protagonist}')
        #print(f'Antagonist: {antagonist}')
    print(f'Number of protagonists detected: {count_protagonists} \n Number of antagonists detected: {count_antagonists}')



        


def main():
    corpus_path = "../results/sentiment/sentiment_prediction.json"
    with open(corpus_path) as f:
        data = json.load(f)
    stories = data["stories"]
    stanza_ner_prediction(stories)
        

if __name__ == "__main__":
    main()