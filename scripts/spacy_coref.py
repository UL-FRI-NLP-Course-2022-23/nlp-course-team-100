# General Imports
import json

# Specific Imports
from utils import *
import spacy
import neuralcoref


nlp_spacy = spacy.load('en_core_web_sm')  # load the model
neuralcoref.add_to_pipe(nlp_spacy)


def spacy_coref(story):

    # Perform coref
    cor = nlp_spacy(story)

    coref_result = cor._.coref_resolved  # You can see cluster of similar mentions

    return coref_result


def main():
    corpus_path = "../data/corpus/AesopFables.json"
    with open(corpus_path) as f:
        data = json.load(f)
    stories = data["stories"]
    for story in stories:
        results = spacy_coref(story)
        filename = "../results/spacy/coref/" + story['title'].replace(" ", "_")+".json"
        write_characters_to_json(results[0], filename)


if __name__ == "__main__":
    main()