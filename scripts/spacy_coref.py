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


def write_to_json(results, title):

    filename = "../results/spacy/coref/" + title.replace(" ", "_") + ".json"
    write_coref_to_json(results, filename)


def main():
    corpus_path = "../data/corpus/AesopFables.json"
    with open(corpus_path) as f:
        data = json.load(f)
    stories = data["stories"]
    for story in stories:
        # Concatenate the stories
        concat_story = concatenate_sentences(story["story"])

        results = spacy_coref(concat_story)

        write_to_json(results, story['title'])


if __name__ == "__main__":
    main()