# General Imports
import json
from collections import Counter

# Specific Imports
from spacy_coref import *
from utils import *
import spacy


nlp_spacy = spacy.load('en_core_web_sm')


def spacy_ner(story):

    # Perform ner
    ner = nlp_spacy(story)

    # Find only named entities
    named_entity = [i.text for i in ner.ents if i.label_ in ['PERSON']]

    # Convert all names to lowercase and remove 's in names
    named_entity = [str(i).lower().replace("'s", "") for i in named_entity]

    # Remove article words
    named_entity = remove_articles(named_entity)

    # Get only unique entities
    named_entity = get_unique_entities(named_entity)

    return named_entity


def name_entity_recognition(story, use_cor_res=False):

    if use_cor_res:
        story_out = spacy_coref(story)

    characters = spacy_ner(story_out)

    counts = Counter(characters)
    characters = [i for i in counts]
    counts = [counts[i] for i in counts]

    return characters, counts, story


def write_to_json(results, title, use_cor_res=False):
    if use_cor_res:
        filename = "../results/spacy/ner_coref/" + title.replace(" ", "_") + ".json"
        write_characters_to_json(results[0], filename)
    else:
        filename = "../results/spacy/ner/" + title.replace(" ", "_") + ".json"
        write_characters_to_json(results[0], filename)


def main():
    corpus_path = "../data/corpus/AesopFables.json"

    with open(corpus_path) as f:
        data = json.load(f)
    stories = data["stories"]

    for story in stories:
        # Concatenate the stories
        concat_story = concatenate_sentences(story["story"])

        results = name_entity_recognition(concat_story, True)
        write_to_json(results, story['title'], True)


if __name__ == "__main__":
    main()