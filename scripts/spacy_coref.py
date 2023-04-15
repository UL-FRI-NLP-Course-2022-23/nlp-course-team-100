import spacy
import neuralcoref


nlp_spacy = spacy.load('en_core_web_sm')  # load the model
neuralcoref.add_to_pipe(nlp_spacy)


def spacy_coref(story):

    # Concatenate the stories
    concat_story = concatenate_sentences(story["story"])

    # Perform ner
    cor = nlp_spacy(concat_story)

    print(doc._.coref_clusters)

    coref_result = doc._.coref_resolved  # You can see cluster of similar mentions

    return coref_result


def main():
    corpus_path = "../data/corpus/AesopFables.json"
    with open(corpus_path) as f:
        data = json.load(f)
    stories = data["stories"]
    for story in stories:
        results = spacy_coref(story)
        print(results)
        #filename = "../results/spacy/ner/" + results[2]['title'].replace(" ", "_")+".json"
        #print(filename)
        #write_characters_to_json(results[0], filename)


if __name__ == "__main__":
    main()