from utils import *
import re

def write_story_to_json(story, title):

    filename = "../data/concat_stories/" + title.replace(" ", "_") + ".json"
    write_concat_story_to_json(story, filename)


def main():
    corpus_path = "../data/corpus/AesopFables.json"

    with open(corpus_path) as f:
        data = json.load(f)
    stories = data["stories"]

    for story in stories:
        # Concatenate the stories
        concat_story = concatenate_sentences(story["story"])

        sentences = (re.split(r"(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<=\.|\?)", concat_story))

        sentencesOut = sentences[0]

        write_story_to_json(sentencesOut, story['title'])


if __name__ == "__main__":
    main()