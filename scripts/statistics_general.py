import re
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from collections import Counter
import json
import matplotlib.pyplot as plt
from utils import *

def plot_top_non_stopwords_barchart(text):
    stop = set(stopwords.words('english'))

    new = text.split()
    print(new)
    corpus = [word for word in new]

    print(corpus)
    counter = Counter(corpus)
    most = counter.most_common()

    x, y = [], []
    for word, count in most[:40]:
        if (word not in stop):
            x.append(word)
            y.append(count)

    sns.barplot(x=y, y=x)
    plt.show()


def write_to_txt(results):
    filename = "../results/statistics/story_stats.txt"

    with open(filename, 'w') as f:
        f.write(results)

def main():

    corpus_path = "../data/corpus/AesopFables.json"
    with open(corpus_path) as f:
        data = json.load(f)
    stories = data["stories"]

    sentences = []
    words = []
    chars = []
    results = ""

    storiesCount = 0

    for story in stories:

        # Concatenate the stories
        concat_story = concatenate_sentences(story["story"])
        words += concat_story.split()
        chars += [char for i in concat_story for char in i]
        sentences += re.split(r"(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<=\.|\?)", concat_story)
        storiesCount += 1

    avgLetters = len(chars) / len(words)
    results += ("Average letters per word: " + str(avgLetters)) + "\n"

    avgWords = len(words) / len(sentences)
    results += ("Average words per sentence: " + str(avgWords)) + "\n"

    avgSentences = len(sentences) / storiesCount
    results += ("Average sentences per story: " + str(avgSentences)) + "\n"

    results += ("Characters: " + str(len(chars))) + "\n"
    results += ("Words: " + str(len(words))) + "\n"
    results += ("Sentences: " + str(len(sentences))) + "\n"

    write_to_txt(results)


if __name__ == "__main__":
    main()