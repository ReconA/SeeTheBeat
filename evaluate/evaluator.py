import evaluate.senti_classifier.senti_classifier as classifier
import nltk
import math
from PIL import Image, ImageStat


def brightness(canvas):
    """
    Calculate average brightness of given image.
    From https://stackoverflow.com/questions/3490727/what-are-some-methods-to-analyze-image-brightness-using-python
    :param im_file: Image to analyze.
    :return: Average brightness of image as a float.
    """
    im = Image.fromarray(canvas, 'RGB')
    stat = ImageStat.Stat(im)
    r,g,b = stat.rms
    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))


def get_pos_neg(lyrics):
    """
    Use sentinet classifier to get positive and negative scores of lyrics.
    :param lyrics: Lyrics to analyze.
    :return: pos_score: Float. The estimated positivity of lyrics.
             neg_score: Float. The estimated negativity of lyrics
    """
    nltk.download('wordnet')
    pos_score, neg_score = classifier.polarity_scores(lyrics)
    return pos_score, neg_score


def evaluate(canvas, lyrics):
    """
    Print an evaluation based on sentiment analysis of lyrics 
    :param canvas:
    :param lyrics:
    :return:
    """
    pos_score, neg_score = get_pos_neg(open(lyrics))
    b = brightness(canvas)

    print(pos_score, neg_score)
    print(b)