import bing_api
import numpy as np
from PIL import Image


def draw_background(canvas, song_name):
    """
    Make a image search based on song name (which should be lyrics file name), and draw a background
    from the image.
    :param canvas: Canvas on which to draw the background.
    :param song_name: Name of the song.
    :return: Nothing. Canvas is modified directly.
    """

    # Do a search with the lyrics file name to get a background image.
    background_name = "background"
    bing_api.get_image(song_name, background_name)
    background = np.array(Image.open(background_name))

    x, y = calculate_start_points(canvas, background)

    for j in range(len(canvas[0])):
        for i in range(len(canvas)):
            try:
                canvas[j,i] = background[j+y, i+x]
            except IndexError:
                pass


def calculate_start_points(canvas, background_image):
    """
    Calculate where do start drawing background on canvas to get the center.
    :param canvas: Canvas on which to draw.
    :param background_image: Image from which to draw the background.
    :return: x, y coordinates of the start point.
    """

    b_y, b_x = background_image.shape[:2]
    c_y, c_x = canvas.shape[:2]

    x = int(b_x/2 - c_x/2)
    y = int(b_y/2 - c_y/2)

    if x < 0:
        x = 0
    if y < 0:
        y = 0

    return x,y
