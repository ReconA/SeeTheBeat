import sys
import librosa
from draw.canvas_tool import *
from matplotlib import pyplot as plt
import find_images
from draw.background_tool import *
from PIL import Image
import evaluate.evaluator as eva
import scipy.misc

if len(sys.argv) != 4:
    print("Invalid arguments.")
    print("Usage: see_the_beat.py <song> <lyrics.txt> <song_name>")
    exit()

song = sys.argv[1]
lyrics = sys.argv[2]
song_name = sys.argv[3]

y, sr = librosa.load(song)

x_sections = 4
y_sections = 3
section_count = x_sections * y_sections
sections = np.array_split(y, section_count)

# Download images based on lyrics
image_paths = find_images.find_images(lyrics, nb_imgs=section_count)
images = list()
for path in image_paths:
    images.append(np.array(Image.open(path)))

# Initialize canvas.
x_len = 256
y_len = 256
canvas = np.zeros((x_len, y_len, 3), dtype=np.uint8)

# Draw the background image on canvas
draw_background(canvas, song_name)

# Variables for drawing inside sections of canvas.
nx = 0
ny = 0
i = 0

# A set of all pixels used to draw non-background images.
used_pixels = set()

# Draw an image on each section
for section in sections:
    # Get a random point inside a section of the canvas
    x0 = random.randint(math.ceil(nx*x_len/x_sections), math.floor((nx + 1) * x_len/x_sections) - 1)
    y0 = random.randint(math.ceil(ny*y_len/y_sections), math.floor((ny + 1) * y_len/y_sections) - 1)

    # Get the tempo of this section.
    tempo, beat_frames = librosa.beat.beat_track(y=section, sr=sr)
    # Get points inside a canvas to draw to.
    canvas_section = create_canvas_section(canvas, x0, y0, tempo)

    # Draw the image inside the canvas section.
    for y in canvas_section.keys():
        for x in canvas_section.get(y):
            try :
                # If we have fewer images than expected, reuse them from start.
                canvas[y,x] = images[i % len(images)][y,x]
                used_pixels.add((y,x))
            except IndexError:
                pass

    i += 1
    # Move the possible location of the next section.
    nx += 1
    if nx == x_sections:
        nx = 0
        ny += 1


# Evaluate the image
eva.evaluate(canvas, lyrics, used_pixels)

# Save the image.
scipy.misc.imsave('outfile.jpg', canvas)

# Show image
plt.imshow(canvas, interpolation='nearest')
plt.show()


