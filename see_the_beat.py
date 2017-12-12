import sys
import librosa
from draw.canvas_tool import *
from PIL import Image

if len(sys.argv) != 2:
    print("Invalid arguments.")
    print("Usage: see_the_beat.py <song.wav> <lyrics.txt>")
    #exit()

#song = sys.argv[0]
#lyrics = sys.argv[1]

filename = librosa.util.example_audio_file()

y, sr = librosa.load(filename)

x_sections = 3
y_sections = 2
sections = x_sections * y_sections
sections = np.array_split(y, sections)


# Initialize canvas.
x_len = 1024
y_len = 1024
canvas = np.zeros((x_len, y_len, 3), dtype=np.uint8)

images = None # TODO: an array of images from lyrics.

image = Image.open("draw/forest.jpg")
pix = np.array(image)

nx = 0
ny = 0
i = 0
for section in sections:

    # Get a random point inside a section of the canvas
    x0 = random.randint(math.ceil(nx*x_len/x_sections), math.floor((nx + 1) * x_len/x_sections))
    y0 = random.randint(math.ceil(ny*y_len/y_sections), math.floor((ny + 1) * y_len/y_sections))

    # Get the tempo of this section.
    tempo, beat_frames = librosa.beat.beat_track(y=section, sr=sr)

    # Get points inside a canvas to draw to.
    canvas_section = get_points_inside(canvas, x0, y0, tempo)

    # Draw the image inside the canvas section.
    for y in canvas_section.keys():
        for x in canvas_section.get(y):
            canvas[y,x] = images[i][y,x]

    i += 1
    # Move the possible location of the next section.
    nx += 1
    if nx == x_sections:
        nx = 0
        ny += 1


from matplotlib import pyplot as plt
plt.imshow(canvas, interpolation='nearest')
plt.show()
