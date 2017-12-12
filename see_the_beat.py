import sys
import librosa
from draw.canvas_tool import *
from matplotlib import pyplot as plt
import find_images
from PIL import Image

if len(sys.argv) != 3:
    print("Invalid arguments.")
    print("Usage: see_the_beat.py <song.wav> <lyrics.txt>")
    exit()

song = sys.argv[1]
lyrics = sys.argv[2]


#song = librosa.util.example_audio_file()

y, sr = librosa.load(song)

x_sections = 5
y_sections = 3
section_count = x_sections * y_sections
sections = np.array_split(y, section_count)

# download images based on lyrics
image_paths = find_images.find_images(lyrics, nb_imgs=section_count)
images = list()
for path in image_paths:
    images.append(np.array(Image.open(path)))

# Initialize canvas.
x_len = 256
y_len = 256
canvas = np.zeros((x_len, y_len, 3), dtype=np.uint8)

nx = 0
ny = 0
i = 0

for section in sections:

    # Get a random point inside a section of the canvas
    x0 = random.randint(math.ceil(nx*x_len/x_sections), math.floor((nx + 1) * x_len/x_sections) - 1)
    y0 = random.randint(math.ceil(ny*y_len/y_sections), math.floor((ny + 1) * y_len/y_sections) - 1)

    print(x0, y0)
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
            except IndexError:
                pass

    i += 1
    # Move the possible location of the next section.
    nx += 1
    if nx == x_sections:
        nx = 0
        ny += 1


plt.imshow(canvas, interpolation='nearest')
plt.show()
