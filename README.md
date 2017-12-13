# SeeTheBeat

Visualize song lyrics and beat. 
Course project for Computational Creativity at CS department of University of Helsinki.  



## Running environnemt
- install python 3.5
- ``pip install -r requirements.txt`` (inside the project)
- install ``librosa`` (follow instruction here : https://librosa.github.io/librosa/install.html)
- install opencv with ``pip3 install opencv-python`` 

## How to use the project

run

	python3 see_the_beat.py <path_audio_file> <path_lyrics_file> <song_name>
	- path_audio_file : path of audio file; can be .wav or .mp3 .
	- path_lyrics_file :  path of a text file that contain the lyrics of the song. 
	- song_name : is the name of the song as a string. 

example

	python3 see_the_beat.py data/songs/astronomy_domine.mp3 data/lyrics/astronomy_domine.txt astronomy\ domine


## Project description

The cover of a pink Floyd album, A Saucerful of Secrets from 1968, inspires this project. This artwork is made of multiple drawing, put together without any particular order. The result is abstract but very interesting. When we begin the project, we were wondering if a computer could by program to do the same kind of work. 

Our objective with this project is to use a computer to generate image that will look like the original cover from pink Floyd, but any kind of song. To do so, we are using lyrics and audio from the song to create the new cover. 

see_the_beat load the song and the lyrics. Based on the length of the song, it selects the number of images that the program will try to download. Based on the lyrics, the program creates a markov chain that will be used to generate short sentences of 3 or less words. Those sentences must contain at least a noun or a proposition. 

After having generated some sentence, the system uses Bing search engine from Microsoft to look online for pictures that should match the query (or the sentences generated by the markov chain). 

The system then download a random picture from the set of picture returned by Bing. 

On the other hand, the system analyzes the audio and starts to draw shape on the 256x256 canvas image. Those shapes will be used to insert part of the downloaded images.

 ## Modules
 Sentiment analysis classifier in evaluate/senti_classifier is directly taken from https://github.com/kevincobain2000/sentiment_classifier.
 
