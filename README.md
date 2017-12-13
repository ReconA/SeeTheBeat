# SeeTheBeat

Visualize song lyrics and beat. 
Course project for Computational Creativity at CS department of University of Helsinki.  



## Running environnemt
- python 3.x
- install ``librosa`` (https://librosa.github.io/librosa/install.html)
- install ``ntkl``
- install ``opencv-python``

## How to use the project

run

	python3 see_the_beat.py <path_audio_file> <path_lyrics_file> <song_name>
	- path_audio_file can be .wav or .mp3 .
	- path_lyrics_file the is a text file that contain the lyrics of the song. 
	- song_name is the name of the song as a string. 

## How does it work
see_the_beat will load the song and the lyrics. based on the lenght of the song, it select the number of images that the programm will try to download. Based on the lyrics, the programm create a markov chain that is used to generate short setences of 3 words. Those setences must contain at least a noun and a proposition. 

After having generate some setence, the system will used the bing search engine from Microsoft to look online for picture that should match the query (or the setence). 

The system then download a random picture from the set of picture returned by bing. 

On the otherhand, the system analyse the audio audio and start to draw shape on the 256x256 canvas image. Those shape will be used to insert part of the downloaded images. 

 
