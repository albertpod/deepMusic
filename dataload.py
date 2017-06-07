"""
We create the dataset : import songs from a folder, turn them into songstruct, and assign them an artist
"""

import os
import midiconnector

MIN_SIZE = 500  # FIXME: to change


class DataLoad:
    def __init__(self):

        self.songs = []  # List[Songs]
        self.artists = []  # bool or 0/1 vector, indicate the artist of the song
        # One artist VS all, or one VS another ?

    def is_empty(self):
        return self.songs == []

    def main(self, artist, path):
        """
        Load all songs from a certain path for a certain artist
        Params:
            artist: bool (0/1)
            path: folder where the files are
        """
        os.chdir(os.curdir + path)

        for file in os.listdir():
            try:
                print(file)
                new_song = midiconnector.MidiConnector.load_file(file)
                assert max(new_song.tracks) > 500
                self.songs.append(new_song)
                self.artists.append(artist)
            except:
                pass

        print("Artist imported")
        os.chdir('..')
        os.chdir('..')
        os.chdir('..')

    def reset(self):
        """
        Resets the lists
        """
        self.songs = []
        self.artists = []
