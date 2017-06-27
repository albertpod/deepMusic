"""
We create the dataset : import songs from a folder, turn them into songstruct, and assign them an artist
"""

import os
import mido
import midiconnector

MIN_SIZE = 500  # FIXME: to change


class DataLoad:
    def __init__(self):

        self.songs = []  # List[(msg, absolute time]
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
                new_midi = mido.MidiFile(file)
                new_song = []
                tmp = []
                tempo = 500000  # default, 120BPM
                ticks_per_beat = new_midi.ticks_per_beat
                last_msg_tick = 0
                msec = 0
                abs_tick = 0

                for msg in new_midi:
                    if msg.type == "set_tempo":
                        tempo = msg.tempo
                    if (not msg.is_meta) and (msg.type != "sysex") and (len(msg) == 3):
                        m = int(msg.hex(''),16)   # turns the message into an integer
                        abs_tick += msg.time
                        delta_ticks = abs_tick - last_msg_tick
                        last_msg_tick = abs_tick
                        delta_msec = tempo * delta_ticks/ticks_per_beat
                        msec += delta_msec
                        new_song.append(float(str(m) + '.' + str(int(msec))))
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
