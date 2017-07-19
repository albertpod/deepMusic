import csv
import os
import matplotlib.pyplot as pl
import mido
import musicGraph
import networkx

nb_out = musicGraph.nb_out

f = [file for file in os.listdir("midifiles") if file.endswith(".mid")]
for file in f:
    os.remove("midifiles/"+file)


def run(times, curves):
    """
    Each node of the graph is run in a certain order. When a node of type "output" is run, the values are stored.
    :param times: list of 2 floats : "bar" and "beat"
    :param curves: list of 3 floats : "x", "y" and "z"
    :return: matrix 2*nb_out, contains note and velocity for each nb_out
    """
    retval = [[0 for k in range(2)] for k in range(nb_out)]
    compt = 0

    """Pseudo-code:
    for i in orderedNodes:
        mn = getMusicNode(i)
        pred = getPredecessors(i)
        mn.run(predecessors, times, curves)
        
        if mn.getFunction() == "output":
            retval[0][compt] = mn.note
            retval[1][compt] = mn.velocity
            compt +=1
    """

    return retval


def parse_xyz(filename):
    """
    Parses txyz file and turns it into list
    :param filename: name of the file
    :return: out: list of 5 lists of length data.length : "bar", "beat", "x", "y" and "z"
    """
    out = [[],[],[],[],[]]
    with open(filename, "r") as f :
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            for k in range(5):
                out[k].append(float(row[k]))
    return out

data = parse_xyz('schema/AABA_3_4.txyz')

t = data[:][0]  # bar
b = data[:][1]  # beat
x = data[:][2]
y = data[:][3]
z = data[:][4]

def runFromData(data=data, music_graph=None, cmpt=0):
    """
    Generates a MIDI file given time and data array
    :param music_graph: graph that will be run
    :param filename: txyz file. Turned into list of 5 lists of length data.length : "bar", "beat", "x", "y" and "z"
    :return: null (or a MIDI format object ?)
    """
    mid = mido.MidiFile()
    tpb = 3
    # results is of shape 2*len(data)*nb_out, which means there are nb_out tracks, which contains len(data) steps
    # which contains 2 information : the note and its velocity
    results = [[[0 for i in range(len(data))] for k in range(2)] for j in range(nb_out)]
    if music_graph is None:


        curves = [x, y, z]
        times = [t, b]
        mg = musicGraph.MusicGraph({"X": x, "Y": y, "Z": z, "beat": b, "bar": t},
                                   outputs=["output1", "output2", "output3"], internal_nodes_n=100)
        mg.connect_random()
        # mg.plot()
        # result = run(times, curves)
    else:
        mg = music_graph

    for node in mg._outputs:
        for k in range(2):
            r = mg.node[node]["values"][k]
            results[mg._outputs.index(node)][k] = r
    # mg.plot()
    instruments = [0, 27, 32]
    # Once results is computed, it is turned into MIDI file
    for i in range(nb_out):
        track = mido.MidiTrack()
        track.append(mido.Message("program_change", program=instruments[i], time=0, channel=i))
        cur_note = results[i][0][0]
        cur_vel = results[i][1][0]
        quantum = int(mido.second2tick(1./48., tpb, 120))  # time unit, in ticks
        cur_dur = 0  # duration of a note
        abs_time = 0
        # state machine : 0 means a note is playing, 1 means no note is playing
        state = 1
        for j in range(len(x)+1):

            tmp = []
            if j == len(x):
                input_note = 1
                input_vel = 1   # puts a note_off at the end
                state = 1
            else:
                input_note = results[i][0][j]
                input_vel = results[i][1][j]

            if state == 0:
                if input_vel < 0:  # keep on playing the note
                    cur_dur += quantum
                elif input_vel == 0:  # note_off, end the note. For now, we put absolute time in each note
                    track.append(mido.Message("note_on", note=cur_note, velocity=cur_vel, channel=i, time=abs_time))
                    tmp.append((cur_note, cur_vel, i, cur_dur))
                    cur_dur = quantum  # resets duration of note
                    state = 1
                elif input_note > 0 and input_vel > 0:  # note_on, add another note
                    if False and input_note == cur_note:
                        cur_dur += quantum  # but we could chose to create a new note instead
                        # what about random decision ? for now, false
                    else:
                        track.append(mido.Message("note_on", note=cur_note, velocity=cur_vel, channel=i, time=abs_time))
                        tmp.append((cur_note, cur_vel, i, cur_dur))
                        cur_note = input_note
                        cur_vel = input_vel
                        cur_dur = quantum  # resets

            elif state == 1:
                if input_vel < 0:
                    cur_dur += quantum
                elif input_vel == 0:
                    cur_dur += quantum
                elif input_note > 0 and input_vel > 0:
                    while len(tmp) > 0:
                        n, v, c, dur = tmp.pop()
                        track.append(mido.Message("note_off", note=n, velocity=v, channel=c, time=abs_time + dur))
                    cur_note = input_note
                    cur_vel = input_vel
                    abs_time += cur_dur  # adds a rest
                    cur_dur = quantum  # resets
                    state = 0

            abs_time += quantum

        # We turn absolute time into relative time
        track.sort(key=lambda s: s.time)
        last_time = 0
        new_track = mido.MidiTrack()

        for message in track:
            message.time -= last_time
            last_time += message.time
            new_track.append(message)

        mid.tracks.append(new_track)
    if len(mid.tracks[0]) + len(mid.tracks[1]) + len(mid.tracks[2]) != 3:
        mid.save("midifiles/test%s.mid" % cmpt)
        return True
    else:
        return False
