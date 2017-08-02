"""
Main file, where initial population is generated, then evolution is performed and the best of generation are saved.
"""
import musicGeneration
import jSymbolic
import mgGP
import fitness as f
import musicGraph as mg
import os
import joblib as jl
from shutil import copyfile

pop_size = 100
best_kept = 5

files = [file for file in os.listdir("midifiles") if file.endswith(".mid")]
for file in files:
    os.remove("midifiles/" + file)

init = mgGP.create_population(pop_size)
chromosomes = []
print("Initial population generating...")

for k in range(pop_size):
    out = musicGeneration.runFromData(music_graph=init[k], cmpt=k)
    if out:
        chromosomes.append(init[k].to_array())

print("Initial population generated !")
print("Extracting features...")

features = jSymbolic.get_features("midifiles")

print("Features extracted !")
print("Calculating fitness...")

f_pop = f.fitness(features)

print("Fitness calculated !")
print("Evolution start")
best_of_gen = [ANCESTORS for (F_POP, ANCESTORS) in sorted(zip(f_pop, chromosomes), reverse=True)][:best_kept]

for j in range(100):
    # Chromosomes
    population = mgGP.evolve(chromosomes, f_pop, generations=1)
    chromosomes = []  # stores the chromosomes
    init = []  # stores the graph

    for i in range(len(population)):  # Generate empty graphs, then fill them using arrays
        tmp = mg.MusicGraph(inputs={"X": mgGP.x, "Y": mgGP.y, "Z": mgGP.z, "beat": mgGP.beat, "bar": mgGP.bar},
                # outputs=["output1", "output2", "output3"],
                internal_nodes_n=0, connect=False)
        tmp.array_to_graph(population[i])
        init.append(tmp)
    # Now the graphs are done, we generate midi files
    # First, we empty the folder that contains the old midi files
    files = [file for file in os.listdir("midifiles") if file.endswith(".mid")]
    for file in files:
        os.remove("midifiles/" + file)
    z = []
    # This part generates the midi files
    for k in range(pop_size):
        out = musicGeneration.runFromData(music_graph=init[k], cmpt=k)
        if out:
            chromosomes.append(init[k].to_array())
            z.append(k)
    # We use jSymbolic to extract features from these files
    features = jSymbolic.get_features("midifiles")
    # Fitness is calculated
    f_pop = f.fitness(features)
    #f_pop = [init[k].variety() for k in range(len(init))]
    # The 5 best individuals are stored and re-generated later
    best_of_gen += [ANCESTORS for (F_POP, ANCESTORS) in sorted(zip(f_pop, chromosomes),  key=lambda x: x[0], reverse=True)][:best_kept]
    # The best individual is stored separately
    copyfile("midifiles/test%s.mid" % z[f_pop.index(max(f_pop))], "best/gen%s_test%s_fit%.3f.mid" % (j, z[f_pop.index(max(f_pop))], max(f_pop)))
    # If fitness is high enough, not only the midi file is stored, but also its list representation
    if max(f_pop) > 0.6:
        jl.dump(max(f_pop), "gen%s_fit%.3f.pkl" % (j, max(f_pop)))
    print("\nGeneration :", j)
    print("Max fitness :", max(f_pop))

print("Evolution over")
print("Saving best midi files...")

i = 0
# Re-generate 5 best songs
for song in best_of_gen:
    tmp = mg.MusicGraph(inputs={"X": mgGP.x, "Y": mgGP.y, "Z": mgGP.z, "beat": mgGP.beat, "bar": mgGP.bar},
                        # outputs=["output1", "output2", "output3"],
                        internal_nodes_n=0, connect=False)
    tmp.array_to_graph(song)
    musicGeneration.runFromData(music_graph=tmp, cmpt="gen%s_%s" % (int(i/5), i))
    i += 1
