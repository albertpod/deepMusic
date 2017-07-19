import musicGeneration
import jSymbolic
import mgGP
import fitness as f
import musicGraph as mg
import os

pop_size = 100

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

for _ in range(10):
    # Chromosomes
    population = mgGP.evolve(chromosomes, f_pop, generations=1)
    chromosomes = []
    init = []
    for i in range(len(population)):
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
    for k in range(pop_size):
        out = musicGeneration.runFromData(music_graph=init[k], cmpt=k)
        if out:
            chromosomes.append(init[k].to_array())
    features = jSymbolic.get_features("midifiles")
    f_pop = []
    for feat in features:
        f_pop.append(f.fitness([feat]))
    print("\n", max(f_pop))

print("Evolution over")