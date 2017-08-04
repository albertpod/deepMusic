Summary
-------
Deep Music is a tentative of evolutionary music system, based on the work of James McDermott and Una-May O'Reilly : XG
(Music Graph). Basically, the graph representation and the evolutionary part (genetic evolution) is their work, and was
originally written in Java, but we implemented it Python. You can find their GitHub here : https://github.com/jmmcd/XG .
The main difference with their work is the fitness function used : instead of relying on human feedback as they did, we
made a classifier using Artificial Neural Network (ANN), which classifies song using low-level features extracted via
jSymbolic2. The ANN is implemented with Keras.

How it works
------------
The first thing you have to do is gather a database of MIDI files, and to classify them using genre / artist. Then,
features have to be extracted using jSymbolic2, then fed into `dense_features.py`. This will produce a model, that you
can then use in order to classify midi files.
Then, `generateXYZ.py` has to be run once to generate schema (cf the paper of McDermott, "An Executable Graph Representation for
Evolutionary Generative Music" for more detail on the way schema works).
Then you can start the evolution : simply run `main.py`, it will produce an initial population and evolve it, creating
midi files you can listen to.

Arrays <=> Graph
----------------
MusicGraph objects are explained in the paper or McDermott. You can visualize them using the function `plot`. Basically,
a slightly different version of Cartesian GP (CGP) is used to represent the graphs as arrays.
Since the maximum arity is two, our array is divided in chunk of three integers that each represent a node. The first
number indicates the function of the node, the second and third indicate which ones are the parents.

How to improve it
-----------------
The main problem with our system is that the fitness function does not guarantee that the output midi files are getting
"better", at best, it tells us when it "sounds like" some particular genre or artist. So, in order to improve our system,
one should seek a better fitness function.
Also, try playing with generateXYZ, and remember that you can use one particular array with different input for the tree,
which could lead to interesting result (for instance, if you find a good array, you may want to check if it stays "good"
with different input).