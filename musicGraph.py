import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

nb_out = 3
max_arity = 2  # maximum number of input for a particular node
node_types = ["UNARY_MIN", "UNARY_PLUS", "COS", 'SIN', "SUM", "MULT"]  # does NOT contain output
inputs = ["X","Y","Z","bar","beat"]

class MusicGraph(nx.DiGraph):
    dict_functions = {"SUM": lambda x, y: np.array(x) + np.array(y),
                      "MULT": lambda x, y: np.array(x)*np.array(y),
                      "SIN": lambda x, y: np.sin(([(x[k]+y[k])/2 for k in range(len(x))])),
                      "COS": lambda x, y: np.cos(([(x[k]+y[k])/2 for k in range(len(x))])),
                      "UNARY_MIN": lambda x: -np.array(x),
                      "UNARY_PLUS": lambda x: np.array(x) + 1}

    f_node = [{"name": fname, "binary": False if fname.find('UNARY') != -1 else True} for fname in dict_functions]

    def __init__(self, inputs, outputs=None, internal_nodes_n=10, connect=True):
        """ Initialize the inputs and outputs

        Params:
            input: dictionary type input: vector
            output: list of output names
            internal_nodes_n: amount of internal nodes
        """
        nx.DiGraph.__init__(self)

        # color map for possible plotting
        self.color_map = []
        # internal functional nodes
        self.internals = []

        # private
        self._inputs = inputs
        if outputs is not None:self._outputs = outputs
        else:self._outputs= []
        self._nodes_priority = []

        self.add_nodes_from(inputs)
        if outputs is not None: self.add_nodes_from(outputs)
        for i in range(internal_nodes_n): self.add_internal_node()

        nx.set_node_attributes(self, 'values', inputs)
        if connect:
            self.connect_random()

    def connect_random(self):
        """ Connect all the nodes with respect to the structure of MusicGraph:
         Output nodes have two inputs from functional nodes (f_node) or input nodes
         Binary nodes repeats the policy of output nodes
         Ordinary nodes have one input either from functional node or input_node
         """

        self.__set_priority()
        for index in range(len(self._inputs), len(self._nodes_priority)):
            # no output node can be an input for another output node
            if str(self._nodes_priority[index]).find("output") != -1:
                in_pair = random.sample(self._nodes_priority[:len(self._nodes_priority) - len(self._outputs)], 2)
            elif self.node[self._nodes_priority[index]]["binary"]:
                in_pair = random.sample(self._nodes_priority[:index], 2)
            else:
                in_node = random.choice(self._nodes_priority[:index])
                self.add_path([in_node, self._nodes_priority[index]])
                self.__compute_nodes(self._nodes_priority[index], in_node)
                continue
            self.add_path([in_pair[0], self._nodes_priority[index]])
            self.add_path([in_pair[1], self._nodes_priority[index]])
            self.__compute_nodes(self._nodes_priority[index], in_pair[0], in_pair[1])

    def connect(self, node, in1, in2):
        """
        Connects a node to its parents in the graph. Does not compute the operation
        """
        if self.node[node]["name"].find("output") or self.node[node]["name"].find("binary"):
            self.add_path(in1, self.node[node])
            self.add_path(in2, self.node[node])
        else:
            self.add_path(in1, self.node[node])

    def __set_priority(self):
        """ Create a stack list of all nodes. """
        self._nodes_priority = list(self._inputs.keys()) + self.internals + list(self._outputs)

    def __paint(self):
        """ Paint all the nodes """
        for node in self:
            if node in self._outputs:
                self.color_map.append('y')
            elif node in self._inputs:
                self.color_map.append('r')
            else:
                self.color_map.append('b')

    def __compute_nodes(self, calc_node, input1, input2=None):
        """ Compute the result of each node

            Params:
                calc_node: node to compute
                input1: first input
                input2: second input for binary nodes
        """
        if str(calc_node).find("output") != -1:
            n, vel = output([self.node[input1]["values"], self.node[input2]["values"]])
            self.node[calc_node]["values"] = np.vstack((n, vel))
        elif input2 is not None:
            self.node[calc_node]["values"] = self.dict_functions[self.node[calc_node]["name"]] \
                (self.node[input1]["values"], self.node[input2]["values"])
        else:
            self.node[calc_node]["values"] = self.dict_functions[self.node[calc_node]["name"]] \
                (self.node[input1]["values"])

    def add_internal_node(self, function=None):
        if function is None:
            r = random.randint(0,len(self.f_node)-1)
            self.add_node(len(self.internals), self.f_node[:][r].copy())
        else:
            self.add_node(len(self.internals), function)
        self.internals.append(len(self.internals))

    def check_consistency(self):
        for node in self.node:
            if self.in_degree(node) > 2:
                print("Consistency is impaired")
                return False

    def array_to_graph(self, genes):
        """
        Turns array into graphs.
        :param gene: divided in chunks of size max_arity + 1
        :return: MusicGraph object
        """
        # Creates a graph with only input and output
        genes_per_node = max_arity + 1
        nb_nodes = int(len(genes) / genes_per_node)

        compt = 1
        for node_id in range(nb_nodes):
            gene_id = node_id * genes_per_node
            gene = genes[gene_id]
            if node_id >= nb_nodes - nb_out:
                type = "output"
                self.add_node(gene, {}.copy())
                self._outputs.append(gene)
                compt += 1
            else:
                type = node_types[gene % len(node_types)]
                dic = {"name": type, "binary": False if type.find('UNARY') != -1 else True}.copy()
                self.add_node(node_id, dic)
                self.internals.append(node_id)
            # Then we connect the node

            eligible_node = min(self.number_of_nodes() - 5, nb_nodes - nb_out)
            if type.find("UNARY") != -1:
                arity = 1
            else:
                arity = 2
            for i in range(arity):
                gene_id = node_id * genes_per_node + i + 1  # now we look at the connected nodes
                gene_in = genes[gene_id]
                if gene_in in inputs:
                    input_id = gene_in
                else:
                    input_id = gene_in % eligible_node
                if node_id >= nb_nodes - nb_out:
                    self.add_path([input_id, gene])
                else:
                    self.add_path([input_id, node_id])
                if input_id == node_id:
                    print(node_id)

        #nx.topological_sort(self)
        return self.node, self.edge

    def to_array(self):
        array = []
        new_node_id = {}
        compt = 0
        for node in [node for node in nx.topological_sort(self) if isinstance(node, int)]:
            if node in inputs:
                continue
            else:
                r = [node_types.index(self.node[node]["name"])]
                pred = self.predecessors(node)
                new_node_id[node] = compt
                compt += 1
                for k in range(len(pred)):
                    if pred[k] in new_node_id.keys():
                        pred[k] = new_node_id[pred[k]]
                r += pred
                if len(r) == 2:
                    r += [0]
                array += r
        i = 0
        for node in self._outputs:
            i += 1
            r = ['output%s' % i]
            r += self.predecessors(node)
            array += r
        return array

    def __gt__(self, other):
        return False

    def pos_nodes(self):
        """ Assign a position for each node for further plotting """
        INPUTS_Y = 500
        OUTPUTS_Y = 0
        BIAS = 100

        for in_node in self._inputs:
            self.node[in_node]["pos"] = (BIAS * list(self._inputs.keys()).index(in_node), INPUTS_Y)
        for internal in self.internals:
            r = random.randint(OUTPUTS_Y+50, INPUTS_Y-50)
            self.node[internal]["pos"] = (random.randint(10 * self.internals.index(internal), 20 * self.internals.index(internal))
                                          , r)
        for out_node in self._outputs:
            self.node[out_node]["pos"] = (BIAS * self._outputs.index(out_node), OUTPUTS_Y)

        return nx.get_node_attributes(self, 'pos')

    def plot(self):
        self.__paint()
        NODE_SIZE = 500
        nx.draw_networkx(self, node_color=self.color_map, with_labels=True, pos=self.pos_nodes(), node_size=NODE_SIZE)
        plt.axis("off")
        plt.show()


def midiMap(n):
    octave = n / 7
    chroma = n % 7
    retval = octave * 12
    if chroma == 0: return retval + 0
    elif chroma == 1: return retval + 2
    elif chroma == 2: return retval + 3
    elif chroma == 3: return retval + 5
    elif chroma == 4: return retval + 7
    elif chroma == 5: return retval + 8
    elif chroma == 6: return retval + 10


def output(args):
    note, velocity = [], []
    activity = 0
    for k in range(len(args[0])):
        activityIn = abs(args[0][k])
        activity *= 0.25
        activity += activityIn
        threshold = 1.5
        value = args[1][k]

        if float(activity) < 0.05 * threshold:
            velocity.append(0)
            try:note.append(note[-1])
            except:note.append(0)
        elif float(activity) < threshold:
            velocity.append(-1)
            try:note.append(note[-1])
            except:note.append(0)
        else:
            vel = int(50.0 + 50.0 * np.log(1.0 + activity - threshold))
            activity *= 0.05
            if vel > 127:
                velocity.append(127)
            else:
                velocity.append(vel)

            note.append(int(midiMap(int(18 + 32 * 0.5 * (1.0 + np.tanh(value))))))

    return note, velocity

G = MusicGraph(inputs={"X": [0, 1, 1], "Y": [0, 1, 1], "Z": [0, 1, 1], "beat": [0, 1, 1], "bar": [0, 1, 1]},
               outputs=["output1", "output2", "output3"],
               internal_nodes_n=20, connect=True)

array = G.to_array()

G2 = MusicGraph(inputs={"X": [0, 1, 1], "Y": [0, 1, 1], "Z": [0, 1, 1], "beat": [0, 1, 1], "bar": [0, 1, 1]},
                # outputs=["output1", "output2", "output3"],
                internal_nodes_n=0, connect=False)

G2.array_to_graph(array)

array2 = G2.to_array()
