from eva import EvaProgram, Input, Output, evaluate
from eva.ckks import CKKSCompiler
from eva.seal import generate_keys
from eva.metric import valuation_mse
import timeit
import networkx as nx
import random
import itertools
from functools import lru_cache

# Generates a random undirecred graph
# @param(n): node count
def generateGraph(n):
    #ws = nx.cycle_graph(n)
    # ws = nx.watts_strogatz_graph(n,k,p)
    random_edge = random.randint(0, n*(n-1))
    ws = nx.gnm_random_graph(n=n, m=random_edge, directed=False)
    return ws

# If there is an edge between two vertices its weight is 1 otherwise it is zero
# You can change the weight assignment as required
# Two dimensional adjacency matrix is represented as a vector
# Assume there are n vertices
# (i,j)th element of the adjacency matrix corresponds to (i*n + j)th element in the vector representations
def serializeGraphZeroOne(GG,vec_size, node_count):
    # n = GG.size()
    n = node_count
    graphdict = {}
    g = []
    for row in range(n):
        for column in range(n):
            if GG.has_edge(row, column) or row==column: # I assumed the vertices are connected to themselves
                weight = 1
            else:
                weight = 0 
            g.append( weight  )  
            key = str(row)+'-'+str(column)
            graphdict[key] = [weight] # EVA requires str:listoffloat
    # EVA vector size has to be large, if the vector representation of the graph is smaller, fill the eva vector with zeros
    for i in range(vec_size - n*n): 
        g.append(0.0)
    return g, graphdict

# To display the generated graph
def printGraph(graph,n):
    for row in range(n):
        for column in range(n):
            print("{:.5f}".format(graph[row*n+column]), end = '\t')
        print() 

# Eva requires special input, this function prepares the eva input
# Eva will then encrypt them
def prepareInput(n, m):
    input = {}
    GG = generateGraph(n)
    truth = ground_truth_cycle(GG)
    graph, graphdict = serializeGraphZeroOne(GG,m, n)
    input['Graph'] = graph
    return input, truth, graph

# creates possible routes for the cycle
# @param(size): size of the nodes in the graph
def create_routes(size):
    for i in range(size):
      # create an array of possible nodes and remove the current one
      arr = [i for i in range(size)]
      arr.remove(i)

      # get permutations per size
      for j in range(1,size-1):
        routes = list(itertools.permutations(arr, j+1))
        # append the starting node to the list
        for r in routes:
          r = list(r)
          r.insert(0,i)
          r.append(i)
          all_routes.append(r)

    return all_routes

# generates ground truth for the cycle detection
# @param(GG): nx graph object
def ground_truth_cycle(GG):
  return True if len(list(nx.cycle_basis(GG))) > 0 else False

# checks whether a cycle is present in the graph
# 1. create all possible routes for a cycle
# 2. shift the graph so that weight between selected nodes are present
# 3. multiply weights on the each consequtive nodes in the route
# 4. add result to accumulated 
# 5. if accumulated is greater than zero, it means that there is a cycle in the graph
def graphanalticprogram(graph, n):
    routes = create_routes(size=n)
    accumulated = [0]
    temp = [1]

    @lru_cache()
    def getWeight(i,j):
        return graph<<((i*n + j))

    for route in routes:
        for i in range(len(route)-1):
            from_node = route[i]
            to_node = route[i+1]
            # shift the graph object so that weight between two selected nodes is in the first index
            weight = getWeight(min(from_node,to_node), max(from_node,to_node)) 
            # graph<<((from_node*n + to_node))
            temp = temp * weight
            accumulated = accumulated + temp
        temp = [1]

    return accumulated
    
# Do not change this 
# the parameter n can be passed in the call from simulate function
class EvaProgramDriver(EvaProgram):
    def __init__(self, name, vec_size=4096, n=4):
        self.n = n
        super().__init__(name, vec_size)

    def __enter__(self):
        super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

# Repeat the experiments and show averages with confidence intervals
# You can modify the input parameters
# n is the number of nodes in your graph
# If you require additional parameters, add them
def simulate(n):
    m = 4096*4
    print("Will start simulation for ", n)
    config = {}
    config['warn_vec_size'] = 'false'
    config['lazy_relinearize'] = 'true'
    config['rescaler'] = 'always'
    config['balance_reductions'] = 'true'
    inputs, truth = prepareInput(n, m)

    graphanaltic = EvaProgramDriver("graphanaltic", vec_size=m,n=n)
    with graphanaltic:
        graph = Input('Graph')
        acc = graphanalticprogram(graph,n)
        # acc[0] has the accumulated val
        Output('ReturnedValue', acc)
        Output('Ground Truth', truth)

    prog = graphanaltic
    prog.set_output_ranges(30)
    prog.set_input_scales(30)

    start = timeit.default_timer()
    compiler = CKKSCompiler(config=config)
    compiled_multfunc, params, signature = compiler.compile(prog)
    compiletime = (timeit.default_timer() - start) * 1000.0 #ms

    start = timeit.default_timer()
    public_ctx, secret_ctx = generate_keys(params)
    keygenerationtime = (timeit.default_timer() - start) * 1000.0 #ms
    
    start = timeit.default_timer()
    encInputs = public_ctx.encrypt(inputs, signature)
    encryptiontime = (timeit.default_timer() - start) * 1000.0 #ms

    start = timeit.default_timer()
    encOutputs = public_ctx.execute(compiled_multfunc, encInputs)
    executiontime = (timeit.default_timer() - start) * 1000.0 #ms

    start = timeit.default_timer()
    outputs = secret_ctx.decrypt(encOutputs, signature)
    decryptiontime = (timeit.default_timer() - start) * 1000.0 #ms

    start = timeit.default_timer()
    reference = evaluate(compiled_multfunc, inputs)
    referenceexecutiontime = (timeit.default_timer() - start) * 1000.0 #ms
    
    # Change this if you want to output something or comment out the two lines below
    for key in outputs:
        print(key, float(outputs[key][0]), float(reference[key][0]))

    mse = valuation_mse(outputs, reference) # since CKKS does approximate computations, this is an important measure that depicts the amount of error

    return compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, mse


######################################3
import collections

compiler_config = {}
compiler_config['warn_vec_size'] = 'true'
compiler_config['lazy_relinearize'] = 'true'
compiler_config['rescaler'] = 'always'
compiler_config['balance_reductions'] = 'true'

compiler = CKKSCompiler(config=compiler_config)

def generate_simulation_reqs(node_count, vec_size):
    eva_prog = EvaProgram("detect_cycle", vec_size=vec_size)
    
    with eva_prog:
        graph = Input('Graph')

        res = []
        for i in range(node_count):
            shifted = graph << (i*node_count)
            res.append(shifted)

        for i in range(node_count):
            Output(f"Neighbours_{i}", res[i])

    eva_prog.set_output_ranges(30)
    eva_prog.set_input_scales(30)

    start_time = timeit.default_timer()
    compiled_func, params, signature = compiler.compile(eva_prog)
    compilation_time = (timeit.default_timer() - start_time) * 1000.0

    start_time = timeit.default_timer()
    public_ctx, secret_ctx = generate_keys(params)
    keygen_time = (timeit.default_timer() - start_time) * 1000.0

    return compiled_func, public_ctx, secret_ctx, signature, compilation_time, keygen_time

def get_neighbours(node_count, node, graph, data_collector, public_ctx,secret_ctx, signature , compiled_func):
    start_time = timeit.default_timer()
    enc_inputs = public_ctx.encrypt(graph, signature)
    encryptiontime = (timeit.default_timer() - start_time) * 1000.0

    start_time = timeit.default_timer()
    enc_outputs = public_ctx.execute(compiled_func, enc_inputs)
    executiontime = (timeit.default_timer() - start_time) * 1000.0

    start_time = timeit.default_timer()
    outputs = secret_ctx.decrypt(enc_outputs, signature)
    decryptiontime = (timeit.default_timer() - start_time) * 1000.0

    start_time = timeit.default_timer()
    reference = evaluate(compiled_func, graph)
    referenceexecutiontime = (timeit.default_timer() - start_time) * 1000.0

    mse = valuation_mse(outputs, reference)

    data_collector["encryptiontime"].append(encryptiontime)
    data_collector["executiontime"].append(executiontime)
    data_collector["referenceexecutiontime"].append(referenceexecutiontime)
    data_collector["decryptiontime"].append(decryptiontime)
    data_collector["mse"].append(mse)

    # tidy output
    raw_output = outputs[f"Neighbours_{node}"]
    threshold = 0.5
    neighbours = []

    for i in range(node_count):
        if (i == node):
            continue
        if raw_output[i] > threshold:
            neighbours.append(i)      

    return neighbours

def cycle_helper(node_count, current, visited, parent, graph, data_collector, public_ctx,secret_ctx, signature , compiled_func):
    visited[current] = True

    neighbours = get_neighbours(node_count, current,graph, data_collector, public_ctx,secret_ctx,  signature , compiled_func)

    for n in neighbours:
        if visited[n] == False:
            if cycle_helper(node_count, n, visited, current, graph, data_collector, public_ctx,secret_ctx, signature , compiled_func):
                return True
        elif parent != n:
            return True
    return False

def make_simulation(node_count):
    # compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, mse

    compiled_func, public_ctx, secret_ctx, signature, compilation_time, keygen_time = generate_simulation_reqs(node_count, 4096)

    data_collector = collections.defaultdict(list)
    data_collector["compiletime"].append(compilation_time)
    data_collector["keygenerationtime"].append(keygen_time)

    graph, truth, graph_arr = prepareInput(node_count, 4096)

    visited = [False]*(node_count)

    result = False
    for i in range(node_count):
        if visited[i] == False:
            if cycle_helper(node_count, i, visited, -1, graph, data_collector, public_ctx,secret_ctx, signature , compiled_func ):
                result = True
    
    return result, truth, data_collector

def process_data_collector(data_collector):
    compiletime = data_collector["compiletime"][0]
    keygenerationtime = data_collector["keygenerationtime"][0]
    encryptiontime =0 
    executiontime = 0 
    referenceexecutiontime = 0
    decryptiontime = 0
    mse = 0
    for x in data_collector["encryptiontime"]:
        encryptiontime += x
    for x in data_collector["executiontime"]:
        executiontime += x
    for x in data_collector["decryptiontime"]:
        decryptiontime += x
    for x in data_collector["referenceexecutiontime"]:
        referenceexecutiontime += x
    for x in data_collector[mse]:
        mse += x
    mse = mse / len(data_collector["mse"])
    

    return compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, mse

if __name__ == "__main__":
    simcnt = 100

    #Note that file is opened in append mode, previous results will be kept in the file
    resultfile = open("results.csv", "a")  # Measurement results are collated in this file for you to plot later on
    resultfile.write("NodeCount,SimCnt,CompileTime,KeyGenerationTime,EncryptionTime,ExecutionTime,DecryptionTime,ReferenceExecutionTime,Mse\n")

    print("Simulation campaing started:")
    for nc in [4,8,16,32]: # Node counts for experimenting various graph sizes
        node_count = nc
        for i in range(simcnt):
            print("node count: ", nc, " step: ", i)
            result, truth, data_collector = make_simulation(node_count)
            compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, mse = process_data_collector(data_collector)
            res = str(node_count) + "," + str(i) + "," + str(compiletime) + "," + str(keygenerationtime) + "," +  str(encryptiontime) + "," +  str(executiontime) + "," +  str(decryptiontime) + "," +  str(referenceexecutiontime) + "," +  str(mse) + "\n"
            resultfile.write(res)
    resultfile.close()