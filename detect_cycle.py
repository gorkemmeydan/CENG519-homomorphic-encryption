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
def serializeGraphZeroOne(GG,vec_size):
    n = GG.size()
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
    graph, graphdict = serializeGraphZeroOne(GG,m)
    input['Graph'] = graph
    return input, truth

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


if __name__ == "__main__":
    simcnt = 50 #The number of simulation runs, set it to 3 during development otherwise you will wait for a long time
    # For benchmarking you must set it to a large number, e.g., 100
    #Note that file is opened in append mode, previous results will be kept in the file
    resultfile = open("results.csv", "a")  # Measurement results are collated in this file for you to plot later on
    resultfile.write("NodeCount,SimCnt,CompileTime,KeyGenerationTime,EncryptionTime,ExecutionTime,DecryptionTime,ReferenceExecutionTime,Mse\n")
    resultfile.close()
    
    print("Simulation campaing started:")
    for nc in range(4,5,1): # Node counts for experimenting various graph sizes
        n = nc
        resultfile = open("results.csv", "a") 
        for i in range(simcnt):
            print("sim set ", i," out of ", simcnt)
            #Call the simulator
            compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, mse = simulate(n)
            res = str(n) + "," + str(i) + "," + str(compiletime) + "," + str(keygenerationtime) + "," +  str(encryptiontime) + "," +  str(executiontime) + "," +  str(decryptiontime) + "," +  str(referenceexecutiontime) + "," +  str(mse) + "\n"
            # print(res)
            resultfile.write(res)
        resultfile.close()