from collections import deque
import csv
import random
import math
import numpy
from tqdm import tqdm
import networkx as nx
from pathlib import Path
from queue import PriorityQueue


# Generalized Watts-Strogatz (EK 20)
# n is the number of nodes (we assume n is a perfect square or it will be rounded to the closest perfect square)
# r is the radius of each node (a node u is connected with each other node at distance at most r) - strong ties
# k is the number of random edges for each node u - weak ties
#
# Here, the weak ties are still proportional to distance
# q is a term that evaluate how much the distance matters.
# Specifically, the probability of an edge between u and v is proportional to 1/dist(u,v)**q
# Hence, q=0 means that weak ties are placed uniformly at random, q=infinity only place weak ties towards neighbors.
#
# Next implementation of Watts-Strogatz graphs assumes that nodes are on a two-dimensional space (similar implementation can be given on larger dimensions).
# Here, distance between nodes will be set to be the Euclidean distance.
# This approach allows us a more fine-grained and realistic placing of nodes (i.e., they not need to be all at same distance as in the grid)
def GenWS2DG(n, r, k, q):
    G = nx.Graph()
    nodes=dict() #This will be used to associate to each node its coordinates
    prob=dict() #Keeps for each pair of nodes (u,v) the term 1/dist(u,v)**q

    # dim is the dimension of the area in which we assume nodes are placed.
    # Here, we assume that the 2D area has dimension sqrt(n) x sqrt(n).
    # Anyway, one may consider a larger or a smaller area.
    # E.g., if dim = 1 we assume that all features of a nodes are within [0,1].
    # However, recall that the radius r given in input must be in the same order of magnitude as the size of the area
    # (e.g., you cannot consider the area as being a unit square, and consider a radius 2, otherwise there will be an edge between each pair of nodes)
    # Hence, the choice of larger values for dim can be preferred if one want to represent r as an integer and not a floating point number
    dim = math.sqrt(n)

    # The following for loop creates n nodes and place them randomly in the 2D area.
    # If one want to consider a different placement, e.g., for modeling communities, one only need to change this part.

    print("nodes_creation: ")
    for i in tqdm(range(n)):
        x=random.random()
        y=random.random()
        nodes[i]=(x*dim,y*dim)
        prob[i]=dict()

    print("edges_creation: ")
    for i in tqdm(range(n)):
        # Strong-ties
        for j in range(i+1,n):
            # we add edge only towards next nodes,
            # since edge to previous nodes have been added when these nodes have been processed
            dist = math.sqrt((nodes[i][0]-nodes[j][0])**2 + (nodes[i][1]-nodes[j][1])**2) #Euclidean Distance
            prob[i][j] = 1/(dist**q)
            prob[j][i] = prob[i][j]
            if dist <= r:
                G.add_edge(str(i), str(j))

        # Terms 1/dist(u,v)**q are not probabilities since their sum can be different from 1.
        # To translate them in probabilities we normalize them, i.e, we divide each of them for their sum
        norm=sum(prob[i].values())
        # Weak ties
        for h in range(k):
            # They are not exactly h, since the random choice can return a node s such that edge (i, s) already exists
            # Next instruction allows to choice from the list given as first argument according to the probability distribution given as second argument
            s=numpy.random.choice([x for x in range(n) if x != i],p=[prob[i][x]/norm for x in range(n) if x != i])
            G.add_edge(str(i), str(s))

    return G

def auction_results(allocations: dict, bids: dict, payments: dict):
    rw = sum(payments.values())
    sw = 0.
    for bidder, alloc in allocations.items():
        if alloc:
            sw += bids[bidder]
    
    return sw, rw

def create_auction(num_nodes, r = 2.71,k = 1, q=4):
    G = GenWS2DG(num_nodes,r,k,q)
    # scelta casuale del seller
    seller = random.choice(list(G.nodes()))
    # costruzione di seller_net
    seller_net = {bidder for bidder in G[seller]}

    # costruzione dizionario reports
    reports = {}
    level = deque([seller])
    visited = [seller]

    while len(level) > 0:

        n = level.popleft()
        for c in G[n]:
            if c not in visited:
                level.append(c)
                visited.append(c)
                if n not in reports:
                    reports[n] = [c]
                else:
                    reports[n].append(c)

    del reports[seller]
    # costruzione di bids
    bids = {}
    for n in G.nodes():
        bid = random.randrange(1, 10000, 1)
        bids[n] = bid

    return seller_net, reports, bids

def BFS(G,u):
    """
    A BFS algorithm that returns the set of nodes reachable from u in the graph G

    Parameters
    ----------
    G: nx.Graph or nx.DiGraph
        A networkx undirected or directed graphs
    u: node
        A node of G

    Returns
    ---------
    set:
        the set of nodes reachable from u in the graph G
    """
    clevel=[u]
    visited=set(u)
    while len(clevel) > 0:
        nlevel=[]
        for c in clevel:
            for v in G[c]:
                if v not in visited:
                    visited.add(v)
                    nlevel.append(v)
        clevel = nlevel
    return visited


#Returns the top k nodes of G according to the centrality measure "measure"
def top(G,cen,k):
    pq = PriorityQueue()
    for u in G.nodes():
        x = -cen[u]
        pq.put((x,u))  # We use negative value because PriorityQueue returns first values whose priority value is lower
    
    out={}
    for i in range(k):
        x = pq.get()
        out[x[1]] = -x[0]
    return out

#Returns the top k nodes of G according to the centrality measure "measure"
def bottom(G,cen,k):
    pq = PriorityQueue()
    for u in G.nodes():
        x = cen[u]
        pq.put((x,u))  # We use negative value because PriorityQueue returns first values whose priority value is lower
    
    out={}
    for i in range(k):
        x = pq.get()
        out[x[1]] = -x[0]
    return out


def create_graph_from_csv(filename, sep=',', directed=False):
    path = Path(filename)
    if not path.is_file():
        raise ValueError('File Not Exist!')

    if not directed:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=sep)
        # l'intestazione del csv non viene considerata
        header = next(reader)
        #header =1
        if header is not None:
            for row in reader:
                if len(row) == 2:
                    u, v = row
                    G.add_edge(u, v)
                else:
                    u, v, _, _ = row
                    G.add_edge(u, v)

    return G


def create_graph_from_txt(filename, sep=',', directed=False):
    path = Path(filename)
    if not path.is_file():
        raise ValueError('File Not Exist!')

    if not directed:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    file = open(filename, 'r')
    lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line.startswith('#'):
            row = line.split(sep)
            if len(row) == 2:
                u, v = row
                G.add_edge(u, v)

            else:
                raise ValueError('Format File Error!')

    return G


def chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# def chunks(data, size):
#     idata = iter(data)
#     for i in range(0, len(data), size):
#         yield {k: data[k] for k in it.islice(idata, size)}

if __name__ == '__main__':
    G = create_graph_from_csv('data/musae_facebook_edges.csv')
    print('Numero di nodi:', G.number_of_nodes())
    print('Numero di archi: ', G.number_of_edges())
    G2 = create_graph_from_txt('data/Cit-HepTh.txt', directed=True, sep='\t')
    print('Numero di nodi:', G2.number_of_nodes())
    print('Numero di archi: ', G2.number_of_edges())
