import networkx as nx
import matplotlib.pyplot as plt
from numpy import *


def iniate_nodes(nodes):
    listenod = []
    c = 0
    while c < nodes:
        no = input("please enter the node of index " + str(c) + " ")
        listenod.append(no)
        c += 1
    print("all nodes where added")
    print(listenod)
    return listenod


def iniate_edges(edges, pondere):
    c = 0
    listeedg = []
    two=input("does the nodes have more than one chars? ")
    chars=False
    if two=="yes" or two=="y" or two=="Y":
        print("please use space between the nodes when inserting the edges!!")
        chars=True
    while c < edges:
        lis = input("please enter the edge: ")
        edg=lis
        if chars:
            edg=lis.split()
        if pondere:
            weight = int(input("please enter the weight of the edge "))
            listeedg.append((edg[0], edg[1], weight))
        else:
            listeedg.append((edg[0], edg[1]))
        c += 1
    print("all edges where added successfully!! \n time to draw")
    print(listeedg)
    return listeedg


def graph(listnode, listedges, pondere, oriented):
    if oriented:
        K = nx.DiGraph()
    else:
        K = nx.Graph()
    K.add_nodes_from(listnode)
    if pondere:
        K.add_weighted_edges_from(listedges)
        pos = nx.planar_layout(K)
        labels = nx.get_edge_attributes(K, 'weight')
        nx.draw_networkx(K, pos, with_labels=True)
        nx.draw_networkx_edge_labels(K, pos, edge_labels=labels)
    else:
        K.add_edges_from(listedges)
        nx.draw(K, with_labels=True)

    plt.savefig('grtrphit.png')
    plt.show()
    plt.close()

    stay = input("do you want to show the caracteristic: ")
    if stay == "no" or stay == "n" or stay == "N" or stay == "NO":
        return K
    if oriented:
        print("topological sorting: ")
        print(list(reversed(list(nx.topological_sort(K)))))

        x = input("how do you want the shortest path dijkstra or bellmand: ")
        if (x == 'd'):
            print("using dijkstra: ")
            pip, t = nx.dijkstra_predecessor_and_distance(K, listnode[0])
            for i in pip, t:
                print(i)
        elif x == "b":
            toti = list(nx.bellman_ford_predecessor_and_distance(K, listnode[0]))
            for i in toti:
                print(i)

        print("strongly connected : ", list(nx.strongly_connected_components(K)))
        print("number of strongly connected: ", len(list(nx.strongly_connected_components(K))))
        print("IS DAG", nx.is_directed_acyclic_graph(K))
    else:
        if(nx.is_connected(K)):

           exci = (nx.eccentricity(K))
           print("the eccentricity of the nodes are : ", exci)
           print("the central node is at  :", min(exci.values()))

           diam = nx.diameter(K)
           print("the diameter of the graoh is :", diam)


           print("the articulation point are: ")
           arti = list(nx.articulation_points(K))
           print(arti)

           edg = nx.minimum_edge_cut(K)
           # print("the type of edg is: ",type(edg))
           print("minimum edge cut: ")
           for i in edg:
               print(i)

        print("using bellmand-ford for the sortest path: ")
        toti = list(nx.bellman_ford_predecessor_and_distance(K, listnode[0]))
        for i in toti:
            print(i)



        n = list(nx.minimum_spanning_tree(K))
        T = list(nx.minimum_spanning_edges(K))
        print("showing the minimum spanning tree : ")
        for i in T:
            print(i)
        G = nx.Graph()
        G.add_nodes_from(n)
        if (pondere):
            G.add_weighted_edges_from(T)
        else:
            G.add_edges_from(T)
        pos = nx.planar_layout(G)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx(G, pos, with_labels=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.savefig('grtrphit.png')
        plt.show()
        plt.close()
        print("connexe? ", nx.is_connected(K))
        print("nombre de composante connexe: ", nx.number_connected_components(K))
        print('les composantes:', list(nx.connected_components(K)))

    listParcours = []

    x = input("what do you want to do largeur/profondeur: ")
    print("please choose a node :", listnode)
    chosen = int(input("enter the index node here: "))
    if (x == "l" or x == "L"):
        listParcours = list(nx.bfs_edges(K, listnode[chosen]))
    elif x == "p" or x == "P":
        listParcours = list(nx.dfs_edges(K, listnode[chosen]))
    for elm in listParcours:
        print(elm)

    # Matrice d'adjacence et fermeture transisive:
    M = nx.adjacency_matrix(K)
    print('adjacency matrix of K:', M)
    M = nx.attr_matrix(K, rc_order=listnode)
    print('adjacency matrix of K:', M)
    # print('M^3:', M ** 3)
    if not pondere:
        # # fermeture transitive
        o = K.order()
        Mb = mat(M, dtype=bool)
        I = eye(o, o, dtype=bool)
        TC = (Mb + I) ** (o - 1)
        print('transitive closure of G:\n', TC * 1)

    return K


def Dijkstra(G, s):

    dist = {}
    pred = {}
    for i in G.nodes():
        dist[i] = inf
        pred[i] = s

    dist[s] = 0
    X = list(G.nodes())
    print(dist)
    print(pred)
    while X:
        i = min(X, key=dist.get)
        for j in G.neighbors(i):
            if dist[j] > (dist[i] + G[i][j]['weight']):
                dist[j] = (dist[i] + G[i][j]['weight'])
                pred[j] = i
        print("after iteration: ")
        print(dist)
        print(pred)
        X.remove(i)

    return dist, pred


def Dijkstra_reverse(G, s):
    dist = {}
    pred = {}
    for i in G.nodes():
        dist[i] = inf
        pred[i] = s

    dist[s] = 0

    X = list(G.nodes())
    print(dist)
    print(pred)
    while X:
        i = min(X, key=dist.get)

        for j in G.predecessors(i):
            if dist[j] > (dist[i] + G[j][i]['weight']):
                dist[j] = (dist[i] + G[j][i]['weight'])
                pred[j] = i
        print("after iteration: ")
        print(dist)
        print(pred)
        X.remove(i)

    return dist, pred


def Arbo(G, s):
    F = []
    C = []
    marque = {}
    for i in G.nodes():
        marque[i] = 0
    F.append(s)
    marque[s] = 1
    while F:
        node = F.pop(0)
        for j in G.neighbors(node):
            if marque[j] == 0:
                F.append(j)
                marque[j] = 1
            else:
                C.append((node, j))
    return C



# Dijkstra Proba

def Dijkstra_Proba(G, s):
    dist = {}
    pred = {}
    for i in G.nodes():
        dist[i] = 0
        pred[i] = s

    dist[s] = 1
    X = list(G.nodes())

    while X:
        i = max(X, key=dist.get)

        for j in G.neighbors(i):
            if dist[j] < (dist[i] * G[i][j]['proba.txt']):
                dist[j] = dist[i] * G[i][j]['proba.txt']
                pred[j] = i
        X.remove(i)

    return dist, pred


# Dijkstra BandWidth

def Dijkstra_BW(G, s):
    dist = {}
    pred = {}
    for i in G.nodes():
        dist[i] = 0
        pred[i] = s

    dist[s] = inf
    X = list(G.nodes())

    while X:
        i = max(X,key=dist.get)

        for j in G.neighbors(i):
            if dist[j] < min(dist[i], G[i][j]['BW']):
                dist[j] = min(dist[i], G[i][j]['BW'])
                pred[j] = i
        X.remove(i)

    return dist, pred


def pf(G, s):
    n = G.order()
    F = []  # file vide
    order = 1  # ordre de visite
    marque = {}
    for i in G.nodes():
        marque[i] = -1;
    marque[s] = order
    F.append(s)
    while F:
        x = F.pop(0)
        for y in G.neighbors(x):
            if marque[y] == 0:
                F.append(y)
                order +=1
                marque[y] = order
    return marque


def pf_cycle(G, s):
    n = G.order()
    F = []  # file vide
    order = 1  # ordre de visite
    marque = {}
    liens_redd = []
    pred = {}
    for i in G.nodes():
        marque[i] = -1
        pred[i] = s
    marque[s] = order
    pred[s] = -1
    F.append(s)
    while F:
        x = F.pop(0)
        for y in G.neighbors(x):

            if marque[y] == -1:
                F.append(y)
                order = order + 1
                marque[y] = order
                pred[y] = x

            elif marque[y] != -1 and pred[x] != y:
                liens_redd.append((x, y))

    return marque, liens_redd




nodes = int(input("pleaase enter the number of nodes ?"))
edges = int(input("please enter the number of edges: ?"))
ispondere = input("please enter if it is pondere enter Y else N ?")
isoriented = input("please enter if the graph is oriented or not: >")

oriented = False
if isoriented == "y" or ispondere == "yes":
    oriented = True
pondere = False
if ispondere == "Y " or ispondere == "yes" or ispondere == "y":
    pondere = True

# listnode = iniate_nodes(nodes)
# listeedg = iniate_edges(edges, pondere)

lala=[('A', 'B', 3), ('B', 'F', 13), ('F', 'H', 3), ('H', 'G', 6), ('G', 'E', 6), ('E', 'C', 5), ('C', 'A', 2), ('D', 'B', 2), ('D', 'A', 5), ('D', 'C', 2), ('D', 'F', 6), ('D', 'G', 3), ('D', 'E', 4), ('F', 'G', 2)]
lolo=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']


grap = graph(lolo, lala, pondere, oriented)

#fille:
# .pop(0)

#sommet de la liste:
#.pop()

#Disjktra :
dis,pr=Dijkstra(grap,'F')
#print(dis)

#reverse dijkstra:
# print("reverse is now: ")
# tal,til=Dijkstra_reverse(grap,'V')

#Arborescence
# p=list(Arbo(grap,'1'))
#print("the edges that need to be cut are:",p)

#read oriented grap:
# wG = nx.read_edgelist('./hi.txt',create_using=nx.DiGraph(), nodetype=str, data=(('weight', int),))
# [d, p] = Dijkstra(wG, 'H1')
# print(d)
# print(p)

#read non oriented:
# G=nx.read_edgelist('./edges.txt',nodetype=int)


# wG = nx.read_edgelist('./proba.txt', nodetype=int, data=(('proba.txt', float),))
# [d, p] = Dijkstra_Proba(wG, 1)
# print(d)
# print(p)
