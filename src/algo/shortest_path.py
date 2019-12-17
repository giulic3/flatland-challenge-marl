import numpy as np
import time
#import bisect

#un nodo e' definito da un id. Una mappa connections associa all'id un array di 0 o 1
#con le posisibili connessione NE, NS, NO, ES, EO, SO (array 4*4)

#un arco (non orientato) e' un id, con informazioni relative a ((id1,c1),(id2,c2),lunghezza)
# dove c1 e c2 sono N,E,SU,O

#ogni treno ha un suo id, una posizione definita da
# - un binario (un arco)
# - una direzione di attraversamento (0,1)
# - la distanza percorsa sul binario (int)
# una velocita
# una destinazione (id del nodo)
# altre info

def insert(a, frontier, visited):
    """
    
    :param a: 
    :param frontier: 
    :param visited: 
    :return: 
    """
    (t, ct), length, tim, path = a
    if (t,ct) in visited:
        #print("visited : {}".format((t,ct)))
        return frontier
    elif frontier==[]:
        return([a])
    else:
        new_frontier = []
        (x,xc), xlength, xtime, xpath = frontier[0]
        tail = frontier[1:]
        while xlength < length and not(tail==[]):
            new_frontier.append(((x,xc), xlength, xtime, xpath))
            (x,xc), xlength, xtime, xpath = tail[0]
            tail = tail[1:]
        if xlength < length: 
            new_frontier.append(((x,xc), xlength, xtime, xpath))
        else:
            tail.append(((x,xc), xlength, xtime, xpath))  #put back in tail
        new_frontier.append(((t,ct), length, tim, path))
        new_frontier = new_frontier + tail
        
        return new_frontier

def insert_wrt_time(a, frontier, visited):
    """
    
    :param a: 
    :param frontier: 
    :param visited: 
    :return: 
    """
    (t,ct), length, tim, path = a
    if (t,ct) in visited:
        #print("visited : {}".format((t,ct)))
        return frontier
    elif frontier==[]:
        return [a]
    else:
        new_frontier = []
        (x,xc), xlength, xtime, xpath = frontier[0]
        tail = frontier[1:]
        while xtime < tim and not(tail==[]):
            new_frontier.append(((x,xc), xlength, xtime, xpath))
            (x,xc), xlength, xtime, xpath = tail[0]
            tail = tail[1:]
        if xtime < tim: 
            new_frontier.append(((x,xc),xlength,xtime,xpath))
        else:
            tail.append(((x,xc), xlength, xtime, xpath))  #put back in tail
        new_frontier.append(((t,ct),length,tim,path))
        new_frontier = new_frontier + tail
        return new_frontier    

def find_shortest_paths(G, train, available_at, info, connections):
    #print(available_at)
    #available_at is a map associating to each rail the timestep
    #at which it will be available. We keep the invariant that if a
    #rail is available at time t0, it will be available at any time
    #t > t0 (with the current scheduling)
    #The greedy policy books, for each train, ALL rails along its path,
    #until its transit on the rail
    
    V, E = G
    shortest_paths = []
    id_train, pos, target, speed = train
    for cp in target[1]: # Build a shortest path for each possible entry point of the target node
        #print("train id: {}".format(id_train))
        rail, direction, dist = pos
        (s, cs), (t, ct), l = info[rail]
        
        current_length = l-dist
        current_time = int(current_length/speed) + available_at[rail]
        current_path = [(
            rail, #id del binario
            direction, #direzione di percorrenza
            current_time, #at exit time the rail will be availbale again
        )]
        visited = []
        frontier = []
        if direction == 1:
            current_node, current_c = t, ct
        else:
            current_node, current_c = s, cs
       
        #invariante: mantengo la frontier ordinata rispetto alle lunghezze correnti 
        #todo = [n for n in V if not(n==target_node)]
        #print("ct before while {}",current_time)
        while not (current_node, current_c) == (target[0], cp):
            #print(frontier)
            #time.sleep(1)
            visited.append((current_node,current_c))
            #cerco i binari adiacenti al current_node
            for e in E:
                (s,cs),(t,ct),l = info[e]
                if ((s == current_node and connections[current_node][current_c, cs] == 1) or
                    (t == current_node and connections[current_node][current_c, ct] == 1)):
                    new_length = current_length + l
                    new_path = list(current_path)
                    #if we want to use this rail, we need to wait until it is available
                    new_current_time = max(current_time, available_at[e])
                    transit_time = new_current_time +int(l/speed)
                    if s == current_node:
                        #print("right {}".format(e))
                        new_path.append((e, 1, transit_time))  #direction = 1 !!
                        #aggiungo alla frontiera la tuple ((t,ct),s_lenght,new_path) se non e' gia'
                        #stata visitata
                        #print(frontier)
                        #frontier = insert(((t,ct),newlenght,transit_time,new_path),frontier,visited)
                        frontier = insert_wrt_time(((t,ct),new_length,transit_time,new_path),frontier,visited)
                        #print(frontier)
                    else: #t == current_node
                        #print("left {}".format(e))
                        new_path.append((e, 0, transit_time))  #direction = 0 !!
                        #frontier = insert(((s,cs),newlenght,transit_time,new_path),frontier,visited)
                        frontier = insert_wrt_time(((s,cs), new_length, transit_time, new_path), frontier, visited)
            if not frontier:
                # print("No path found")
                current_path = [] #empty path means failure
                break
            else:
                (current_node, current_c), current_length, current_time, current_path = frontier[0]
                #print("current from frontier = {}", current_time)
                frontier = frontier[1:]
    
        shortest_paths.append((current_length, current_path))
    # return current_length, current_path
    index = 0
    min_length = shortest_paths[0][0]
    for i in range(1, len(shortest_paths)):
        if shortest_paths[i][0] < min_length:
            min_length = shortest_paths[i][0]
            index = i
    
    return shortest_paths[index]


#esempio
'''
EO = np.zeros((4,4))
EO[1,3]=EO[3,1]=1

NSO = np.zeros((4,4))
NSO[0,3]=NSO[3,0]=NSO[2,3]=NSO[3,2]=1

NES = np.zeros((4,4))
NES[0,1]=NES[1,0]=NES[1,2]=NES[2,1]=1


V = [0,1,2,3]

def connection(n):
    if n == 0:
        return EO
    elif n== 1:
        return NSO
    elif n==2:
        return NES
    elif n==3:
        c = np.zeros((4,4))
        c[3,0]=c[0,3]=c[1,3]=c[3,1]=1
        return(c)

E = [0,1,2,3,4]

def info(e):
    if e == 0: return ((0,1),(1,3),5)
    elif e == 1:
        return ((1,0),(2,0),3)
    elif e == 2:
        return ((1,2),(2,2),4)
    elif e == 3:
        return ((2,1),(3,3),1)
    elif e == 4:
        return ((0,0),(3,0),8) # c'è un errore qua, il nodo 0 non ha connessione a nord TODO
'''
# no_trails = len(E)
# available_at = np.zeros(no_trails,dtype=int)


#lenght,path = find_shortest_paths((V,E),train,available_at)

def update_availability(available_at, path):
    """
    
    :param available_at: 
    :param path: 
    :return: 
    """
    for (id_edge, direction, transit_time) in path[1]:
        available_at[id_edge] = transit_time
    return available_at

#available_at = update_availability(available_at,path)
#print(available_at)


def scheduling(G, trains, info, connections):
    """
    
    :param G: 
    :param trains: 
    :param info: 
    :param connections: 
    :return: paths: , available_at: list of times at which edges will be available again (index in list corresponds to edge id)
    """
    paths = []
    num_rails = len(G[1]) # G[0] = vertices, G[1] = edges
    available_at = np.zeros(num_rails, dtype=int)
    # TODO Agents order matters
    for train in trains:
        path = find_shortest_paths(G, train, available_at, info, connections)
        paths.append(path)
        available_at = update_availability(available_at, path)
        
    return paths, available_at

'''
train = (0, #id del treno
         (0,1,0), #pos: id del binario, direzione, distanza percorsa
         (2,2), #target: nodo e punto cardinale
         1 #speed
         )

train2 = (1, #id del treno
         (0,1,0), #pos: id del binario, direzione, distanza percorsa
         (0,0), #target: nodo e punto cardinale
         .25 #speed
         )
'''
#paths, available_at = scheduling((V,E),[train2,train])

#print(paths)
#print(available_at)