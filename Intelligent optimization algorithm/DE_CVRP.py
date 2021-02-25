import pandas as pd
import math
import random
import numpy as np
import copy
import xlsxwriter
import matplotlib.pyplot as plt
class Sol():
    def __init__(self):
        self.nodes_seq=None
        self.obj=None
        self.routes=None
class Node():
    def __init__(self):
        self.id=0
        self.name=''
        self.seq_no=0
        self.x_coord=0
        self.y_coord=0
        self.demand=0
class Model():
    def __init__(self):
        self.best_sol=None
        self.node_list=[]
        self.sol_list=[]
        self.node_seq_no_list=[]
        self.depot=None
        self.number_of_nodes=0
        self.opt_type=0
        self.vehicle_cap=0
        self.Cr=0.5
        self.F=0.5
        self.popsize=4*self.number_of_nodes

def readXlsxFile(filepath,model):
    # It is recommended that the vehicle depot data be placed in the first line of xlsx file
    node_seq_no =-1 #the depot node seq_no is -1,and demand node seq_no is 0,1,2,...
    df = pd.read_excel(filepath)
    for i in range(df.shape[0]):
        node=Node()
        node.id=node_seq_no
        node.seq_no=node_seq_no
        node.x_coord= df['x_coord'][i]
        node.y_coord= df['y_coord'][i]
        node.demand=df['demand'][i]
        if df['demand'][i] == 0:
            model.depot=node
        else:
            model.node_list.append(node)
            model.node_seq_no_list.append(node_seq_no)
        try:
            node.name=df['name'][i]
        except:
            pass
        try:
            node.id=df['id'][i]
        except:
            pass
        node_seq_no=node_seq_no+1
    model.number_of_nodes=len(model.node_list)
def genInitialSol(model):
    nodes_seq=copy.deepcopy(model.node_seq_no_list)
    for i in range(model.popsize):
        seed=int(random.randint(0,10))
        random.seed(seed)
        random.shuffle(nodes_seq)
        sol=Sol()
        sol.nodes_seq=copy.deepcopy(nodes_seq)
        sol.obj,sol.routes=calObj(nodes_seq,model)
        model.sol_list.append(sol)
        if sol.obj<model.best_sol.obj:
            model.best_sol=copy.deepcopy(sol)

def splitRoutes(nodes_seq,model):
    num_vehicle = 0
    vehicle_routes = []
    route = []
    remained_cap = model.vehicle_cap
    for node_no in nodes_seq:
        if remained_cap - model.node_list[node_no].demand >= 0:
            route.append(node_no)
            remained_cap = remained_cap - model.node_list[node_no].demand
        else:
            vehicle_routes.append(route)
            route = [node_no]
            num_vehicle = num_vehicle + 1
            remained_cap =model.vehicle_cap - model.node_list[node_no].demand
    vehicle_routes.append(route)
    return num_vehicle,vehicle_routes
def calDistance(route,model):
    distance=0
    depot=model.depot
    for i in range(len(route)-1):
        from_node=model.node_list[route[i]]
        to_node=model.node_list[route[i+1]]
        distance+=math.sqrt((from_node.x_coord-to_node.x_coord)**2+(from_node.y_coord-to_node.y_coord)**2)
    first_node=model.node_list[route[0]]
    last_node=model.node_list[route[-1]]
    distance+=math.sqrt((depot.x_coord-first_node.x_coord)**2+(depot.y_coord-first_node.y_coord)**2)
    distance+=math.sqrt((depot.x_coord-last_node.x_coord)**2+(depot.y_coord - last_node.y_coord)**2)
    return distance
def calObj(nodes_seq,model):
    num_vehicle, vehicle_routes = splitRoutes(nodes_seq, model)
    if model.opt_type==0:
        return num_vehicle,vehicle_routes
    else:
        distance=0
        for route in vehicle_routes:
            distance+=calDistance(route,model)
        return distance,vehicle_routes
def adjustRoutes(nodes_seq,model):
    all_node_list=copy.deepcopy(model.node_seq_no_list)
    repeat_node=[]
    for id,node_no in enumerate(nodes_seq):
        if node_no in all_node_list:
            all_node_list.remove(node_no)
        else:
            repeat_node.append(id)
    for i in range(len(repeat_node)):
        nodes_seq[repeat_node[i]]=all_node_list[i]
    return nodes_seq
#Differential mutation; mutation strategies: DE/rand/1/bin
def muSol(model,v1):
    x1=model.sol_list[v1].nodes_seq
    while True:
        v2=random.randint(0,model.number_of_nodes-1)
        if v2!=v1:
            break
    while True:
        v3=random.randint(0,model.number_of_nodes-1)
        if v3!=v2 and v3!=v1:
            break
    x2=model.sol_list[v2].nodes_seq
    x3=model.sol_list[v3].nodes_seq
    mu_x=[min(int(x1[i]+model.F*(x2[i]-x3[i])),model.number_of_nodes-1) for i in range(model.number_of_nodes) ]
    return mu_x
#Differential Crossover
def crossSol(model,vx,vy):
    cro_x=[]
    for i in range(model.number_of_nodes):
        if random.random()<model.Cr:
            cro_x.append(vy[i])
        else:
            cro_x.append(vx[i])
    cro_x=adjustRoutes(cro_x,model)
    return cro_x

def plotObj(obj_list):
    plt.rcParams['font.sans-serif'] = ['SimHei'] #show chinese
    plt.rcParams['axes.unicode_minus'] = False  # Show minus sign
    plt.plot(np.arange(1,len(obj_list)+1),obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Obj Value')
    plt.grid()
    plt.xlim(1,len(obj_list)+1)
    plt.show()
def outPut(model):
    work=xlsxwriter.Workbook('result.xlsx')
    worksheet=work.add_worksheet()
    worksheet.write(0,0,'opt_type')
    worksheet.write(1,0,'obj')
    if model.opt_type==0:
        worksheet.write(0,1,'number of vehicles')
    else:
        worksheet.write(0, 1, 'drive distance of vehicles')
    worksheet.write(1,1,model.best_sol.obj)
    for row,route in enumerate(model.best_sol.routes):
        worksheet.write(row+2,0,'v'+str(row+1))
        r=[str(i)for i in route]
        worksheet.write(row+2,1, '-'.join(r))
    work.close()
def run(filepath,epochs,Cr,F,popsize,v_cap,opt_type):
    """
    :param filepath: Xlsx file path
    :param epochs: Iterations
    :param Cr: crossover rate
    :param F:  scaling factor
    :param popsize: population size
    :param v_cap: Vehicle capacity
    :param opt_type: Optimization type:0:Minimize the number of vehicles,1:Minimize travel distance
    :return:
    """
    model=Model()
    model.vehicle_cap=v_cap
    model.Cr=Cr
    model.F=F
    model.popsize=popsize
    model.opt_type=opt_type

    readXlsxFile(filepath,model)
    best_sol = Sol()
    best_sol.obj = float('inf')
    model.best_sol = best_sol
    genInitialSol(model)
    history_best_obj = []
    for ep in range(epochs):
        for i in range(popsize):
            v1=random.randint(0,model.number_of_nodes-1)
            sol=model.sol_list[v1]
            mu_x=muSol(model,v1)
            u=crossSol(model,sol.nodes_seq,mu_x)
            u_obj,u_routes=calObj(u,model)
            if u_obj<=sol.obj:
                sol.nodes_seq=copy.deepcopy(u)
                sol.obj=copy.deepcopy(u_obj)
                sol.routes=copy.deepcopy(u_routes)
                if sol.obj<model.best_sol.obj:
                    model.best_sol=copy.deepcopy(sol)
            history_best_obj.append(model.best_sol.obj)
        print("%s/%s， best obj: %s" % (ep, epochs, model.best_sol.obj))
    plotObj(history_best_obj)
    outPut(model)

if __name__ == '__main__':
    file = '../data/cvrp.xlsx'
    run(filepath=file, epochs=150, Cr=0.5,F=0.5, popsize=400, v_cap=80, opt_type=1)

