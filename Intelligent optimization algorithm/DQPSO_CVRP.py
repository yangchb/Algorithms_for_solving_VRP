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
        self.sol_list=[]
        self.best_sol=None
        self.node_list=[]
        self.node_seq_no_list=[]
        self.depot=None
        self.number_of_nodes=0
        self.opt_type=0
        self.vehicle_cap=0
        self.popsize=100
        self.pl=[]
        self.pg=None
        self.mg=None
        self.alpha=1.0
def readXlsxFile(filepath,model):
    # It is recommended that the vehicle depot data be placed in the first line of xlsx file
    node_seq_no = -1 #the depot node seq_no is -1,and demand node seq_no is 0,1,2,...
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
    node_seq=copy.deepcopy(model.node_seq_no_list)
    best_sol=Sol()
    best_sol.obj=float('inf')
    mg=[0]*model.number_of_nodes
    for i in range(model.popsize):
        seed = int(random.randint(0, 10))
        random.seed(seed)
        random.shuffle(node_seq)
        sol=Sol()
        sol.nodes_seq= copy.deepcopy(node_seq)
        sol.obj,sol.routes=calObj(sol.nodes_seq,model)
        model.sol_list.append(sol)
        #init the optimal position of each particle
        model.pl.append(sol.nodes_seq)
        #init the average optimal position of particle population
        mg=[mg[k]+node_seq[k]/model.popsize for k in range(model.number_of_nodes)]
        #init the optimal position of particle population
        if sol.obj<best_sol.obj:
            best_sol=copy.deepcopy(sol)
    model.best_sol=best_sol
    model.pg=best_sol.nodes_seq
    model.mg=mg
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

def updatePosition(model):
    alpha=model.alpha
    pg=model.pg
    mg=model.mg
    mg_=[0]*model.number_of_nodes  #update optimal position of each particle for next iteration
    for id, sol in enumerate(model.sol_list):
        x=sol.nodes_seq
        pl = model.pl[id]
        pi=[]
        for k in range(model.number_of_nodes): #calculate pi(ep+1)
            phi = random.random()
            pi.append(phi*pl[k]+(1-phi)*pg[k])
        #calculate x(ep+1)
        if random.random()<=0.5:
            X=[min(int(pi[k]+alpha*abs(mg[k]-x[k])*math.log(1/random.random())),model.number_of_nodes-1)
               for k in range(model.number_of_nodes)]
        else:
            X=[min(int(pi[k]-alpha*abs(mg[k]-x[k])*math.log(1/random.random())),model.number_of_nodes-1)
               for k in range(model.number_of_nodes)]

        X= adjustRoutes(X, model)
        X_obj, X_routes = calObj(X,model)
        # update pl
        if X_obj < sol.obj:
            model.pl[id] = copy.deepcopy(X)
        # update pg,best_sol
        if X_obj < model.best_sol.obj:
            model.best_sol.obj = copy.deepcopy(X_obj)
            model.best_sol.nodes_seq = copy.deepcopy(X)
            model.best_sol.routes = copy.deepcopy(X_routes)
            model.pg = copy.deepcopy(X)
        mg_ = [mg_[k] + model.pl[id][k] / model.popsize for k in range(model.number_of_nodes)]
        model.sol_list[id].nodes_seq = copy.deepcopy(X)
        model.sol_list[id].obj = copy.deepcopy(X_obj)
        model.sol_list[id].routes = copy.deepcopy(X_routes)
    # update mg
    model.mg=copy.deepcopy(mg_)
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
def run(filepath,epochs,popsize,alpha,v_cap,opt_type):
    """
    :param filepath: Xlsx file path
    :type str
    :param epochs:Iterations
    :type int
    :param popsize:Population size
    :type int
    :param alpha:Innovation(Control) parameters,(0,1]
    :type float,
    :param v_cap:Vehicle capacity
    :type float
    :param opt_type:Optimization type:0:Minimize the number of vehicles,1:Minimize travel distance
    :type int,0 or 1
    :return:
    """
    model=Model()
    model.vehicle_cap=v_cap
    model.opt_type=opt_type
    model.alpha=alpha
    model.popsize=popsize
    readXlsxFile(filepath,model)
    history_best_obj=[]
    genInitialSol(model)
    history_best_obj.append(model.best_sol.obj)
    for ep in range(epochs):
        updatePosition(model)
        history_best_obj.append(model.best_sol.obj)
        print("%s/%s: best obj: %s"%(ep,epochs,model.best_sol.obj))
    plotObj(history_best_obj)
    outPut(model)
if __name__=='__main__':
    file='../data/cvrp.xlsx'
    run(filepath=file,epochs=300,popsize=150,alpha=0.8,v_cap=80,opt_type=1)
