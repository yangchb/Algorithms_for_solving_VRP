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
        self.pl=[]
        self.pg=None
        self.v=[]
        self.Vmax = 5
        self.w=0.8
        self.c1=2
        self.c2=2
def readXlsxFile(filepath,model):
    # It is recommended that the vehicle depot data be placed in the first line of xlsx file
    node_seq_no = -1#the depot node seq_no is -1,and demand node seq_no is 0,1,2,...
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

def genInitialSol(model,popsize):
    node_seq=copy.deepcopy(model.node_seq_no_list)
    best_sol=Sol()
    best_sol.obj=float('inf')
    for i in range(popsize):
        seed = int(random.randint(0, 10))
        random.seed(seed)
        random.shuffle(node_seq)
        sol=Sol()
        sol.nodes_seq= copy.deepcopy(node_seq)
        sol.obj,sol.routes=calObj(sol.nodes_seq,model)
        model.sol_list.append(sol)
        model.v.append([model.Vmax]*model.number_of_nodes)
        model.pl.append(sol.nodes_seq)
        if sol.obj<best_sol.obj:
            best_sol=copy.deepcopy(sol)
    model.best_sol=best_sol
    model.pg=best_sol.nodes_seq

def updatePosition(model):
    w=model.w
    c1=model.c1
    c2=model.c2
    pg = model.pg
    for id,sol in enumerate(model.sol_list):
        x=sol.nodes_seq
        v=model.v[id]
        pl=model.pl[id]
        r1=random.random()
        r2=random.random()
        new_v=[]
        for i in range(model.number_of_nodes):
            v_=w*v[i]+c1*r1*(pl[i]-x[i])+c2*r2*(pg[i]-x[i])
            if v_>0:
                new_v.append(min(v_,model.Vmax))
            else:
                new_v.append(max(v_,-model.Vmax))
        new_x=[min(int(x[i]+new_v[i]),model.number_of_nodes-1) for i in range(model.number_of_nodes) ]
        new_x=adjustRoutes(new_x,model)
        model.v[id]=new_v

        new_x_obj,new_x_routes=calObj(new_x,model)
        if new_x_obj<sol.obj:
            model.pl[id]=copy.deepcopy(new_x)
        if new_x_obj<model.best_sol.obj:
            model.best_sol.obj=copy.deepcopy(new_x_obj)
            model.best_sol.nodes_seq=copy.deepcopy(new_x)
            model.best_sol.routes=copy.deepcopy(new_x_routes)
            model.pg=copy.deepcopy(new_x)
        model.sol_list[id].nodes_seq = copy.deepcopy(new_x)
        model.sol_list[id].obj = copy.deepcopy(new_x_obj)
        model.sol_list[id].routes = copy.deepcopy(new_x_routes)

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
def run(filepath,epochs,popsize,Vmax,v_cap,opt_type,w,c1,c2):
    """
    :param filepath: Xlsx file path
    :param epochs: Iterations
    :param popsize: Population size
    :param v_cap: Vehicle capacity
    :param Vmax: Max speed
    :param opt_type: Optimization type:0:Minimize the number of vehicles,1:Minimize travel distance
    :param w: Inertia weight
    :param c1:Learning factors
    :param c2:Learning factors
    :return:
    """
    model=Model()
    model.vehicle_cap=v_cap
    model.opt_type=opt_type
    model.w=w
    model.c1=c1
    model.c2=c2
    model.Vmax = Vmax
    readXlsxFile(filepath,model)
    history_best_obj=[]
    genInitialSol(model,popsize)
    history_best_obj.append(model.best_sol.obj)
    for ep in range(epochs):
        updatePosition(model)
        history_best_obj.append(model.best_sol.obj)
        print("%s/%s: best obj: %s"%(ep,epochs,model.best_sol.obj))
    plotObj(history_best_obj)
    outPut(model)
if __name__=='__main__':
    file='../data/cvrp.xlsx'
    run(filepath=file,epochs=100,popsize=150,Vmax=2,v_cap=70,opt_type=1,w=0.9,c1=1,c2=5)




