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
        self.action_id=None
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
        self.node_seq_no_list=[]
        self.depot=None
        self.number_of_nodes=0
        self.tabu_list=None
        self.TL=30
        self.opt_type=0
        self.vehicle_cap=0

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
def genInitialSol(node_seq):
    node_seq=copy.deepcopy(node_seq)
    random.seed(0)
    random.shuffle(node_seq)
    return node_seq
def createActions(n):
    action_list=[]
    nswap=n//2
    #Single point exchange
    for i in range(nswap):
        action_list.append([1,i,i+nswap])
    #Two point exchange
    for i in range(0,nswap,2):
        action_list.append([2,i,i+nswap])
    #Reverse sequence
    for i in range(0,n,4):
        action_list.append([3,i,i+3])
    return action_list
def doACtion(nodes_seq,action):
    nodes_seq=copy.deepcopy(nodes_seq)
    if action[0]==1:
        index_1=action[1]
        index_2=action[2]
        temporary=nodes_seq[index_1]
        nodes_seq[index_1]=nodes_seq[index_2]
        nodes_seq[index_2]=temporary
        return nodes_seq
    elif action[0]==2:
        index_1 = action[1]
        index_2 = action[2]
        temporary=[nodes_seq[index_1],nodes_seq[index_1+1]]
        nodes_seq[index_1]=nodes_seq[index_2]
        nodes_seq[index_1+1]=nodes_seq[index_2+1]
        nodes_seq[index_2]=temporary[0]
        nodes_seq[index_2+1]=temporary[1]
        return nodes_seq
    elif action[0]==3:
        index_1=action[1]
        index_2=action[2]
        nodes_seq[index_1:index_2+1]=list(reversed(nodes_seq[index_1:index_2+1]))
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
    # calculate obj value
    num_vehicle, vehicle_routes = splitRoutes(nodes_seq, model)
    if model.opt_type==0:
        return num_vehicle,vehicle_routes
    else:
        distance=0
        for route in vehicle_routes:
            distance+=calDistance(route,model)
        return distance,vehicle_routes
def plotObj(obj_list):
    plt.rcParams['font.sans-serif'] = ['SimHei']  #show chinese
    plt.rcParams['axes.unicode_minus'] = False  # Show minus sign
    plt.plot(np.arange(1,len(obj_list)+1),obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Obj Value')
    plt.grid()
    plt.xlim(1,len(obj_list)+1)
    plt.show()
def outPut(model):
    work = xlsxwriter.Workbook('result.xlsx')
    worksheet = work.add_worksheet()
    worksheet.write(0, 0, 'opt_type')
    worksheet.write(1, 0, 'obj')
    if model.opt_type == 0:
        worksheet.write(0, 1, 'number of vehicles')
    else:
        worksheet.write(0, 1, 'drive distance of vehicles')
    worksheet.write(1, 1, model.best_sol.obj)
    for row, route in enumerate(model.best_sol.routes):
        worksheet.write(row + 2, 0, 'v' + str(row + 1))
        r = [str(i) for i in route]
        worksheet.write(row + 2, 1, '-'.join(r))
    work.close()
def run(filepath,epochs,v_cap,opt_type):
    """
    :param filepath: :Xlsx file path
    :param epochs: Iterations
    :param v_cap: Vehicle capacity
    :param opt_type:Optimization type:0:Minimize the number of vehicles,1:Minimize travel distance
    :return: 无
    """
    model=Model()
    model.vehicle_cap=v_cap
    model.opt_type=opt_type
    readXlsxFile(filepath,model)
    action_list=createActions(model.number_of_nodes)
    model.tabu_list=np.zeros(len(action_list))
    history_best_obj=[]
    sol=Sol()
    sol.nodes_seq=genInitialSol(model.node_seq_no_list)
    sol.obj,sol.routes=calObj(sol.nodes_seq,model)
    model.best_sol=copy.deepcopy(sol)
    history_best_obj.append(sol.obj)
    for ep in range(epochs):
        local_new_sol=Sol()
        local_new_sol.obj=float('inf')
        for i in range(len(action_list)):
            if model.tabu_list[i]==0:
                new_sol=Sol()
                new_sol.nodes_seq=doACtion(sol.nodes_seq,action_list[i])
                new_sol.obj,new_sol.routes=calObj(new_sol.nodes_seq,model)
                new_sol.action_id=i
                if new_sol.obj<local_new_sol.obj:
                    local_new_sol=copy.deepcopy(new_sol)
        sol=local_new_sol
        for i in range(len(action_list)):
            if i==sol.action_id:
                model.tabu_list[sol.action_id]=model.TL
            else:
                model.tabu_list[i]=max(model.tabu_list[i]-1,0)
        if sol.obj<model.best_sol.obj:
            model.best_sol=copy.deepcopy(sol)
        history_best_obj.append(model.best_sol.obj)
        print("%s/%s: best obj: %s"%(ep,epochs,model.best_sol.obj))
    plotObj(history_best_obj)
    outPut(model)

if __name__=='__main__':
    file='../data/cvrp.xlsx'
    run(file,100,80,1)