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
        self.fit=None
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
        self.pc=0.5
        self.pm=0.2
        self.n_select=80
        self.popsize=100

def readXlsxFile(filepath,model):
    #It is recommended that the vehicle depot data be placed in the first line of xlsx file
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
        model.sol_list.append(sol)

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
def calFit(model):
    #calculate fit value：fit=Objmax-obj
    Objmax=-float('inf')
    best_sol=Sol()#record the local best solution
    best_sol.obj=float('inf')
    #计算目标函数
    for sol in model.sol_list:
        nodes_seq=sol.nodes_seq
        num_vehicle, vehicle_routes = splitRoutes(nodes_seq, model)
        if model.opt_type==0:
            sol.obj=num_vehicle
            sol.routes=vehicle_routes
            if sol.obj>Objmax:
                Objmax=sol.obj
            if sol.obj<best_sol.obj:
                best_sol=copy.deepcopy(sol)
        else:
            distance=0
            for route in vehicle_routes:
                distance+=calDistance(route,model)
            sol.obj=distance
            sol.routes=vehicle_routes
            if sol.obj>Objmax:
                Objmax=sol.obj
            if sol.obj < best_sol.obj:
                best_sol = copy.deepcopy(sol)
    #calculate fit value
    for sol in model.sol_list:
        sol.fit=Objmax-sol.obj
    #update the global best solution
    if best_sol.obj<model.best_sol.obj:
        model.best_sol=best_sol
    #Binary tournament
def selectSol(model):
    sol_list=copy.deepcopy(model.sol_list)
    model.sol_list=[]
    for i in range(model.n_select):
        f1_index=random.randint(0,len(sol_list)-1)
        f2_index=random.randint(0,len(sol_list)-1)
        f1_fit=sol_list[f1_index].fit
        f2_fit=sol_list[f2_index].fit
        if f1_fit<f2_fit:
            model.sol_list.append(sol_list[f2_index])
        else:
            model.sol_list.append(sol_list[f1_index])
    #Order Crossover (OX)
def crossSol(model):
    sol_list=copy.deepcopy(model.sol_list)
    model.sol_list=[]
    while True:
        if random.random()<=model.pc:
            f1_index = random.randint(0, len(sol_list) - 1)
            f2_index = random.randint(0, len(sol_list) - 1)
            if f1_index!=f2_index:
                f1 = copy.deepcopy(sol_list[f1_index])
                f2 = copy.deepcopy(sol_list[f2_index])
                cro1_index=int(random.randint(0,model.number_of_nodes-1))
                cro2_index=int(random.randint(cro1_index,model.number_of_nodes-1))
                new_c1_f = []
                new_c1_m=f1.nodes_seq[cro1_index:cro2_index+1]
                new_c1_b = []
                new_c2_f = []
                new_c2_m=f2.nodes_seq[cro1_index:cro2_index+1]
                new_c2_b = []
                for index in range(model.number_of_nodes):
                    if len(new_c1_f)<cro1_index:
                        if f2.nodes_seq[index] not in new_c1_m:
                            new_c1_f.append(f2.nodes_seq[index])
                    else:
                        if f2.nodes_seq[index] not in new_c1_m:
                            new_c1_b.append(f2.nodes_seq[index])
                for index in range(model.number_of_nodes):
                    if len(new_c2_f)<cro1_index:
                        if f1.nodes_seq[index] not in new_c2_m:
                            new_c2_f.append(f1.nodes_seq[index])
                    else:
                        if f1.nodes_seq[index] not in new_c2_m:
                            new_c2_b.append(f1.nodes_seq[index])
                new_c1=copy.deepcopy(new_c1_f)
                new_c1.extend(new_c1_m)
                new_c1.extend(new_c1_b)
                f1.nodes_seq=new_c1
                new_c2=copy.deepcopy(new_c2_f)
                new_c2.extend(new_c2_m)
                new_c2.extend(new_c2_b)
                f2.nodes_seq=new_c2
                model.sol_list.append(copy.deepcopy(f1))
                model.sol_list.append(copy.deepcopy(f2))
                if len(model.sol_list)>model.popsize:
                    break
    #mutation
def muSol(model):
    sol_list=copy.deepcopy(model.sol_list)
    model.sol_list=[]
    while True:
        if random.random()<=model.pm:
            f1_index = int(random.randint(0, len(sol_list) - 1))
            f1 = copy.deepcopy(sol_list[f1_index])
            m1_index=random.randint(0,model.number_of_nodes-1)
            m2_index=random.randint(0,model.number_of_nodes-1)
            if m1_index!=m2_index:
                node1=f1.nodes_seq[m1_index]
                f1.nodes_seq[m1_index]=f1.nodes_seq[m2_index]
                f1.nodes_seq[m2_index]=node1
                model.sol_list.append(copy.deepcopy(f1))
                if len(model.sol_list)>model.popsize:
                    break
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


def run(filepath,epochs,pc,pm,popsize,n_select,v_cap,opt_type):
    """
    :param filepath:Xlsx file path
    :param epochs:Iterations
    :param pc:Crossover probability
    :param pm:Mutation probability
    :param popsize:Population size
    :param n_select:Number of excellent individuals selected
    :param v_cap:Vehicle capacity
    :param opt_type:Optimization type:0:Minimize the number of vehicles,1:Minimize travel distance
    :return:
    """
    model=Model()
    model.vehicle_cap=v_cap
    model.opt_type=opt_type
    model.pc=pc
    model.pm=pm
    model.popsize=popsize
    model.n_select=n_select

    readXlsxFile(filepath,model)
    genInitialSol(model)
    history_best_obj = []
    best_sol=Sol()
    best_sol.obj=float('inf')
    model.best_sol=best_sol
    for ep in range(epochs):
        calFit(model)
        selectSol(model)
        crossSol(model)
        muSol(model)
        history_best_obj.append(model.best_sol.obj)
        print("%s/%s， best obj: %s" % (ep,epochs,model.best_sol.obj))
    plotObj(history_best_obj)
    outPut(model)
if __name__=='__main__':
    file='../data/cvrp.xlsx'
    run(filepath=file,epochs=150,pc=0.6,pm=0.2,popsize=100,n_select=80,v_cap=80,opt_type=1)
