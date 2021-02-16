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
        self.distance={}
        self.popsize=100
        self.alpha=2
        self.beta=3
        self.Q=100
        self.rho=0.5
        self.tau={}
def readXlsxFile(filepath,model):
    node_seq_no = -1
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

def initParam(model):
    for i in range(model.number_of_nodes):
        for j in range(i+1,model.number_of_nodes):
            d=math.sqrt((model.node_list[i].x_coord-model.node_list[j].x_coord)**2+
                        (model.node_list[i].y_coord-model.node_list[j].y_coord)**2)
            model.distance[i,j]=d
            model.distance[j,i]=d
            model.tau[i,j]=10
            model.tau[j,i]=10
def movePosition(model):
    sol_list=[]
    local_sol=Sol()
    local_sol.obj=float('inf')
    for k in range(model.popsize):
        #Random ant position
        nodes_seq=[int(random.randint(0,model.number_of_nodes-1))]
        all_nodes_seq=copy.deepcopy(model.node_seq_no_list)
        all_nodes_seq.remove(nodes_seq[-1])
        #Determine the next moving position according to pheromone
        while len(all_nodes_seq)>0:
            next_node_no=searchNextNode(model,nodes_seq[-1],all_nodes_seq)
            nodes_seq.append(next_node_no)
            all_nodes_seq.remove(next_node_no)
        sol=Sol()
        sol.nodes_seq=nodes_seq
        sol.obj,sol.routes=calObj(nodes_seq,model)
        sol_list.append(sol)
        if sol.obj<local_sol.obj:
            local_sol=copy.deepcopy(sol)
    model.sol_list=copy.deepcopy(sol_list)
    if local_sol.obj<model.best_sol.obj:
        model.best_sol=copy.deepcopy(local_sol)

def searchNextNode(model,current_node_no,SE_List):
    prob=np.zeros(len(SE_List))
    for i,node_no in enumerate(SE_List):
        eta=1/model.distance[current_node_no,node_no]
        tau=model.tau[current_node_no,node_no]
        prob[i]=((eta**model.alpha)*(tau**model.beta))
    #use Roulette to determine the next node
    cumsumprob=(prob/sum(prob)).cumsum()
    cumsumprob -= np.random.rand()
    next_node_no= SE_List[list(cumsumprob > 0).index(True)]
    return next_node_no
def upateTau(model):
    rho=model.rho
    for k in model.tau.keys():
        model.tau[k]=(1-rho)*model.tau[k]
    #update tau according to sol.nodes_seq(solution of TSP)
    for sol in model.sol_list:
        nodes_seq=sol.nodes_seq
        for i in range(len(nodes_seq)-1):
            from_node_no=nodes_seq[i]
            to_node_no=nodes_seq[i+1]
            model.tau[from_node_no,to_node_no]+=model.Q/sol.obj

    #update tau according to sol.routes(solution of CVRP)
    # for sol in model.sol_list:
    #     routes=sol.routes
    #     for route in routes:
    #         for i in range(len(route)-1):
    #             from_node_no=route[i]
    #             to_node_no=route[i+1]
    #             model.tau[from_node_no,to_node_no]+=model.Q/sol.obj

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
    plt.rcParams['font.sans-serif'] = ['SimHei']  #show chinese
    plt.rcParams['axes.unicode_minus'] = False   # Show minus sign
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
def run(filepath,Q,alpha,beta,rho,epochs,v_cap,opt_type,popsize):
    """
    :param filepath:Xlsx file path
    :param Q:Total pheromone
    :param alpha:Information heuristic factor
    :param beta:Expected heuristic factor
    :param rho:Information volatilization factor
    :param epochs:Iterations
    :param v_cap:Vehicle capacity
    :param opt_type:Optimization type:0:Minimize the number of vehicles,1:Minimize travel distance
    :param popsize:Population size
    :return:
    """
    model=Model()
    model.vehicle_cap=v_cap
    model.opt_type=opt_type
    model.alpha=alpha
    model.beta=beta
    model.Q=Q
    model.rho=rho
    model.popsize=popsize
    sol=Sol()
    sol.obj=float('inf')
    model.best_sol=sol
    history_best_obj = []
    readXlsxFile(filepath,model)
    initParam(model)
    for ep in range(epochs):
        movePosition(model)
        upateTau(model)
        history_best_obj.append(model.best_sol.obj)
        print("%s/%s， best obj: %s" % (ep,epochs, model.best_sol.obj))
    plotObj(history_best_obj)
    outPut(model)
if __name__=='__main__':
    file='../data/cvrp.xlsx'
    run(filepath=file,Q=10,alpha=1,beta=5,rho=0.1,epochs=100,v_cap=60,opt_type=1,popsize=60)



