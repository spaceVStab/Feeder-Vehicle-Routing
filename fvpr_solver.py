import pandas as pd  
import pdb
from tqdm import tqdm
import numpy as np
import random  
import time
from datetime import datetime
from utils import Node
"""
Structure of the solver will be as:

"""


class FVRP_ACO_Solver(object):
    def __init__(self, dataset):
        self.extractDataset(dataset)
        self.createDistanceMatrix()

    def createDistanceMatrix(self):
        self.distance_matrix = np.zeros(shape=(self.node_num, self.node_num), dtype=np.float)
        for node_a in self.nodes:
            for node_b in self.nodes:
                self.distance_matrix[node_a.id][node_b.id] = np.sqrt(pow(node_a.x-node_b.x,2) + pow(node_a.y-node_b.y,2))
        # import pdb; pdb.set_trace()
        self.distance_inverse_matrix = 1./self.distance_matrix

    def extractDataset(self, dataset):
        node_list = []
        with open(dataset, 'rt') as f:
            count = 1
            for line in f:
                if count == 5:
                    pass
                elif count >= 10:
                    node_list.append(line.split())
                count+=1
        self.node_num = len(node_list)
        print(self.node_num)
        
        # Node -> (id(including 0), x, y, demand, ready_time, due_date, service_time)
        self.nodes = []
        for item in node_list:
            self.nodes.append(Node(int(item[0]), int(item[1]), int(item[2]), int(item[3]), int(item[4]), int(item[5]), int(item[6])))

        return


    def calcObjectiveFitness(self, truck, bike, add_bike):
        # Fitness Function according to the paper is defined as :
        # Cij is the truck and bike travel cost which is equal to some constant multiplied with distance between them
        # Fc/m is the fixed cost for bike / truck
        # import pdb; pdb.set_trace()
        objective = np.sum(self.travel_cost_bike * np.sum(bike, 2)) + np.sum(self.travel_cost_truck * np.sum(truck,2)) + \
                    self.fixed_cost_bike * np.sum(add_bike[0] + np.sum(bike[0], 1)) + self.fixed_cost_truck * np.sum(truck[0])

        return objective


    def initACOParameters(self):
        """
        Parameters:
            ssii : no of sub-fleets
            A : no of ants
            I : no of iterations
            delta : no of dispatched sub-fleets
            Wc- : current truck working time 
            Wm- : current bike working time 
            solutions :
            best-solutions : 
        """
        self.total_sub_fleets = self.getNoOfSubFleets()
        self.num_of_ants = 100
        self.iterations = 50
        self.alpha = 2
        self.beta = 8
        self.gamma = 3
        self.pheromones = np.ones(shape=(self.node_num, self.node_num))*0.5
        # self.vega = np.empty(shape=(self.node_num, self.node_num))
        self.rho_1 = 0.2 
        self.rho_2 = 0.2
        self.pheromone_constant = 0.5

        return

    def initProblemParameters(self):
        """
        Parameters:
            travel_cost_{truck/bike}
            travel_time_{truck/bike}
            customer_service_time_{truck/bike}
            customer_demand
            capacity_{truck/bike}
            fixed_cost_{truck/bike}
        """

        # assuming only single truck and bike  

        # travel cost of truck b/w i and j  
        self.bike_travel_constant = 0.77
        self.truck_travel_constant = 3.3
        self.travel_cost_bike = self.distance_matrix * self.bike_travel_constant
        self.travel_cost_truck = self.distance_matrix * self.truck_travel_constant

        # travel time of truck b/w i and j  
        # self.travel_time_truck = np.zeros(shape=(self.node_num, self.node_num))
        # self.travel_time_bike = np.zeros(shape=(self.node_num, self.node_num))        

        # # customer service time of truck in node i
        # self.customer_serve_time_truck = np.zeros(shape=(self.node_num))
        # self.customer_serve_time_bike = np.zeros(shape=(self.node_num))

        self.reloading_time = 0

        # demand of ith customer
        self.customer_demand = np.zeros(shape=(self.node_num))      # includes depot also
        self.customer_service_time = np.zeros(shape=(self.node_num))    
        for node in self.nodes:
            self.customer_demand[node.id] = node.demand
            self.customer_service_time[node.id] = node.service_time
            
        # vehicle capacity 
        self.truck_capacity = 500
        self.bike_capacity = 50

        self.fixed_cost_truck = 4360
        self.fixed_cost_bike = 1720

        self.MAX_TIME = 0

        self.customer_nodes = []
        for node in self.nodes:
            if not node.is_depot:
                self.customer_nodes.append(node.id)


    def initSolverParameters(self):
        for node in self.nodes:
            if node.is_depot:
                self.DEPOT = node.id
                break       # single DEPOT

        self.ants = [i for i in range(self.num_of_ants)]
        
        return

    def getNoOfSubFleets(self):
        return np.int(np.floor(np.sum(self.customer_demand) / (self.truck_capacity + self.bike_capacity)))


    def getNextStop(self, ant_index, origin, istruck=False, c=None):
        # get the possible nodes which can be visited by this ant 
        self.possible_transition_nodes = list(set(self.customer_nodes) - set(self.nodes_visited_by_ant[ant_index]))
        # print(possible_transition_nodes)

        if self.possible_transition_nodes is None:
            raise NotImplementedError

        # create a PM (i,j) dict : {(i,j):value}
        self.probability_matrix = {}
        self.weighted_val = {}
        self.weighted_val_sum = 0
        if not istruck:
            for destination in self.possible_transition_nodes:
                self.weighted_val[(origin, destination)] = pow(self.pheromones[origin][destination], self.alpha) * pow(self.distance_inverse_matrix[origin][destination], self.beta)
                self.weighted_val_sum += self.weighted_val[(origin, destination)]

        if istruck:
            for destination in self.possible_transition_nodes:
                self.weighted_val[(origin, destination)] = pow(self.pheromones[origin][destination], self.alpha) * pow(self.distance_inverse_matrix[origin][destination], self.beta) * pow(self.distance_inverse_matrix[destination][c], self.gamma)
                self.weighted_val_sum += self.weighted_val[(origin, destination)]

        cum_val = 0
        self.cum_prob_matrix = {}
        for destination in self.possible_transition_nodes:
            self.probability_matrix[(origin, destination)] = self.weighted_val[(origin, destination)] / self.weighted_val_sum
            cum_val += self.probability_matrix[(origin, destination)]
            self.cum_prob_matrix[(origin, destination)] = cum_val
        
        destination_random_val = np.random.rand(len(self.possible_transition_nodes))
        next_node = None
        for i, destination in enumerate(self.possible_transition_nodes):
            if self.cum_prob_matrix[(origin, destination)] > destination_random_val[i]:
                next_node = destination
                break

        return next_node


    def checkFeasibility1(self, origin, destination, ant_index, freight):
        # POINTS TO BE CONSIDERED FOR USUSAL BIKE DELIVERY
        # 1. Bike should have quantity more or equal to destination demand and less than Bike capacity
        # 2. Already visited demand nodes should not be visited  
        # 3. After delivery, make the customer demand as demand fulfiled 

        if freight < self.customer_demand[destination]:
            return False

        if destination in self.nodes_visited_by_ant[ant_index]:
            return False 

        if destination is self.DEPOT:
            return False 

        return True

    def printStatus(self, ant, bikeF, truckF):
        print("Nodes Visited : ", len(self.nodes_visited_by_ant[ant]))
        print("Freight Bike : ",bikeF," Truck : ", truckF)



    def solveForNFleet(self, total_sub_fleets, ant):
        # Decision Variables
        # import pdb; pdb.set_trace()
        assert isinstance(total_sub_fleets, int)
        truck_i_j_DV = np.zeros(shape=(self.node_num, self.node_num, total_sub_fleets), dtype=np.int)
        bike_i_j_DV = np.zeros(shape=(self.node_num, self.node_num, total_sub_fleets), dtype=np.int)

        # truck_decision = {}     # dict of [origin,destination]
        truck_decision = []
        bike_decision = []

        # to save
        self.nodes_visited_by_ant[ant] = list()
        self.reloading_points[ant] = list()

        originB = 0      # bike it is the depot 
        originT = 0      # truck is at the depot
        self.demand_satisfied  = np.zeros(shape=self.node_num, dtype=np.int)
        self.demand_satisfied[self.DEPOT] = 1
        # Calculate rho, delta=1, Wc=0, Wm=0, sol and best_sol
        
        # considering while dispatch from depot fully loaded
        bike_freight = self.bike_capacity
        truck_freight = self.truck_capacity
        
        bike_working_time = 0
        truck_working_time = 0

        current_sub_fleet = 0

        bike_curr_fleet = []    # visiting nodes
        truck_curr_fleet = []   # visiting nodes
        bike_curr_fleet_material = []   # bike total carrying load current
        truck_curr_fleet_material = []  # truck total carrying load current
        
        bike_curr_fleet.append(self.DEPOT)
        truck_curr_fleet.append(self.DEPOT)
        bike_curr_fleet_material.append(bike_freight)
        truck_curr_fleet_material.append(truck_freight)


        while current_sub_fleet < total_sub_fleets:
            # import pdb; pdb.set_trace()
            # print("Fleet : {}".format(current_sub_fleet))
            # min_demand = np.min(self.customer_demand.take(list(set(self.customer_nodes) - set(self.nodes_visited_by_ant[ant]) - set([self.DEPOT]))))
            count = 0

            destinationB = self.getNextStop(ant, originB)
            while destinationB!=None and bike_freight >= self.customer_demand[destinationB]:
                count+=1
                if count > self.node_num:
                    import pdb; pdb.set_trace()
                # print("bike, destin : ",destinationB)
                if self.checkFeasibility1(originB, destinationB, ant, bike_freight):
                    bike_i_j_DV[originB][destinationB][current_sub_fleet] = 1
                    bike_curr_fleet.append(destinationB)
                    self.nodes_visited_by_ant[ant].append(destinationB)
                    bike_freight -= self.customer_demand[destinationB]
                    bike_curr_fleet_material.append(bike_freight)
                    self.demand_satisfied[destinationB] = 1
                    originB = destinationB
                    bike_working_time += self.customer_service_time[destinationB]
                    # min_demand = np.min(self.customer_demand.take(list(set(self.customer_nodes) - set(self.nodes_visited_by_ant[ant]) - set([self.DEPOT]))))
                else:
                    destinationB = self.getNextStop(ant, originB)

            bike_last_halt = originB
            # print("Bike Completed at {}".format(originB))
            
            # now dispatch the truck
            # min_demand = np.min(self.customer_demand.take(list(set(self.customer_nodes) - set(self.nodes_visited_by_ant[ant]) - set([self.DEPOT]))))
            count = 0
            destinationT = self.getNextStop(ant, originT, istruck=True, c=bike_last_halt)
            while destinationT!=None and truck_freight>=self.customer_demand[destinationT]:
                count+=1
                if count > self.node_num:
                    import pdb; pdb.set_trace()
                if bike_working_time <= truck_working_time:
                    break 
                else:
                    # print("truck, destin ",destinationT)
                    if self.checkFeasibility1(originT, destinationT, ant, truck_freight):
                        truck_i_j_DV[originT][destinationT][current_sub_fleet] = 1
                        truck_curr_fleet.append(destinationT)
                        self.nodes_visited_by_ant[ant].append(destinationT)
                        truck_freight -= self.customer_demand[destinationT]
                        truck_curr_fleet_material.append(truck_freight)
                        self.demand_satisfied[destinationT] = 1
                        originT = destinationT
                        truck_working_time += self.customer_service_time[destinationT]
                        # min_demand = np.min(self.customer_demand.take(list(set(self.customer_nodes) - set(self.nodes_visited_by_ant[ant]) - set([self.DEPOT]))))
                    else:
                        destinationT = self.getNextStop(ant, originT, istruck=True, c=bike_last_halt)

            # print("Truck Completed at {}".format(originT))
            # now the both truck and bike should meet at a common point 
            # we will move according to truck and make now next destination as a joint for bike
            # check if all customers are satisfied before freight is exhausted
            # atleast total_fleet has to be completed for global demand fulfilment before that CHILL !!

            # is the refill necessary 
            # refill done only 
            if truck_freight >= self.bike_capacity and len(self.nodes_visited_by_ant[ant])<len(self.customer_nodes):
                count = 0
                while(1):
                    # print("joint")
                    count+=1
                    if count > self.node_num:
                        import pdb; pdb.set_trace()  
                    destinationT = self.getNextStop(ant, originT, istruck=True, c=bike_last_halt)
                    if destinationT is None:
                        continue
                    if self.checkFeasibility1(originT, destinationT, ant, truck_freight):
                        #DV
                        truck_i_j_DV[originT][destinationT][current_sub_fleet] = 1
                        bike_i_j_DV[bike_last_halt][destinationT][current_sub_fleet] = 1
                        truck_curr_fleet.append(destinationT)
                        bike_curr_fleet.append(destinationT)
                        # self.reload_i_DV[destinationT][current_sub_fleet] = 1
                        
                        # do the necessary changes
                        self.nodes_visited_by_ant[ant].append(destinationT)
                        truck_freight -= self.customer_demand[destinationT]
                        self.demand_satisfied[destinationT] = 1
                        originB = destinationT
                        originT = destinationT
                        
                        # append the reloading points for this ant
                        self.reloading_points[ant].append(destinationT)

                        # make the serving time equal 
                        truck_working_time = bike_working_time

                        # reload the bike 
                        reload_amount = self.bike_capacity - bike_freight
                        bike_freight = self.bike_capacity
                        truck_freight -= reload_amount

                        truck_curr_fleet_material.append(truck_freight)
                        bike_curr_fleet_material.append(bike_freight)

                        break 

            else:
                # both bike and truck freight is less than the minimum required demand thus, return back ot depot
                bike_i_j_DV[originB][self.DEPOT][current_sub_fleet]=1
                truck_i_j_DV[originT][self.DEPOT][current_sub_fleet]=1
                destinationT = self.DEPOT
                destinationB = self.DEPOT
                originB = self.DEPOT
                originT = self.DEPOT
                bike_freight = self.bike_capacity
                truck_freight = self.truck_capacity

                truck_curr_fleet.append(self.DEPOT)
                bike_curr_fleet.append(self.DEPOT)

                truck_decision.append({
                    'fleet_no':current_sub_fleet, 
                    'truck_routes':truck_curr_fleet, 
                    'truck_freight':truck_curr_fleet_material})
                bike_decision.append({
                    'fleet_no':current_sub_fleet, 
                    'bike_routes':bike_curr_fleet, 
                    'bike_freight':bike_curr_fleet_material})

                bike_curr_fleet = []    # visiting nodes
                truck_curr_fleet = []   # visiting nodes
                bike_curr_fleet_material = []   # bike total carrying load current
                truck_curr_fleet_material = []  # truck total carrying load current
                
                bike_curr_fleet.append(self.DEPOT)
                truck_curr_fleet.append(self.DEPOT)
                bike_curr_fleet_material.append(bike_freight)
                truck_curr_fleet_material.append(truck_freight)


                # continue the same for total number of sub-fleets calculated
                current_sub_fleet += 1
        # import pdb; pdb.set_trace()

        return truck_i_j_DV, bike_i_j_DV, truck_decision, bike_decision

    def globalPheromoneUpdate(self):
        for i in self.customer_nodes:
            for j in self.customer_nodes:
                delta_val = 0
                for ant in self.ants:
                    delta_val += (1 / max(0.001, abs(self.best_solution['objective_value'] - self.solutions_per_ant[ant]['objective_value'])))
                self.pheromones[i][j] = (1 - self.rho_1) * self.pheromones[i][j] + self.rho_1 * delta_val
        
        return

    def localPheremoneUpdate(self):
        # perform local phermone updates
        # local pheromone updates are : Tij(t) = (1 - rho_2) * Tij(t) + rho_2 * T_o
        # T_o is a small positive constant so that the pheromones dont go zero
        for i in self.customer_nodes:
            for j in self.customer_nodes:
                self.pheromones[i][j] = (1-self.rho_2) * self.pheromones[i][j] + self.rho_2 * self.pheromone_constant

        return

    # def printSolution(self):
    #     bike = self.best_solution['bike_route']
    #     truck = self.best_solution['truck_route']
    #     add_bike = self.best_solution['additional_bike_route']
    #     fleet = bike.shape[2]

    #     for f in range(fleet):
    #         print("Fleet no : {}".format(f+1))

    #     return


    def saveSolution(self, ant, truck, bike, add_bike, truck_simple, bike_simple, add_bike_simple):
        # fine save the solution and move ahead 
        objective_value = self.calcObjectiveFitness(truck, bike, add_bike)
        solution_dict = {
            'objective_value' : objective_value,
            'truck_route' : truck,
            'bike_route' : bike,
            'additional_bike_route' : add_bike,
            'truck_decision': truck_simple,
            'bike_decision': bike_simple,
            'add_bike_decision': add_bike_simple
        }
        self.solutions_per_ant[ant] = solution_dict
        if self.best_solution is None:
            self.best_solution = solution_dict
        else:
            # if best solution  > current solution : best solution = current solution
            if self.best_solution['objective_value'] > objective_value:
                self.best_solution = solution_dict

        return


    def solver(self):
        """
        Follow the flow diagram as depicted in the given paper
        """

        # define the problem parameters  
        self.initProblemParameters()

        # define the ACO parameters
        self.initACOParameters()

        self.initSolverParameters()

        self.best_solution = None
        object_val_per_iter = []
        start_time = datetime.now()
        for iter in tqdm(range(self.iterations)):
        # for every iterations defined 

            self.nodes_visited_by_ant = {}
            self.reloading_points = {}
            self.solutions_per_ant = {}

            for ant in self.ants:           # define the type of ant here / most probably is a range(NO_OF_ANTS)
            # for every ants taken (cumulative most probably)
                # print("Iteration {} Ant {}".format(iter,ant))
                sub_fleets = self.getNoOfSubFleets()

                while(True):
                    # print("start")
                    truck_i_j_DV = None
                    bike_i_j_DV = None
                    truck_decision = None
                    bike_decision = None
                    add_bike_i_j_DV = np.zeros(shape=(self.node_num, self.node_num), dtype=np.int)
                    add_bike_decision = {}
                    add_bike_decision['add_bike_routes']=None
                    add_bike_decision['add_bike_freight']=None
                    add_bike_curr = [self.DEPOT]
                    add_bike_curr_freight = []
                    truck_i_j_DV, bike_i_j_DV, truck_decision, bike_decision = self.solveForNFleet(sub_fleets, ant)
                    if np.sum(self.demand_satisfied) == self.node_num:
                        break
                    else:
                        demand_left = np.sum(self.customer_demand.take(np.where(self.demand_satisfied)))
                        if demand_left <= self.bike_capacity:
                            # bring in additional bike
                            originAdd = self.DEPOT
                            bike_freight = min(demand_left, self.bike_capacity)
                            add_bike_curr_freight.append(bike_freight)
                            while(bike_freight>0):
                                # print("add_bike")
                                destinationAdd = self.getNextStop(ant, originAdd)
                                # see if other constraints persists
                                if self.checkFeasibility1(originAdd, destinationAdd, ant, bike_freight):
                                    add_bike_i_j_DV[originAdd][destinationAdd] = 1
                                    add_bike_curr.append(destinationAdd)
                                    self.nodes_visited_by_ant[ant].append(destinationAdd)
                                    bike_freight -= self.customer_demand[destinationAdd]
                                    add_bike_curr_freight.append(bike_freight)
                                    self.demand_satisfied[destinationAdd] = 1
                                    originAdd = destinationAdd
                                    # bike_working_time += self.customer_service_time[destinationB]
                            destinationAdd = self.DEPOT
                            add_bike_i_j_DV[originAdd][destinationAdd] = 1
                            add_bike_curr.append(self.DEPOT)
                            add_bike_decision['add_bike_routes'] = add_bike_curr
                            add_bike_decision['add_bike_freight'] = add_bike_curr_freight

                        else:
                            sub_fleets += 1

                self.saveSolution(ant, truck_i_j_DV, bike_i_j_DV, add_bike_i_j_DV, truck_decision, bike_decision, add_bike_decision)
                self.localPheremoneUpdate()

            # perform global pheromone updates 
            # after the iteration is over perform the global pheromone update
            # Tij(t+1) = (1-rho_1) * Tij(t) + rho_1 * delta(Tij)(t)
            # delta(Tij)(t) = 1 / func(global - last_ob_val(t))
            print(self.best_solution['objective_value'])
            object_val_per_iter.append(str(self.best_solution['objective_value']))
            self.globalPheromoneUpdate()
            # import pdb; pdb.set_trace()

        # print the arrays into a file
        print("\n")
        print("*"*20)
        print("Time Taken : {}".format(datetime.now() - start_time))

        print("Truck Routes : ")
        for trucks in self.best_solution['truck_decision']:
            print(trucks['fleet_no'])
            print(trucks['truck_routes'])
            print(trucks['truck_freight'])

        print("\n")
        print("*"*20)
        print("Bike Routes : ")
        for bike in self.best_solution['bike_decision']:
            print(bike['fleet_no'])
            print(bike['bike_routes'])
            print(bike['bike_freight'])            

        print("\n")
        print("*"*20)
        print("Add Bike Routes : ")
        print(self.best_solution['add_bike_decision']['add_bike_routes'])
        print(self.best_solution['add_bike_decision']['add_bike_freight'])

        # with open("obj_val_per_iter.txt", "w") as fp:
        #     fp.write("\n".join(object_val_per_iter))

        # import pdb; pdb.set_trace()
        # with open("results_truck.txt","w") as fp:
        #     for slice2d in self.best_solution['truck_route']:
        #         np.savetxt(fp, slice2d, fmt='%5d')
        #         fp.write("\n")


        # with open("results_bike.txt","w") as fp:
        #     for slice2d in self.best_solution['bike_route']:
        #         np.savetxt(fp, slice2d, fmt='%5d')
        #         fp.write("\n")


        # with open("results_addbike.txt") as fp:
        #     np.savetxt(fp, self.best_solution['additional_bike_route'], fmt='%5d')
        
        # self.printSolution()



        return