import pandas as pd  
import numpy as np  


"""
Create a list of nodes from the data provided 
"""
class Node(object):
    def __init__(self, id: int, x: int, y: int, demand: int, ready_time: int, due_time: int, service_time:int):
        super()
        self.id = id
        
        # depot centre
        if id == 0:
            self.is_depot = True
        else: 
            self.is_depot = False 
        
        self.x = x
        self.y = y  
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time