# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 09:35:36 2018
@author: Shiva
"""

import numpy as np
from pulp import *
import time
import Zmatrixa as zma
import Zmatrixb as zmb
import Zmatrixc as zmc
import json 
start = time.clock()
import matplotlib.pyplot as plt
import numpy as np




class Restoration_Sequence(object):
    """
    WSU Resilient Restoration Sequence of Switches
    """
   
    def __init__(self):
        pass

    def sequence(self):

        # Parameters
        nNodes=1317
        nEdges=1327
        f = open('load_shed.txt', 'rb')
        obj = json.load(f)
        f.close()
        edges = np.loadtxt('edges.txt') 
        demand = np.loadtxt('LoadData.txt')
        loops = np.loadtxt('cycles.txt') 
        LineData = np.loadtxt('Line_Config.txt')
        sw_ind = np.loadtxt('sw_ind.txt')
        con_ind = np.loadtxt('config.txt')
        M = 3000
        T = 7

        # Different variables for optimization function
        vi = LpVariable.dicts("v_i", ((i, t) for i in range(nNodes) for t in range(T)), lowBound=0, upBound=1, cat='Binary')
        si = LpVariable.dicts("s_i", ((i, t) for i in range(nNodes) for t in range(T)), lowBound=0, upBound=1, cat='Binary')
        xij = LpVariable.dicts("xl", ((i, t) for i in range(nEdges) for t in range(T)), lowBound=0, upBound=1, cat='Binary')
        xij0 = LpVariable.dicts("xl0", ((i, t) for i in range(nEdges) for t in range(T)), lowBound=0, upBound=1, cat='Binary')
        xij1 = LpVariable.dicts("xl1", ((i, t) for i in range(nEdges) for t in range(T)), lowBound=0, upBound=1, cat='Binary')
        Pija = LpVariable.dicts("xPa", ((i, t) for i in range(nEdges) for t in range(T)), lowBound=-M, upBound=M, cat='Continous')
        Pijb = LpVariable.dicts("xPb", ((i, t) for i in range(nEdges) for t in range(T)), lowBound=-M, upBound=M, cat='Continous')
        Pijc = LpVariable.dicts("xPc", ((i, t) for i in range(nEdges) for t in range(T)), lowBound=-M, upBound=M, cat='Continous')
        Qija = LpVariable.dicts("xQa", ((i, t) for i in range(nEdges) for t in range(T)), lowBound=-M, upBound=M, cat='Continous')
        Qijb = LpVariable.dicts("xQb", ((i, t) for i in range(nEdges) for t in range(T)), lowBound=-M, upBound=M, cat='Continous')
        Qijc = LpVariable.dicts("xQc", ((i, t) for i in range(nEdges) for t in range(T)), lowBound=-M, upBound=M, cat='Continous')
        Via = LpVariable.dicts("xVa", ((i, t) for i in range(nEdges) for t in range(T)), lowBound=0.9025, upBound=1.1, cat='Continous')
        Vib = LpVariable.dicts("xVb", ((i, t) for i in range(nEdges) for t in range(T)), lowBound=0.9025, upBound=1.1, cat='Continous')
        Vic = LpVariable.dicts("xVc", ((i, t) for i in range(nEdges) for t in range(T)), lowBound=0.9025, upBound=1.1, cat='Continous')

        # Indices for tie switches and virtual switches to insert into objetive functions
        N=[1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326]

        # Optimization problem objective definitions
        prob = LpProblem("Resilient Restoration",LpMinimize)
        # Maximize the power flow from feeder at each restoration sequence
        Fed = [266, 595, 924, 1253]
        print(Fed.__len__())
        prob += -lpSum(Pija[Fed[k],t]  for k in range(Fed.__len__()) for t in range(T)) -\
                 lpSum(Pijb[Fed[k],t]  for k in range(Fed.__len__()) for t in range(T)) -\
                 lpSum(Pijc[Fed[k],t]  for k in range(Fed.__len__()) for t in range(T))

        # Constraints (v_i<=1)
        for t in range (T):
            for k in range(nNodes):
                prob += vi[k,t] <= 1

        # Constraints (s_i<=v_i)
        for t in range(T):
            for k in range(nNodes):
                prob += si[k,t] <= vi[k,t]
            
        # Constraints (x_ij<=v_i*v_j). This is non-linear and is linearized here
        for t in range(T):
            for k in range(nEdges):                
                prob += xij[k,t] <= vi[edges[k,0]-1,t]
                prob += xij[k,t] <= vi[edges[k,1]-1,t]
            
        # Constraints (x_ij0+x_ij1<=x_ij)
        for t in range(T):
            for k in range(nEdges):
                prob += xij0[k,t] + xij1[k,t] <= xij[k,t]
            
        # Real power flow equation for Phase A phase B and Phase C
        fr = edges[:,0]
        to = edges[:,1]
        for t in range(T):   
            for k in range(nEdges): 
                ed = int(edges[k,1]-1)
                node = edges[k,1]
                pa = np.array(np.where(to==edges[k,1]))
                pa = pa.flatten()
                N = range(0,pa.__len__())
                ch = np.array(np.where(fr==edges[k,1]))
                ch = ch.flatten()
                M = range(0,ch.__len__())
                prob += lpSum(Pija[pa[j],t] for j in N) - demand[ed,1] * si[node-1,t]== \
                lpSum(Pija[ch[j],t] for j in M)  

        for t in range(T):   
            for k in range(nEdges):    
                ed = int(edges[k,1]-1)
                node = edges[k,1]
                pa = np.array(np.where(to==edges[k,1]))
                pa = pa.flatten()
                N = range(0,pa.__len__())
                ch = np.array(np.where(fr==edges[k,1]))
                ch = ch.flatten()
                M = range(0,ch.__len__())   
                prob += lpSum(Pijb[pa[j],t] for j in N)-demand[ed,3] * si[node-1,t]== \
                lpSum(Pijb[ch[j],t] for j in M)

        for t in range(T):   
            for k in range(nEdges):    
                ed = int(edges[k,1]-1)
                node = edges[k,1]
                pa = np.array(np.where(to==edges[k,1]))
                pa = pa.flatten()
                N = range(0, pa.__len__())
                ch = np.array(np.where(fr==edges[k,1]))
                ch = ch.flatten()
                M = range(0, ch.__len__())   
                prob += lpSum(Pijc[pa[j],t] for j in N)-demand[ed,5] * si[node-1,t]== \
                lpSum(Pijc[ch[j],t] for j in M)

        # Imposing the big-M method to ensure the real-power flowing in open line is zero
        M = 3000
        R = sw_ind.__len__()
        for t in range(T):
            for k in range(0, R):    
                prob += Pija[sw_ind[k],t] <= M * xij1[sw_ind[k],t]
                prob += Pijb[sw_ind[k],t] <= M * xij1[sw_ind[k],t] 
                prob += Pijc[sw_ind[k],t] <= M * xij1[sw_ind[k],t]     
                prob += Pija[sw_ind[k],t] >= -M * xij0[sw_ind[k],t]
                prob += Pijb[sw_ind[k],t] >= -M * xij0[sw_ind[k],t] 
                prob += Pijc[sw_ind[k],t] >= -M * xij0[sw_ind[k],t] 

        # Reactive power flow equation for Phase A phase B and Phase C
        fr = edges[:,0]
        to = edges[:,1]
        for t in range(T):
            for k in range(0, nEdges):    
                ed = int(edges[k,1]-1)
                node = edges[k,1]
                pa = np.array(np.where(to==edges[k,1]))
                pa = pa.flatten()
                N = range(0, pa.__len__())
                ch = np.array(np.where(fr==edges[k,1]))
                ch = ch.flatten()
                M = range(0, ch.__len__()) 
                prob += lpSum(Qija[pa[j],t] for j in N)-demand[ed,2] * si[node-1,t]== \
                lpSum(Qija[ch[j],t] for j in M)

        for t in range(T):
            for k in range(0, nEdges):    
                ed = int(edges[k,1]-1)
                node = edges[k,1]
                pa = np.array(np.where(to==edges[k,1]))
                pa = pa.flatten()
                N = range(0, pa.__len__())
                ch = np.array(np.where(fr==edges[k,1]))
                ch = ch.flatten()
                M = range(0, ch.__len__())      
                prob += lpSum(Qijb[pa[j],t] for j in N)-demand[ed,4] * si[node-1,t]== \
                lpSum(Qijb[ch[j],t] for j in M)
        
        for t in range(T):
            for k in range(0, nEdges):    
                ed = int(edges[k,1]-1)
                node = edges[k,1]
                pa = np.array(np.where(to==edges[k,1]))
                pa = pa.flatten()
                N = range(0, pa.__len__()) 
                ch = np.array(np.where(fr==edges[k,1]))
                ch = ch.flatten()
                M = range(0, ch.__len__())
                prob += lpSum(Qijc[pa[j],t] for j in N)-demand[ed,6] * si[node-1,t]== \
                lpSum(Qijc[ch[j],t] for j in M)

        # Imposing the big-M method to ensure the reactive-power flowing in open line is zero
        M=3000
        R=sw_ind.__len__()
        for t in range(T):
            for k in range(0, R):    
                prob += Qija[sw_ind[k],t] <= M * xij1[sw_ind[k],t]
                prob += Qijb[sw_ind[k],t] <= M * xij1[sw_ind[k],t] 
                prob += Qijc[sw_ind[k],t] <= M * xij1[sw_ind[k],t]     
                prob += Qija[sw_ind[k],t] >= -M * xij0[sw_ind[k],t]
                prob += Qijb[sw_ind[k],t] >= -M * xij0[sw_ind[k],t] 
                prob += Qijc[sw_ind[k],t] >= -M * xij0[sw_ind[k],t] 

        # # Voltage constraints using big-M method
        base_Z = 12.47**2/3
        M = 4
        unit = 5280
        # Phase A
        for t in range(T):
            for k in range(0, nEdges):
                len = LineData[int(con_ind[k]),2]
                conf = LineData[int(con_ind[k]),4]
                r_aa,x_aa,r_ab,x_ab,r_ac,x_ac = zma.Zmatrixa(conf)
                line = [edges[k,0], edges[k,1]]
                if (k) in sw_ind:
                    prob += Via[int(line[0])-1,t]-Via[int(line[1])-1,t] - \
                    2*r_aa*len/(unit*base_Z*1000)*Pija[k,t]- \
                    2*x_aa*len/(unit*base_Z*1000)*Qija[k,t]+ \
                    (r_ab+np.sqrt(3)*x_ab)*len/(unit*base_Z*1000)*Pijb[k,t] +\
                    (x_ab-np.sqrt(3)*r_ab)*len/(unit*base_Z*1000)*Qijb[k,t] +\
                    (r_ac-np.sqrt(3)*x_ac)*len/(unit*base_Z*1000)*Pijc[k,t] +\
                    (x_ac+np.sqrt(3)*r_ac)*len/(unit*base_Z*1000)*Qijc[k,t]-M*(1-xij[k,t]) <= 0
                    # Another inequality        
                    prob += Via[int(line[0])-1,t]-Via[int(line[1])-1,t] - \
                    2*r_aa*len/(unit*base_Z*1000)*Pija[k,t]- \
                    2*x_aa*len/(unit*base_Z*1000)*Qija[k,t]+ \
                    (r_ab+np.sqrt(3)*x_ab)*len/(unit*base_Z*1000)*Pijb[k,t] +\
                    (x_ab-np.sqrt(3)*r_ab)*len/(unit*base_Z*1000)*Qijb[k,t] +\
                    (r_ac-np.sqrt(3)*x_ac)*len/(unit*base_Z*1000)*Pijc[k,t] +\
                    (x_ac+np.sqrt(3)*r_ac)*len/(unit*base_Z*1000)*Qijc[k,t]+M*(1-xij[k,t]) >= 0
                else: 
                    prob += Via[int(line[0])-1,t]-Via[int(line[1])-1,t] - \
                    2*r_aa*len/(unit*base_Z*1000)*Pija[k,t]- \
                    2*x_aa*len/(unit*base_Z*1000)*Qija[k,t]+ \
                    (r_ab+np.sqrt(3)*x_ab)*len/(unit*base_Z*1000)*Pijb[k,t] +\
                    (x_ab-np.sqrt(3)*r_ab)*len/(unit*base_Z*1000)*Qijb[k,t] +\
                    (r_ac-np.sqrt(3)*x_ac)*len/(unit*base_Z*1000)*Pijc[k,t] +\
                    (x_ac+np.sqrt(3)*r_ac)*len/(unit*base_Z*1000)*Qijc[k,t] == 0

        # # Phase B
        # for k in range(0, nEdges):
        #     len = LineData[int(con_ind[k]),2]
        #     conf = LineData[int(con_ind[k]),4]
        #     line = [edges[k,0], edges[k,1]]
        #     r_bb,x_bb,r_ba,x_ba,r_bc,x_bc = zmb.Zmatrixb(conf)
        #     if (k) in sw_ind:
        #         prob += Vib[int(line[0])-1]-Vib[int(line[1])-1] - \
        #         2*r_bb*len/(unit*base_Z*1000)*Pijb[k]- \
        #         2*x_bb*len/(unit*base_Z*1000)*Qijb[k]+ \
        #         (r_ba-np.sqrt(3)*x_ba)*len/(unit*base_Z*1000)*Pija[k] +\
        #         (x_ba+np.sqrt(3)*r_ba)*len/(unit*base_Z*1000)*Qija[k] +\
        #         (r_bc+np.sqrt(3)*x_bc)*len/(unit*base_Z*1000)*Pijc[k] +\
        #         (x_bc-np.sqrt(3)*r_bc)*len/(unit*base_Z*1000)*Qijc[k] -M*(1-xij[k]) <=0
        #         # Another inequality
        #         prob += Vib[int(line[0])-1]-Vib[int(line[1])-1] - \
        #         2*r_bb*len/(unit*base_Z*1000)*Pijb[k]- \
        #         2*x_bb*len/(unit*base_Z*1000)*Qijb[k]+ \
        #         (r_ba-np.sqrt(3)*x_ba)*len/(unit*base_Z*1000)*Pija[k] +\
        #         (x_ba+np.sqrt(3)*r_ba)*len/(unit*base_Z*1000)*Qija[k] +\
        #         (r_bc+np.sqrt(3)*x_bc)*len/(unit*base_Z*1000)*Pijc[k] +\
        #         (x_bc-np.sqrt(3)*r_bc)*len/(unit*base_Z*1000)*Qijc[k] +M*(1-xij[k]) >=0
        #     else:
        #         prob += Vib[int(line[0])-1]-Vib[int(line[1])-1] - \
        #         2*r_bb*len/(unit*base_Z*1000)*Pijb[k]- \
        #         2*x_bb*len/(unit*base_Z*1000)*Qijb[k]+ \
        #         (r_ba-np.sqrt(3)*x_ba)*len/(unit*base_Z*1000)*Pija[k] +\
        #         (x_ba+np.sqrt(3)*r_ba)*len/(unit*base_Z*1000)*Qija[k] +\
        #         (r_bc+np.sqrt(3)*x_bc)*len/(unit*base_Z*1000)*Pijc[k] +\
        #         (x_bc-np.sqrt(3)*r_bc)*len/(unit*base_Z*1000)*Qijc[k] ==0

        # # Phase C  
        # for k in range(0, nEdges):
        #     len = LineData[int(con_ind[k]),2]
        #     conf = LineData[int(con_ind[k]),4]
        #     r_cc,x_cc,r_ca,x_ca,r_cb,x_cb = zmc.Zmatrixc(conf)
        #     line = [edges[k,0], edges[k,1]]
        #     if (k) in sw_ind: 
        #         prob += Vic[int(line[0])-1]-Vic[int(line[1])-1] - \
        #         2*r_cc*len/(unit*base_Z*1000)*Pijc[k]- \
        #         2*x_cc*len/(unit*base_Z*1000)*Qijc[k]+ \
        #         (r_ca+np.sqrt(3)*x_ca)*len/(unit*base_Z*1000)*Pija[k] +\
        #         (x_ca-np.sqrt(3)*r_ca)*len/(unit*base_Z*1000)*Qija[k] +\
        #         (r_cb-np.sqrt(3)*x_cb)*len/(unit*base_Z*1000)*Pijb[k] +\
        #         (x_cb+np.sqrt(3)*r_cb)*len/(unit*base_Z*1000)*Qijb[k] -M*(1-xij[k]) <=0
        #         # Another inequality
        #         prob += Vic[int(line[0])-1]-Vic[int(line[1])-1] - \
        #         2*r_cc*len/(unit*base_Z*1000)*Pijc[k]- \
        #         2*x_cc*len/(unit*base_Z*1000)*Qijc[k]+ \
        #         (r_ca+np.sqrt(3)*x_ca)*len/(unit*base_Z*1000)*Pija[k] +\
        #         (x_ca-np.sqrt(3)*r_ca)*len/(unit*base_Z*1000)*Qija[k] +\
        #         (r_cb-np.sqrt(3)*x_cb)*len/(unit*base_Z*1000)*Pijb[k] +\
        #         (x_cb+np.sqrt(3)*r_cb)*len/(unit*base_Z*1000)*Qijb[k] +M*(1-xij[k]) >=0
        #     else:
        #         prob += Vic[int(line[0])-1]-Vic[int(line[1])-1] - \
        #         2*r_cc*len/(unit*base_Z*1000)*Pijc[k]- \
        #         2*x_cc*len/(unit*base_Z*1000)*Qijc[k]+ \
        #         (r_ca+np.sqrt(3)*x_ca)*len/(unit*base_Z*1000)*Pija[k] +\
        #         (x_ca-np.sqrt(3)*r_ca)*len/(unit*base_Z*1000)*Qija[k] +\
        #         (r_cb-np.sqrt(3)*x_cb)*len/(unit*base_Z*1000)*Pijb[k] +\
        #         (x_cb+np.sqrt(3)*r_cb)*len/(unit*base_Z*1000)*Qijb[k] ==0

        for t in range(T):
            prob += Via[1316, t] == 1.1
        # prob += Vib[1316] == 1.1
        # prob += Vic[1316] == 1.1

        # Enforce radial constraints
        nC = loops.__len__()
        for t in range(T):
            for k in range(nC):
                Sw = loops[k]
                nSw_C = np.count_nonzero(Sw)
                n_r = nSw_C-1
                prob += lpSum(xij[Sw[j]-1,t] for j in range(0, nSw_C)) <= n_r

        # Now writing the active and reactive power capacity constraints for feeder and DGs
        Fed = [266, 595, 924, 1253]
        for t in range(T):
            for k in range(0,4):
                prob += Pija[Fed[k],t] <= 2000
                prob += Pijb[Fed[k],t] <= 2000
                prob += Pijc[Fed[k],t] <= 2000
                prob += Qija[Fed[k],t] <= 1000
                prob += Qijb[Fed[k],t] <= 1000
                prob += Qijc[Fed[k],t] <= 1000
            
        DG = [1323, 1324, 1325, 1326]
        #Cap=[1500, 710, 526, 710] 
        Cap=[0, 0, 0, 0]
        for t in range(T):
            for k in range(0,4):
                prob += Pija[DG[k],t] <= Cap[k]
                prob += Pijb[DG[k],t] <= Cap[k]
                prob += Pijc[DG[k],t] <= Cap[k]

        # Following switches are never closed in sequence. First one is fault isolation switch 
        # and remaining are tie switches and DG switch 
        Status = [1227, 1316, 1318, 1319, 1323, 1324, 1325, 1326]
        nStatus = Status.__len__()
        for t in range(T):
            for k in range (0, nStatus):
                prob += xij[Status[k],t] == 0
        
        # Following switches need to change their status in every time step to reach the final state.
        sec = [573, 913, 1240]
        tie = [1317, 1320, 1321, 1322]
        
        # Number of switching actions for each time step to be 1
        for t in range(1, T):
            prob += (xij[573,t-1]-xij[573,t]) + (xij[913,t-1]-xij[913,t]) + (xij[1240,t-1]-xij[1240,t]) +\
                (xij[1317,t]-xij[1317,t-1]) + (xij[1320,t]-xij[1320,t-1]) + (xij[1321,t]-xij[1321,t-1]) + (xij[1322,t]-xij[1322,t-1]) == 1
        
        # Switches status cannot be reversed as it takes time to operate
        for t in range(1, T):
            for k in range(sec.__len__()):
                prob += (xij[sec[k],t-1] >= xij[sec[k],t])

        for t in range(1, T):
            for k in range(tie.__len__()):
                prob += (xij[tie[k],t] >= xij[tie[k],t-1])

        # At end they should reach the status given by restoration problem        
        prob += xij[573,T-1] == 0
        prob += xij[913,T-1] == 0
        prob += xij[1240,T-1] == 0
        prob += xij[1317,T-1] == 1
        prob += xij[1320,T-1] == 1
        prob += xij[1321,T-1] == 1
        prob += xij[1322,T-1] == 1
        
        # Loads once picked cannot be again shed in the restoration problem
        for t in range(1, T):
            for k in range (nNodes):
                prob += si[k,t] >= si[k,t-1]

        # # Write load switch status as we obtain from restoration problem
        # nobj = obj.__len__()
        # for t in range(T-1, T):
        #     for k in range (0, nobj):
        #         prob += si[obj[k],t] == 0

        # Freezing the remaining switches which donot take part in the restoration problem to be 1
        Status=[573, 913, 1240, 1227, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326]
        for t in range(T):
            for k in range (nEdges):
                if k not in Status:
                    prob += xij[k,t] == 1

        # prob.solve(CPLEX(msg=1, options=['set mip tolerances mipgap 0.0025']))
        prob.solve(CPLEX(msg=1))
        print ("Status:", LpStatus[prob.status])

        varsdict = {}
        for v in prob.variables():
            varsdict[v.name] = v.varValue

        # See the switch status
        for k in range(sec.__len__()):   
            sw_status = []  
            for var in xij:   
                if var[0] == sec[k]:             
                    var_value = xij[var].varValue
                    sw_status.append(var_value)
            print(sec[k], sw_status)    

        for k in range(tie.__len__()):   
            sw_status = []  
            for var in xij:   
                if var[0] == tie[k]:             
                    var_value = xij[var].varValue
                    sw_status.append(var_value)
            print(tie[k], sw_status)  

        # See the Feeder flow
        Feda = []
        for k in range(Fed.__len__()):   
            FlowPa = [] 
            for var in Pija:   
                if var[0] == Fed[k]:             
                    var_value = Pija[var].varValue
                    FlowPa.append(var_value)
            Feda.append(FlowPa)
            print(FlowPa)

        Fedb = []
        for k in range(Fed.__len__()):   
            FlowPb = []     
            for var in Pijb:   
                if var[0] == Fed[k]:             
                    var_value = Pijb[var].varValue
                    FlowPb.append(var_value)
            Fedb.append(FlowPb)
            print(FlowPb)

        Fedc = []
        for k in range(Fed.__len__()):   
            FlowPc = []     
            for var in Pijc:   
                if var[0] == Fed[k]:             
                    var_value = Pijc[var].varValue
                    FlowPc.append(var_value) 
            Fedc.append(FlowPc)
            print(FlowPc)           

        print(Feda)
        Tot_P = sum(demand[:,1] + demand[:,3] + demand[:,5])
        lossT= []
        for t in range(T):
            supplied = 0.
            for i in range(Fed.__len__()):
                supplied += Feda[i][t] + Fedb[i][t] + Fedc[i][t]
            loss = Tot_P - supplied
            lossT.append(loss)
            print (loss)
        
        print (lossT)
        x = [0, 1, 2, 3, 4, 5, 6]
        plt.step(x,lossT)
        plt.show()
        

def _main():
    RR = Restoration_Sequence()
    RR.sequence()

if __name__ == "__main__":
    _main()