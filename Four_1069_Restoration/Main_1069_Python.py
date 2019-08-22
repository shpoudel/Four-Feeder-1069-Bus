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

class Restoration(object):
    """
    WSU Resilient Restoration Sequence of Switches
    """
   
    def __init__(self, fault):
        self.fault = fault

    def restoration(self):  

        # Define parameters and load data
        nNodes=1317
        nEdges=1327
        edges = np.loadtxt('edges.txt') 
        demand = np.loadtxt('LoadData.txt')
        loops = np.loadtxt('cycles.txt') 
        LineData = np.loadtxt('Line_Config.txt')
        sw_ind = np.loadtxt('sw_ind.txt')
        con_ind = np.loadtxt('config.txt') # Find a line index in Line_Config file
        mult = -1*(demand[:,1]+demand[:,3]+demand[:,5])
        M=3000

        # Different variables for optimization function
        vi = LpVariable.dicts("v_i", (i for i in range(nNodes)), lowBound=0, upBound=1, cat='Binary')
        si = LpVariable.dicts("s_i", (i for i in range(nNodes)), lowBound=0, upBound=1, cat='Binary')
        xij = LpVariable.dicts("xl", (i for i in range(nEdges)), lowBound=0, upBound=1, cat='Binary')
        xij0 = LpVariable.dicts("xl0", (i for i in range(nEdges)), lowBound=0, upBound=1, cat='Binary')
        xij1 = LpVariable.dicts("xl1", (i for i in range(nEdges)), lowBound=0, upBound=1, cat='Binary')
        Pija = LpVariable.dicts("xPa", (i for i in range(nEdges)), lowBound=-M, upBound=M, cat='Continous')
        Pijb = LpVariable.dicts("xPb", (i for i in range(nEdges)), lowBound=-M, upBound=M, cat='Continous')
        Pijc = LpVariable.dicts("xPc", (i for i in range(nEdges)), lowBound=-M, upBound=M, cat='Continous')
        Qija = LpVariable.dicts("xQa", (i for i in range(nEdges)), lowBound=-M, upBound=M, cat='Continous')
        Qijb = LpVariable.dicts("xQb", (i for i in range(nEdges)), lowBound=-M, upBound=M, cat='Continous')
        Qijc = LpVariable.dicts("xQc", (i for i in range(nEdges)), lowBound=-M, upBound=M, cat='Continous')
        Via = LpVariable.dicts("xVa", (i for i in range(nNodes)), lowBound=0.9025, upBound=1.1, cat='Continous')
        Vib = LpVariable.dicts("xVb", (i for i in range(nNodes)), lowBound=0.9025, upBound=1.1, cat='Continous')
        Vic = LpVariable.dicts("xVc", (i for i in range(nNodes)), lowBound=0.9025, upBound=1.1, cat='Continous')

        # Indices for tie switches and virtual switches to insert into objetive functions
        N=[1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326]

        # Optimization problem objective definitions
        prob = LpProblem("Resilient Restoration",LpMinimize)
        prob += lpSum(si[k] * mult[k] for k in range(nNodes))+\
                lpSum(xij[N[k]] for k in range(0, 11))-\
                lpSum(xij[k] for k in range(0, 1316))
            
        # Constraints (v_i<=1)
        for k in range(0,nNodes):
            prob += vi[k] <= 1
            
        # Constraints (s_i<=v_i)
        for k in range(0,nNodes):
            prob += si[k] <= vi[k]
            
        # Constraints (x_ij<=v_i*v_j). This is non-linear and is linearized here..
        for k in range(0,nEdges):
            prob += xij[k] <= vi[edges[k,0]-1]
            prob += xij[k] <= vi[edges[k,1]-1]
            
        # Constraints (x_ij0+x_ij1<=x_ij)
        for k in range(0,nEdges):
            prob += xij0[k] + xij1[k] <= xij[k]
            
        # Real power flow equation for Phase A phase B and Phase C
        fr=edges[:,0]
        to=edges[:,1]
        for k in range(0, nEdges):    
            ed=int(edges[k,1]-1)
            node=edges[k,1]
            pa=np.array(np.where(to==edges[k,1]))
            pa=pa.flatten()
            N=range(0,pa.__len__())
            ch=np.array(np.where(fr==edges[k,1]))
            ch=ch.flatten()
            M=range(0,ch.__len__())
            prob += lpSum(Pija[pa[j]] for j in N) - demand[ed,1] * si[node-1]== \
            lpSum(Pija[ch[j]] for j in M)
            
        for k in range(0, nEdges):    
            ed=int(edges[k,1]-1)
            node=edges[k,1]
            pa=np.array(np.where(to==edges[k,1]))
            pa=pa.flatten()
            N=range(0,pa.__len__())
            ch=np.array(np.where(fr==edges[k,1]))
            ch=ch.flatten()
            M=range(0,ch.__len__())   
            prob += lpSum(Pijb[pa[j]] for j in N)-demand[ed,3] * si[node-1]== \
            lpSum(Pijb[ch[j]] for j in M)
        
        for k in range(0, nEdges):    
            ed=int(edges[k,1]-1)
            node=edges[k,1]
            pa=np.array(np.where(to==edges[k,1]))
            pa=pa.flatten()
            N=range(0, pa.__len__())
            ch=np.array(np.where(fr==edges[k,1]))
            ch=ch.flatten()
            M=range(0, ch.__len__())   
            prob += lpSum(Pijc[pa[j]] for j in N)-demand[ed,5] * si[node-1]== \
            lpSum(Pijc[ch[j]] for j in M)

        # Imposing the big-M method to ensure the real-power flowing in open line is zero
        M=3000
        R=sw_ind.__len__()
        for k in range(0, R):    
            prob += Pija[sw_ind[k]] <= M * xij1[sw_ind[k]]
            prob += Pijb[sw_ind[k]] <= M * xij1[sw_ind[k]] 
            prob += Pijc[sw_ind[k]] <= M * xij1[sw_ind[k]]     
            prob += Pija[sw_ind[k]] >= -M * xij0[sw_ind[k]]
            prob += Pijb[sw_ind[k]] >= -M * xij0[sw_ind[k]] 
            prob += Pijc[sw_ind[k]] >= -M * xij0[sw_ind[k]] 

        # Reactive power flow equation for Phase A phase B and Phase C
        fr=edges[:,0]
        to=edges[:,1]
        for k in range(0, nEdges):    
            ed=int(edges[k,1]-1)
            node=edges[k,1]
            pa=np.array(np.where(to==edges[k,1]))
            pa=pa.flatten()
            N=range(0, pa.__len__())
            ch=np.array(np.where(fr==edges[k,1]))
            ch=ch.flatten()
            M=range(0, ch.__len__()) 
            prob += lpSum(Qija[pa[j]] for j in N)-demand[ed,2] * si[node-1]== \
            lpSum(Qija[ch[j]] for j in M)

        for k in range(0, nEdges):    
            ed=int(edges[k,1]-1)
            node=edges[k,1]
            pa=np.array(np.where(to==edges[k,1]))
            pa=pa.flatten()
            N=range(0, pa.__len__())
            ch=np.array(np.where(fr==edges[k,1]))
            ch=ch.flatten()
            M=range(0, ch.__len__())      
            prob += lpSum(Qijb[pa[j]] for j in N)-demand[ed,4] * si[node-1]== \
            lpSum(Qijb[ch[j]] for j in M)
        
        for k in range(0, nEdges):    
            ed=int(edges[k,1]-1)
            node=edges[k,1]
            pa=np.array(np.where(to==edges[k,1]))
            pa=pa.flatten()
            N=range(0, pa.__len__()) 
            ch=np.array(np.where(fr==edges[k,1]))
            ch=ch.flatten()
            M=range(0, ch.__len__())
            prob += lpSum(Qijc[pa[j]] for j in N)-demand[ed,6] * si[node-1]== \
            lpSum(Qijc[ch[j]] for j in M)

        # Imposing the big-M method to ensure the reactive-power flowing in open line is zero
        M=3000
        R=sw_ind.__len__()
        for k in range(0, R):    
            prob += Qija[sw_ind[k]] <= M * xij1[sw_ind[k]]
            prob += Qijb[sw_ind[k]] <= M * xij1[sw_ind[k]] 
            prob += Qijc[sw_ind[k]] <= M * xij1[sw_ind[k]]     
            prob += Qija[sw_ind[k]] >= -M * xij0[sw_ind[k]]
            prob += Qijb[sw_ind[k]] >= -M * xij0[sw_ind[k]] 
            prob += Qijc[sw_ind[k]] >= -M * xij0[sw_ind[k]] 

        # Voltage constraints using big-M method
        base_Z=12.47**2/3
        M=4
        unit=5280
        # Phase A
        for k in range(0, nEdges):
            len=LineData[int(con_ind[k]),2]
            conf=LineData[int(con_ind[k]),4]
            r_aa,x_aa,r_ab,x_ab,r_ac,x_ac = zma.Zmatrixa(conf)
            line=[edges[k,0], edges[k,1]]
            if (k) in sw_ind:
                prob += Via[int(line[0])-1]-Via[int(line[1])-1] - \
                2*r_aa*len/(unit*base_Z*1000)*Pija[k]- \
                2*x_aa*len/(unit*base_Z*1000)*Qija[k]+ \
                (r_ab+np.sqrt(3)*x_ab)*len/(unit*base_Z*1000)*Pijb[k] +\
                (x_ab-np.sqrt(3)*r_ab)*len/(unit*base_Z*1000)*Qijb[k] +\
                (r_ac-np.sqrt(3)*x_ac)*len/(unit*base_Z*1000)*Pijc[k] +\
                (x_ac+np.sqrt(3)*r_ac)*len/(unit*base_Z*1000)*Qijc[k]-M*(1-xij[k]) <=0
                # Another inequality        
                prob += Via[int(line[0])-1]-Via[int(line[1])-1] - \
                2*r_aa*len/(unit*base_Z*1000)*Pija[k]- \
                2*x_aa*len/(unit*base_Z*1000)*Qija[k]+ \
                (r_ab+np.sqrt(3)*x_ab)*len/(unit*base_Z*1000)*Pijb[k] +\
                (x_ab-np.sqrt(3)*r_ab)*len/(unit*base_Z*1000)*Qijb[k] +\
                (r_ac-np.sqrt(3)*x_ac)*len/(unit*base_Z*1000)*Pijc[k] +\
                (x_ac+np.sqrt(3)*r_ac)*len/(unit*base_Z*1000)*Qijc[k]+M*(1-xij[k]) >=0
            else: 
                prob += Via[int(line[0])-1]-Via[int(line[1])-1] - \
                2*r_aa*len/(unit*base_Z*1000)*Pija[k]- \
                2*x_aa*len/(unit*base_Z*1000)*Qija[k]+ \
                (r_ab+np.sqrt(3)*x_ab)*len/(unit*base_Z*1000)*Pijb[k] +\
                (x_ab-np.sqrt(3)*r_ab)*len/(unit*base_Z*1000)*Qijb[k] +\
                (r_ac-np.sqrt(3)*x_ac)*len/(unit*base_Z*1000)*Pijc[k] +\
                (x_ac+np.sqrt(3)*r_ac)*len/(unit*base_Z*1000)*Qijc[k]==0

        # Phase B
        for k in range(0, nEdges):
            len = LineData[int(con_ind[k]),2]
            conf = LineData[int(con_ind[k]),4]
            line = [edges[k,0], edges[k,1]]
            r_bb,x_bb,r_ba,x_ba,r_bc,x_bc = zmb.Zmatrixb(conf)
            if (k) in sw_ind:
                prob += Vib[int(line[0])-1]-Vib[int(line[1])-1] - \
                2*r_bb*len/(unit*base_Z*1000)*Pijb[k]- \
                2*x_bb*len/(unit*base_Z*1000)*Qijb[k]+ \
                (r_ba-np.sqrt(3)*x_ba)*len/(unit*base_Z*1000)*Pija[k] +\
                (x_ba+np.sqrt(3)*r_ba)*len/(unit*base_Z*1000)*Qija[k] +\
                (r_bc+np.sqrt(3)*x_bc)*len/(unit*base_Z*1000)*Pijc[k] +\
                (x_bc-np.sqrt(3)*r_bc)*len/(unit*base_Z*1000)*Qijc[k] -M*(1-xij[k]) <=0
                # Another inequality
                prob += Vib[int(line[0])-1]-Vib[int(line[1])-1] - \
                2*r_bb*len/(unit*base_Z*1000)*Pijb[k]- \
                2*x_bb*len/(unit*base_Z*1000)*Qijb[k]+ \
                (r_ba-np.sqrt(3)*x_ba)*len/(unit*base_Z*1000)*Pija[k] +\
                (x_ba+np.sqrt(3)*r_ba)*len/(unit*base_Z*1000)*Qija[k] +\
                (r_bc+np.sqrt(3)*x_bc)*len/(unit*base_Z*1000)*Pijc[k] +\
                (x_bc-np.sqrt(3)*r_bc)*len/(unit*base_Z*1000)*Qijc[k] +M*(1-xij[k]) >=0
            else:
                prob += Vib[int(line[0])-1]-Vib[int(line[1])-1] - \
                2*r_bb*len/(unit*base_Z*1000)*Pijb[k]- \
                2*x_bb*len/(unit*base_Z*1000)*Qijb[k]+ \
                (r_ba-np.sqrt(3)*x_ba)*len/(unit*base_Z*1000)*Pija[k] +\
                (x_ba+np.sqrt(3)*r_ba)*len/(unit*base_Z*1000)*Qija[k] +\
                (r_bc+np.sqrt(3)*x_bc)*len/(unit*base_Z*1000)*Pijc[k] +\
                (x_bc-np.sqrt(3)*r_bc)*len/(unit*base_Z*1000)*Qijc[k] ==0

        # Phase C  
        for k in range(0, nEdges):
            len = LineData[int(con_ind[k]),2]
            conf = LineData[int(con_ind[k]),4]
            r_cc,x_cc,r_ca,x_ca,r_cb,x_cb = zmc.Zmatrixc(conf)
            line = [edges[k,0], edges[k,1]]
            if (k) in sw_ind: 
                prob += Vic[int(line[0])-1]-Vic[int(line[1])-1] - \
                2*r_cc*len/(unit*base_Z*1000)*Pijc[k]- \
                2*x_cc*len/(unit*base_Z*1000)*Qijc[k]+ \
                (r_ca+np.sqrt(3)*x_ca)*len/(unit*base_Z*1000)*Pija[k] +\
                (x_ca-np.sqrt(3)*r_ca)*len/(unit*base_Z*1000)*Qija[k] +\
                (r_cb-np.sqrt(3)*x_cb)*len/(unit*base_Z*1000)*Pijb[k] +\
                (x_cb+np.sqrt(3)*r_cb)*len/(unit*base_Z*1000)*Qijb[k] -M*(1-xij[k]) <=0
                # Another inequality
                prob += Vic[int(line[0])-1]-Vic[int(line[1])-1] - \
                2*r_cc*len/(unit*base_Z*1000)*Pijc[k]- \
                2*x_cc*len/(unit*base_Z*1000)*Qijc[k]+ \
                (r_ca+np.sqrt(3)*x_ca)*len/(unit*base_Z*1000)*Pija[k] +\
                (x_ca-np.sqrt(3)*r_ca)*len/(unit*base_Z*1000)*Qija[k] +\
                (r_cb-np.sqrt(3)*x_cb)*len/(unit*base_Z*1000)*Pijb[k] +\
                (x_cb+np.sqrt(3)*r_cb)*len/(unit*base_Z*1000)*Qijb[k] +M*(1-xij[k]) >=0
            else:
                prob += Vic[int(line[0])-1]-Vic[int(line[1])-1] - \
                2*r_cc*len/(unit*base_Z*1000)*Pijc[k]- \
                2*x_cc*len/(unit*base_Z*1000)*Qijc[k]+ \
                (r_ca+np.sqrt(3)*x_ca)*len/(unit*base_Z*1000)*Pija[k] +\
                (x_ca-np.sqrt(3)*r_ca)*len/(unit*base_Z*1000)*Qija[k] +\
                (r_cb-np.sqrt(3)*x_cb)*len/(unit*base_Z*1000)*Pijb[k] +\
                (x_cb+np.sqrt(3)*r_cb)*len/(unit*base_Z*1000)*Qijb[k] ==0
              
        prob += Via[1316] == 1.1
        prob += Vib[1316] == 1.1
        prob += Vic[1316] == 1.1

        # Active and reactive power capacity constraints for feeder and DGs
        Fed = [266, 595, 924, 1253]
        for k in range(0,4):
            prob += Pija[Fed[k]]<=2000
            prob += Pijb[Fed[k]]<=2000
            prob += Pijc[Fed[k]]<=2000
            prob += Qija[Fed[k]]<=1000
            prob += Qijb[Fed[k]]<=1000
            prob += Qijc[Fed[k]]<=1000

        # DGs Capacity   
        DG = [1323, 1324, 1325, 1326]
        # Cap = [1500, 710, 526, 710] 
        Cap = [0, 0, 0, 0]
        for k in range(0,4):
            prob += Pija[DG[k]]<=Cap[k]
            prob += Pijb[DG[k]]<=Cap[k]
            prob += Pijc[DG[k]]<=Cap[k]

        # Enforce radial constraints
        nC = loops.__len__()
        for k in range(0,nC):
            Sw = loops[k]
            nSw_C = np.count_nonzero(Sw)
            n_r = nSw_C-1
            prob += lpSum(xij[Sw[j]-1] for j in range(0, nSw_C)) <= n_r
       
        # Insert fault in the system
        F = self.fault
        nF=1
        for k in range(0, nF):
            prob += xij[F[k]] == 0

        prob.solve(CPLEX(msg=1, options=['set mip tolerances mipgap 0.0025']))
        print ("Status:", LpStatus[prob.status])

        varsdict = {}
        for v in prob.variables():
            varsdict[v.name] = v.varValue

        # Real and reactive power from Feeders
        P_fed1 = [varsdict['xPa_266'], varsdict['xPb_266'], varsdict['xPc_266']]
        P_fed2 = [varsdict['xPa_595'], varsdict['xPb_595'], varsdict['xPc_595']]
        P_fed3 = [varsdict['xPa_924'], varsdict['xPb_924'], varsdict['xPc_924']]
        P_fed4 = [varsdict['xPa_1253'], varsdict['xPb_1253'], varsdict['xPc_1253']]

        Q_fed1 = [varsdict['xQa_266'], varsdict['xQb_266'], varsdict['xQc_266']]
        Q_fed2 = [varsdict['xQa_595'], varsdict['xQb_595'], varsdict['xQc_595']]
        Q_fed3 = [varsdict['xQa_924'], varsdict['xQb_924'], varsdict['xQc_924']]
        Q_fed4 = [varsdict['xQa_1253'], varsdict['xQb_1253'], varsdict['xQc_1253']]

        # Real and reactive power from DGs
        P_DG1 = [varsdict['xPa_1323'], varsdict['xPb_1323'], varsdict['xPc_1323']]
        P_DG2 = [varsdict['xPa_1324'], varsdict['xPb_1324'], varsdict['xPc_1324']]
        P_DG3 = [varsdict['xPa_1325'], varsdict['xPb_1325'], varsdict['xPc_1325']]
        P_DG4 = [varsdict['xPa_1326'], varsdict['xPb_1326'], varsdict['xPc_1326']]

        Q_DG1 = [varsdict['xQa_1323'], varsdict['xQb_1323'], varsdict['xQc_1323']]
        Q_DG2 = [varsdict['xQa_1324'], varsdict['xQb_1324'], varsdict['xQc_1324']]
        Q_DG3 = [varsdict['xQa_1325'], varsdict['xQb_1325'], varsdict['xQc_1325']]
        Q_DG4 = [varsdict['xQa_1326'], varsdict['xQb_1326'], varsdict['xQc_1326']]

        Tot_P = sum(demand[:,1]+demand[:,3]+demand[:,5])
        Tot_Q = sum(demand[:,2]+demand[:,4]+demand[:,6])
        print(Tot_P)
        P_fed = sum(P_fed1+P_fed2+P_fed3+P_fed4)
        Q_fed = sum(Q_fed1+Q_fed2+Q_fed3+Q_fed4)
        P_DG = sum(P_DG1+P_DG2+P_DG3+P_DG4)
        Q_DG = sum(Q_DG1+Q_DG2+Q_DG3+Q_DG4)

        # Find how much load need to be shedded
        Loss_kW = Tot_P-(P_fed+P_DG)
        Loss_kVA = (Tot_P**2+Tot_Q**2)**0.5-((P_fed**2+Q_fed**2)**0.5+(P_DG**2+Q_DG**2)**0.5)
        print(Loss_kW)
        print(Loss_kVA)

        tie_switches = [varsdict['xl_1316'], varsdict['xl_1317'], varsdict['xl_1318'], varsdict['xl_1319'],\
        varsdict['xl_1320'], varsdict['xl_1321'],varsdict['xl_1322'],varsdict['xl_1323'],varsdict['xl_1324'],\
        varsdict['xl_1325'],varsdict['xl_1326']]  
        print(tie_switches)

        # Printing all sectionalizing switches having status as 0; i.e., Open
        x = LpVariable.dicts("xl",range(nEdges))
        for k in range (0, nEdges):
            if k in sw_ind:
                a = str(x[k])
                st = varsdict[a]
                if st < 0.5:
                    print(k)

        # Storing all the node variables which are shedded as requirement of restoration problem
        x = LpVariable.dicts("s_i",range(nNodes))
        open_s=[]
        for k in range (0, nNodes):
            a = str(x[k])
            st = varsdict[a]
            if st < 0.5:
                open_s.append(k)        
        with open('load_curtail.txt', 'w') as f:
            json.dump(open_s, f)
        
def _main():
    fault = [1227] #924
    RR = Restoration(fault)
    RR.restoration()

if __name__ == "__main__":
    _main()