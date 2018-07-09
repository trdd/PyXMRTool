#!/usr/bin/env python
"""Contains different optimization algorithms designed for to fit reflectivity data.
   They take advantage of parallelization to be used on multiprocessor system.
   
   The algorithms are developed by Martin Zwiebel and I just adopted them to PyXMRTool.
   More information can be found in the PhD thesis of Martin Zwiebler.
"""

#Python Version 2.7

__author__ = "Yannic Utz"
__copyright__ = ""
__credits__ = ["Yannic Utz and Martin Zwiebler"]
__license__ = ""
__version__ = ""
__maintainer__ = "Yannic Utz"
__email__ = "yannic.utz@tu-dresden.de"
__status__ = "Prototype"


import numbers
import numpy
import scipy.linalg
import os.path
import joblib
import types
import copy

#settings############################
numerical_derivative_factor=1.0e-9                          #defines in principle the the magnitude of  "Delta x" for the aproximation of a derivative by "Delta y/Delta x"
#####################################

def Evolution(costfunction, (startfitparameters, lower_limits, upper_limits), iterations, number_of_cores=1, generation_size=300, mutation_probability=0.6, mutation_strength=0.15, elite=2, parent_percentage=0.25, control_file=None):
    """
    Evolutionary fit algorithm. Slow but good in finding the global minimum.
    Return the optimized parameter set and the coresponding value of the costfunction.
    
    \'costfunction\' should usally be the method \'getSSRLog\' of an instance of \'ReflDataSimulator\' (returns the sum of squared residuals of the logarithm of the reflectivities).
       Can also be any other function which takes the array of fit parameters and returns one real value should be minimized by \'Evolution\'.
    (\'startfitparameters\',\'lower_limits\',\'upper_limits): tuple/list of arrays/lists of start values/lower limits/upper limits for the fit parameters.
    \'iterations\': number of iterations
    \'number_of_cores\': number of jobs used in parallel. Best performance when set to the number of available cores on your computer.
    \'generation_size\':    Use this many individual fit parameter sets in each step
    \'mutation_probability\' chance of mutation of a particular inherited fit parameter
    \'mutation_strength\' Mutates by this factor
    \'elite\' Remember the best individuals for the next generation
    \'parent_percentage\' #Take this subset for reproduction
    
    If \'control_file\' is given, you can abort the optimization routine by writing "terminate 1" to the beginning of its first line
    """
    
    #check parameters
    #if not callable(costfunction):
    #    raise TypeError("\'costfunction\' has to be callable.")
    if not (len(startfitparameters)==len(lower_limits) and len(lower_limits)==len(upper_limits)):
        raise ValueError("\'startfitparameters\' and constraints don't match in length.")
    pos_integer_pars=[number_of_cores,iterations,generation_size]
    for p in pos_integer_pars:
        if not isinstance(p, numbers.Integral):
            raise TypeError("Invalid parameter.")
        if p<0:
            raise ValueError("Parameter has to be positive.")
    pos_real_pars=[mutation_probability, mutation_strength, elite, parent_percentage]
    for p in pos_real_pars:
        if not isinstance(p, numbers.Real):
            raise TypeError("Invalid parameter.")
        if p<0:
            raise ValueError("Parameter has to be positive.")
    if control_file is not None and not os.path.exists(control_file):
        raise Exception(str(control_file)+" is not an existing path.")
    

      
    #use numpy arrays
    startfitparameters=numpy.array(startfitparameters)
    lower_limits=numpy.array(lower_limits)
    upper_limits=numpy.array(upper_limits)
    
    number_of_parents=int(generation_size*parent_percentage)
    
   
    #Initial state:
    number_of_fitparameters=len(startfitparameters)
    randomnumbers=numpy.random.rand(generation_size, number_of_fitparameters)
    all_fitpararrays=lower_limits+randomnumbers*(upper_limits-lower_limits)
    children=numpy.zeros((generation_size, number_of_fitparameters))
    
    print "Start Evolution of "+str(iterations)+" Generations with "+str(len(startfitparameters))+" Parameters."
    ite=0
    while True:

        ite+=1

        #Calculate the costfunction (usually chisquare) for each individuum
        out=joblib.Parallel(n_jobs=number_of_cores)(joblib.delayed(costfunction)(all_fitpararrays[i]) for i in range(generation_size) )  
        ranking_list=numpy.argsort(out)                                 #stores the best results as indices of the elements of out
        #Write the current state and check for termination
        print "   Generation " + str(ite) + ": Cost=" + str(out[ranking_list[0]])
        if control_file is not None:
            with open(control_file) as f:
                for line in f:
                    line=line.split(" ")
                    if(line[0]=="terminate" and int(line[1])==1):
                        print "!! Iteration terminated, returning current status"
                        return all_fitpararrays[ranking_list[0]], out[ranking_list[0]]

        #return if number of iterations is reached
        if(ite==iterations):
            return all_fitpararrays[ranking_list[0]], out[ranking_list[0]]       #return best fit parameters and corresponding value of the costfunction  
        
        #determin next generation
        for i in range(generation_size):
            #Copy the elite
            if(i<elite):
                children[i]=all_fitpararrays[ranking_list[i]]
            else:
                #Reproduce children from the mother and father randomly
                Mother=i%number_of_parents;
                Father=numpy.random.randint(0,number_of_parents)
                #No Cloning
                while(Father==Mother):
                    Father=numpy.random.randint(0,number_of_parents)

                children[i]=0.5*(all_fitpararrays[ranking_list[Mother]]+all_fitpararrays[ranking_list[Father]])

                #Mutate the children
                r=2*numpy.random.rand(number_of_fitparameters)-1
                children[i]=children[i]*(1+r*mutation_strength)
                for j in range(number_of_fitparameters):
                    if(children[i][j]<lower_limits[j]):
                        children[i][j]=lower_limits[j]
                    elif(children[i][j]>upper_limits[j]):
                        children[i][j]=upper_limits[j]
        #The children are now the new generation
        all_fitpararrays=children
        
        

        
        
def Levenberg_Marquardt_Fitter(residualandcostfunction, number_of_datapoints, (startfitparameters, lower_limits, upper_limits), parallel_points ,number_of_cores=1, strict=True, control_file=None):
    """
    Modified Levenberg-Marquard algorithm (see PhD thesis of Martin Zwiebler). Good convergence, but might end up in a local mininum.
    Return the optimized parameter set and the coresponding value of the costfunction.
    
    \'residualandcostfunction\' should usally be the method \'getResidualsSSR\' of an instance of \'ReflDataSimulator\'. Can also be any other function which takes the array of fit parameters and returns a tuple of 
        1.) a list of residuals (will be used to determine derivatives) 2.) a value of the costfunction which should be minimized (usually the sum of squared residuals)
    \'number_of_datapoints\': total number of data points used for the fitting (beware: for reflectivities each direction counts once)
    (\'startfitparameters\',\'lower_limits\',\'upper_limits): tuple/list of arrays/lists of start values/lower limits/upper limits for the fit parameters.
    \'parallel_points\': This should be something like the number of threads that can run in parallel/number of cores. The algorithm will first find a direction for a good descent and then check this number of points on the line. The best one will yield the new fit parameter set
    \'number_of_cores\': number of jobs used in parallel. Best performance when set to the number of available cores on your computer.
    \'strict\': usually this algorithm fails if the residuals are locally independent of one of the parameters. If you set \'stict=False\' this parameter will be neglected locally.
    
    If \'control_file\' is given, you can abort the optimization routine by writing "terminate 1" to the beginning of its first line
    """
    
    #check parameters
    if not callable(residualandcostfunction):
        raise TypeError("\'residualandcostfunction\' has to be callable.")
    if not (len(startfitparameters)==len(lower_limits) and len(lower_limits)==len(upper_limits)):
        raise ValueError("\'startfitparameters\' and constraints don't match in length.")
    pos_integer_pars=[number_of_cores,parallel_points,number_of_datapoints]
    for p in pos_integer_pars:
        if not isinstance(p, numbers.Integral):
            raise TypeError("Invalid parameter.")
        if p<0:
            raise ValueError("Parameter has to be positive.")
    if not isinstance(strict,bool):
        raise TypeError("\'strict\' has to be of type bool.")
    if control_file is not None and not os.path.exists(control_file):
        raise Exception(str(control_file)+" is not an existing path.")
    

   
    #use numpy arrays
    startfitparameters=numpy.array(startfitparameters)
    lower_limits=numpy.array(lower_limits)
    upper_limits=numpy.array(upper_limits)
    
    number_of_fitparameters=len(startfitparameters)
    
    aite=startfitparameters

    #this Matrix stores all the residuals derivatives
    DT=numpy.zeros((number_of_fitparameters,number_of_datapoints))

    "Start Levenberg-Marquardt algorithm with "+str(len(startfitparameters))+" Parameters."
    ite=0
    while(True):

        if(ite>0):
            #Go here once you calculated the first step

            #Calculate the fit parameters on the line of descent
            all_fitpararrays=[copy.copy(aite) for i in range(parallel_points)]
            for i in range(parallel_points):
                scale=0.5**i
                all_fitpararrays[i]=all_fitpararrays[i]-scale*DTr - scale*(1-scale)*DTr2


            for j in range(parallel_points):
                for i in range(number_of_fitparameters):
                    if(all_fitpararrays[j][i]<lower_limits[i]):
                        all_fitpararrays[j][i]=lower_limits[i]
                    elif(all_fitpararrays[j][i]>upper_limits[i]):
                        all_fitpararrays[j][i]=upper_limits[i]

          
            ##Calculate the residuals and cost (e.g. difference between simulated and measured reflectivities) on the line of descent, in parallel
            out=numpy.array(joblib.Parallel(n_jobs=number_of_cores)(joblib.delayed(residualandcostfunction)( all_fitpararrays[i] ) for i in range(parallel_points) ))
            

            
            #Find the lowest chisquare
            min_i=numpy.argmin(out[:,1])
            aite=copy.copy(all_fitpararrays[min_i])
            
            fiterror2=out[min_i][1]
            
            if( abs( (fiterror1-fiterror2)/(fiterror1+fiterror2) ) < 1.0e-7 ):
                print( "  --> Converged at cost=" + str(fiterror2) )
                return aite, out[min_i][1]                      #return best fit parameters and corresponding value of the costfunction      
            elif control_file is not None:
                with open(control_file) as f:
                    for line in f:
                        line=line.split(" ")
                        if(line[0]=="terminate" and int(line[1])==1):
                            print "!! Iteration terminated, current status"
                            return aite, out[min_i][1]
            
            print "   Iteration "+ str(ite)+": old cost = "+str(fiterror1)+", new cost = "+str(fiterror2)
            
       
        
           
        
        #Make Fit parameters for the calculation of the derivative
        all_fitpararrays=[copy.copy(aite) for i in range(number_of_fitparameters+1)]          ##this Matrix stores fitparameters for each point that is calculated in parallel
        delta=numpy.zeros(number_of_fitparameters)
        for i in range(number_of_fitparameters):
            if(all_fitpararrays[i][i]==0 or (lower_limits[i]<0<upper_limits[i]) ):
                delta[i]=numerical_derivative_factor*max( abs(upper_limits[i]), abs(lower_limits[i] ) )
            else:
                delta[i]=numerical_derivative_factor*all_fitpararrays[number_of_fitparameters][i]
            all_fitpararrays[i][i]+=delta[i]


        ##Calculate the residuals and cost (e.g. difference between simulated and measured reflectivities) at each delta-step, in parallel

        out=joblib.Parallel(n_jobs=number_of_cores)(joblib.delayed(residualandcostfunction)( all_fitpararrays[i] ) for i in range(number_of_fitparameters+1) )



        #Calculate the first fit error
        if(ite==0):
            fiterror1= out[ number_of_fitparameters ][1]
        else:
            fiterror1=fiterror2
        #Calculate the derivative of the reflectivity
        for i in range(number_of_fitparameters):
            DT[i] = ( out[i][0]-out[number_of_fitparameters][0] )/delta[i]

        ##Calculate the gradient
        
        A=numpy.dot(DT,DT.T)
        b=numpy.dot(DT,(out[number_of_fitparameters][0]).T )

        #If one of the derivatives is entirely zero, the fit parameter is essentially meaningless. That may happen for a number of reasons. However, Gauss-Newton fails for this case.
        #If strict=False make it work nevertheless
        irrelevantparameterlist=[]
        for i in range(number_of_fitparameters):
            if(b[i]==0):
                if strict:
                    print "WARNING! No gradient component" + str(i) + "! Singular matrix!\n Try \'strict=False\' for less rigorous treatment."
                    print(b)
                    raise Exception
                else:
                    irrelevantparameterlist.append(i)
        if not strict:
            #remove the elements corresponding to the irrelevant parameters
            print "WARNING! Parameters " + str(irrelevantparameterlist) + " are locally irrelevant (no gradient component) and will be ignored for this step."
            DT_reduced=numpy.delete(DT,irrelevantparameterlist,0)
            A=numpy.dot(DT_reduced,DT_reduced.T)
            b=numpy.delete(b,irrelevantparameterlist,0)

        #Solve this system of equations to calculate the descent vector that is used for large steps
        DTr=scipy.linalg.solve(A,b,sym_pos=True)
        #This is another good descent vector that is used for small steps
        DTr2=numpy.zeros(number_of_fitparameters-len(irrelevantparameterlist))
        for i in range(number_of_fitparameters-len(irrelevantparameterlist)):
            DTr2[i]=b[i]/A[i][i]
        
        if not strict:
            #fill components of the descent vector which correspond to irrelevant parameters, with zero
            for parind in irrelevantparameterlist:
                DTr=numpy.insert(DTr,parind,0,0)
                DTr2=numpy.insert(DTr2,parind,0,0)
            
        ite+=1