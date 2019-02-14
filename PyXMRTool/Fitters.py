#<PyXMRTool: A Python Package for the analysis of X-Ray Magnetic Reflectivity data measured on heterostructures>
#    Copyright (C) <2018>  <Yannic Utz>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""Contains different optimization algorithms designed to fit reflectivity data.
   They take advantage of parallelization to be used on multiprocessor system.
   
   The algorithms are developed by Martin Zwiebel and I just adopted them with slight changes to PyXMRTool.
   More information can be found in the PhD thesis of Martin Zwiebler.
   
   The algorithms are not well developed yet. It is better to use existing optimizers. E.g. *scipy.optimize.least_squares*.
   
   
"""

#Python Version 2.7

__author__ = "Yannic Utz"
__copyright__ = ""
__credits__ = ["Yannic Utz and Martin Zwiebler"]
__license__ = "GNU General Public License v3.0"
__version__ = "0.9"
__maintainer__ = "Yannic Utz"
__email__ = "yannic.utz@tu-dresden.de"
__status__ = "beta"


import numbers
import numpy
import scipy.linalg
import scipy.optimize
import os.path
import joblib
import types
import copy
import sklearn.preprocessing
import sklearn.cluster
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

#settings############################
numerical_derivative_factor=1.0e-9                          #defines in principle the the magnitude of  "Delta x" for the aproximation of a derivative by "Delta y/Delta x"
#####################################

def Evolution(costfunction, parameter_settings , iterations, number_of_cores=1, generation_size=300, mutation_strength=0.01, elite=2, parent_percentage=0.25, control_file=None, plotfunction=None):
    """
    Evolutionary fit algorithm. Slow but good in finding the global minimum.
    Return the optimized parameter set and the coresponding value of the costfunction.
    
    Parameters
    ----------
    costfunction : callable
        A function which returns a measure (cost) for the difference between measurement and simulated data according to the paramter set given as list of values. Usually the sum of squared residuals (SSR) is used as cost. It should usally be the method :meth:`SampleRepresentation.ReflDataSimulator.getSSR` of an instance of :class:`SampleRepresentation.ReflDataSimulator` wrapped in a function. The wrapping is necessaray due to some implemetation issues connected to the parallelization.
        Example for the wrapping::
            
            simu = SampleRepresentation.ReflDataSimulator("l")
            ...
            def cost(fitpararray):
                return simu.getSSR(fitpararray)
    
        Pass then the function *cost* as **costfunction**. It can also be any other function which takes the array of fit parameters and returns one real value which should be minimized by :meth:`.Evolution`.
        
    parameter_settings: tuple of lists of floats
        Sets start values, lower and upper limit of the parameters as *(startfitparameters, lower_limits, upper_limits )*, where each of the entries is an list/array of values of same length.
    iterations : int 
        number of iterations/generations
    number_of_cores : int 
        Number of jobs used in parallel. Best performance when set to the number of available cores on your computer.
    generation_size : int 
        Generate this many individual fit parameter sets in each generation.
    mutation_strength : float
        Mutates children by adding this factor times (upper_limit - lower_limit)  --> use rather small values 
    elite : int
        Remember the best individuals for the next generation.
    parent_percentage : flota
        Use this fraction of a gereneration (the best) for reproduction.
    control_file : str 
        Filename of a control file. If it is given, you can abort the optimization routine by writing "terminate 1" to the beginning of its first line.
    plotfunction : callable
        Function which is used to plot the current state of fitting (simulated data with currently best parameter set) after every iteration if given. It should take only one parameter: the array of fitparameters.
    
    
    This Evolutionary algorithm is mainly the same as Martins. Only the rule for mutation has changed:
    
    | Martin: ``children[i]=children[i] * (1 + s * random float(-1,1))``
    | I:      ``children[i]=children[i] + s * random float(-1,1)*(upper_limits-lower_limits)``

    """
    #unpack parameter settings
    (startfitparameters, lower_limits, upper_limits )=parameter_settings
    
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
    pos_real_pars=[mutation_strength, elite, parent_percentage]
    for p in pos_real_pars:
        if not isinstance(p, numbers.Real):
            raise TypeError("Invalid parameter.")
        if p<0:
            raise ValueError("Parameter has to be positive.")
    if control_file is not None and not os.path.exists(control_file):
        raise Exception(str(control_file)+" is not an existing path.")
    if plotfunction is not None and not callable(plotfunction):
        raise TypeError("\'plotfunction\' has to be callable.")

    

      
    #use numpy arrays
    startfitparameters=numpy.array(startfitparameters)
    lower_limits=numpy.array(lower_limits)
    upper_limits=numpy.array(upper_limits)
    
    number_of_parents=int(generation_size*parent_percentage)
    
   
    #Initial state:
    number_of_fitparameters=len(startfitparameters)
    randomnumbers=numpy.random.rand(generation_size, number_of_fitparameters)
    all_fitpararrays=lower_limits+randomnumbers*(upper_limits-lower_limits)
    all_fitpararrays[0]=startfitparameters                                          #use the given start values just as one of many guesses (replace one random guess)
    children=numpy.zeros((generation_size, number_of_fitparameters))
    
    print "Start Evolution of "+str(iterations)+" Generations with "+str(len(startfitparameters))+" Parameters."
    ite=0
    while True:

        ite+=1

        #Calculate the costfunction (usually chisquare) for each individuum
        out=joblib.Parallel(n_jobs=number_of_cores)(joblib.delayed(costfunction)(all_fitpararrays[i]) for i in range(generation_size) )  
        ranking_list=numpy.argsort(out)                                 #stores the best results as indices of the elements of out
        #Write the current state
        print "   Generation " + str(ite) + ": Cost=" + str(out[ranking_list[0]])
        #plot the current state (best guess)
        if plotfunction is not None:
            plotfunction(all_fitpararrays[ranking_list[0]])
        #check for termination
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
                children[i]=children[i] + r*mutation_strength                  #changed mutation rule by Yannic (mutation not depending on parameter value)
                for j in range(number_of_fitparameters):    #periodic boundaries (changed by Yannic; Martin sets to the limits if theey are exceeded)
                    if(children[i][j]<lower_limits[j]):
                        children[i][j]=upper_limits[j]-(lower_limits[j]-children[i][j])%(upper_limits[j]-lower_limits[j])
                    elif(children[i][j]>upper_limits[j]):
                        children[i][j]=lower_limits[j]+(children[i][j]-upper_limits[j])%(upper_limits[j]-lower_limits[j])
        #The children are now the new generation
        all_fitpararrays=children
        
        

        
        
def Levenberg_Marquardt_Fitter(residualandcostfunction,  parameter_settings , parallel_points ,number_of_cores=1, strict=True, convergence_criterium=1e-7, control_file=None, plotfunction=None):
    """
    Modified Levenberg-Marquard algorithm (see PhD thesis of Martin Zwiebler). Good convergence, but might end up in a local mininum.
    Return the optimized parameter set and the coresponding value of the costfunction.
    
    
    Parameters
    ----------
    residualandcostfunction : callable
        A function which returns the differences between simulated and measured data points (residuals) as list and a scalar measure (cost) for these differences in total according to the paramter set given as list of values. Usually the sum of squared residuals (SSR) is used as cost. It should usally be the method :meth:`SampleRepresentation.ReflDataSimulator.getResidualsSSR` of an instance of :class:`SampleRepresentation.ReflDataSimulator` wrapped in a function. The wrapping is necessaray due to some implemetation issues connected to the parallelization.
        Example for the wrapping::
            
            simu = SampleRepresentation.ReflDataSimulator("l")
            ...
            def rescost(fitpararray):
                return simu.getResidualsSSR(fitpararray)
    
        Pass then the function *rescost* as **costfunction**. It can also be any other function which takes the array of fit parameters and returns a tuple of 
        1.) a list of residuals (will be used to determine derivatives) 2.) a value of the costfunction which should be minimized by :meth:`.Levenberg_Marquardt_Fitter`.
        
    parameter_settings: tuple of lists of floats
        Sets start values, lower and upper limit of the parameters as *(startfitparameters, lower_limits, upper_limits )*, where each of the entries is an list/array of values of same length.
    parallel_points : int 
        This should be something like the number of threads that can run in parallel/number of cores. The algorithm will first find a direction for a good descent and then check this number of points on the line. The best one will yield the new fit parameter set.
    number_of_cores : int 
        Number of jobs used in parallel. Best performance when set to the number of available cores on your computer.
    strict : bool
        Usually this algorithm fails if the residuals are locally independent of one of the parameters. If you set **stict** = *False* this parameter will be neglected locally.
    convergence_criterium : float
        If the relative difference between the costs in two succeeding iterations is smaller than **convergence_criterium**, the fitting is defined as \`converged\`.
    control_file : str 
        Filename of a control file. If it is given, you can abort the optimization routine by writing "terminate 1" to the beginning of its first line.
    plotfunction : callable
        Function which is used to plot the current state of fitting (simulated data with currently best parameter set) after every iteration if given. It should take only one parameter: the array of fitparameters.
    """
    
    #unpack parameter settings
    (startfitparameters, lower_limits, upper_limits )=parameter_settings
    
    #check parameters
    if not callable(residualandcostfunction):
        raise TypeError("\'residualandcostfunction\' has to be callable.")
    if not (len(startfitparameters)==len(lower_limits) and len(lower_limits)==len(upper_limits)):
        raise ValueError("\'startfitparameters\' and constraints don't match in length.")
    pos_integer_pars=[number_of_cores,parallel_points]
    for p in pos_integer_pars:
        if not isinstance(p, numbers.Integral):
            raise TypeError("Invalid parameter.")
        if p<0:
            raise ValueError("Parameter has to be positive.")
    if not isinstance(strict,bool):
        raise TypeError("\'strict\' has to be of type bool.")
    if control_file is not None and not os.path.exists(control_file):
        raise Exception(str(control_file)+" is not an existing path.")
    if plotfunction is not None and not callable(plotfunction):
        raise TypeError("\'plotfunction\' has to be callable.")

    

   
    #use numpy arrays
    startfitparameters=numpy.array(startfitparameters)
    lower_limits=numpy.array(lower_limits)
    upper_limits=numpy.array(upper_limits)
    
    number_of_fitparameters=len(startfitparameters)
    
    aite=startfitparameters

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
            
            if( abs( (fiterror1-fiterror2)/(fiterror1+fiterror2) ) < convergence_criterium ):
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
            
            #plot the current state of fitting
            if plotfunction is not None:
                plotfunction(aite)
            

            
       
        
           
        
        #Make Fit parameters for the calculation of the derivative
        all_fitpararrays=[copy.copy(aite) for i in range(number_of_fitparameters+1)]          ##this Matrix stores fitparameters for each point that is calculated in parallel
        delta=numpy.zeros(number_of_fitparameters)
        for i in range(number_of_fitparameters):
        #YANNIC: geaendert, weil es mir so sinnvoller erschien
            #if(all_fitpararrays[i][i]==0 or (lower_limits[i]<0<upper_limits[i]) ):
                #delta[i]=numerical_derivative_factor*max( abs(upper_limits[i]), abs(lower_limits[i] ) )
            #else:
                #delta[i]=numerical_derivative_factor*all_fitpararrays[number_of_fitparameters][i]  
            delta[i]=numerical_derivative_factor*abs(upper_limits[i]-lower_limits[i] )
            all_fitpararrays[i][i]+=delta[i]


        ##Calculate the residuals and cost (e.g. difference between simulated and measured reflectivities) at each delta-step, in parallel

        out=joblib.Parallel(n_jobs=number_of_cores)(joblib.delayed(residualandcostfunction)( all_fitpararrays[i] ) for i in range(number_of_fitparameters+1) )
        
        if(ite==0):
            #Calculate the first fit error
            fiterror1= out[ number_of_fitparameters ][1]
            #in first iteration: create the matrix DT to stores all the residuals derivatives
            number_of_datapoints=len(out[0][0])
            DT=numpy.zeros((number_of_fitparameters,number_of_datapoints))
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
        if not strict and not irrelevantparameterlist==[]:
            #remove the elements corresponding to the irrelevant parameters
            print "WARNING! Parameters " + str(irrelevantparameterlist) + " are locally irrelevant (no gradient component) and will be ignored for this iteration."
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
        
def explore(residualsfunction,  parameter_settings, number_of_seeds, number_of_clusters=8, verbose=2):
    """
    A scanning function which should be usefull to explore the parameter space.
    
    It chooses **number_of_seeds** different random start parameter vectors (seeds) within the given paramteter range. Each seed is used as start parameter set for a least_square fitter to find the minimum of the sum of squared residuals (*ssr*) (using :func:`scipy.optimize.least_squares` with the *trust region reflective algorithm). This will lead to **number_of_seeds** fixpoints. They will then be analysed with a (k-means) clustering algorithm to group these fixpoint in **number_of_clusters** different clusters. These clusters will then be analysed: What is the SSR corresponding to the cluster centers? How many seeds lead to the corresponding clusters? What are the means and spreads of parameter values within each cluster?
    
    Parameters
    ----------
    residualsfunction : callable
        A function which returns the differences between simulated and measured data points (residuals) as list/array. It should usally be the method :meth:`SampleRepresentation.ReflDataSimulator.getResiduals` of an instance of :class:`SampleRepresentation.ReflDataSimulator`.        
    parameter_settings : tuple of lists/arrays of floats
        Sets start values, lower and upper limit of the parameters as *(startfitparameters, lower_limits, upper_limits )*, where each of the entries is an list/array of values of same length. The *startfitparameters* are not used (can be *None*) and just necessaray for compatibility.
    number_of_seeds : int 
        number of random seeds which should be generated
    number_of_clusters : int 
        number of clusters in which the resulting fixpoints shall be grouped
    verbose : {0, 1, 2}
        determines the level of the optimizer's algorithm's verbosity:
        
            0 : work silently.
            1 : display a termination report for each seed.
            2 (default) : display progress during iterations.


    """
    
    #unpack parameter settings
    (startfitparameters, lower_limits, upper_limits )=parameter_settings
    
    #check parameters
    if not callable(residualsfunction):
        raise TypeError("\'residualsfunction\' has to be callable.")
    if not len(lower_limits)==len(upper_limits):
        raise ValueError("Constraints does not match in length.")
    pos_integer_pars=[number_of_seeds,number_of_clusters]
    for p in pos_integer_pars:
        if not isinstance(p, numbers.Integral):
            raise TypeError("Invalid parameter. Has to be integer.")
        if p<0:
            raise ValueError("Parameter has to be positive.")

    #performing the least squares optimizations
    print("... performing least squares optimization for "+str(number_of_seeds) +" seeds")
    fixpoints=[]
    upper_limits=numpy.array(upper_limits)
    lower_limits=numpy.array(lower_limits)
    for i in range(number_of_seeds):
        res = scipy.optimize.least_squares(residualsfunction, numpy.random.rand(len(upper_limits))*(upper_limits-lower_limits)+lower_limits, bounds=(lower_limits,upper_limits), method='trf', x_scale=upper_limits-lower_limits, jac='3-point',verbose=1)
        fixpoints.append(res.x)
    fixpoints=numpy.array(fixpoints)
    
    #performing cluster analysis
    print("... clustering "+str(number_of_seeds) +" fixpoints in "+str(number_of_clusters)+" clusters")
    scaler=sklearn.preprocessing.StandardScaler()       #parameters have to be scaled to allow for a reasonable clustering
    fixpoints_scaled=scaler.fit_transform(fixpoints)
    km=sklearn.cluster.KMeans(n_clusters=number_of_clusters).fit(fixpoints_scaled)
    clusters=[]
    clusters_members=[]
    for i,v in enumerate(km.cluster_centers_):
        number_fp=list(km.labels_).count(i)
        cluster_center=scaler.inverse_transform(v)
        members=fixpoints[numpy.where(km.labels_==i)]
        std_dev=numpy.std(members,axis=0)
        max_spread=numpy.max(members,axis=0)-numpy.min(members,axis=0)
        clusters.append({'center': cluster_center, 'std_dev': std_dev, 'rel_std_dev': std_dev/cluster_center, 'max_spread': max_spread, 'rel_max_spread': max_spread/cluster_center  , 'number_of_members': number_fp, 'ratio_of_seeds': float(number_fp)/len(fixpoints), 'ssr_of_center': numpy.sum(numpy.square(numpy.array(residualsfunction(cluster_center))))})
        clusters_members.append(members)
    print "... printing overview"
    scan_output={'fixpoints': fixpoints, 'clusters': clusters, 'clusters_members': clusters_members}
    list_clusters(scan_output)
    return scan_output


    
def list_clusters(scan_output):
    clusters=scan_output['clusters']
    for i,v in enumerate(clusters):
        print "cluster "+str(i)+": catches "+str(v['ratio_of_seeds']*100)+"% of seeds, ssr_of_center="+str(v['ssr_of_center'])
        
def plot_parameter_spread(scan_output, pnumber):
    #create a color for each cluster
    cmap=plt.get_cmap('gist_rainbow')
    n_clusters=len(scan_output['clusters'])
    colors=[cmap(i/float(n_clusters-1),0.5) for i in range(n_clusters)]
    #get center values
    centers=[]
    for cl in scan_output['clusters']:
        centers.append(cl['center'][pnumber])
    #get sums of squared residuals
    ssrs=[]
    for cl in scan_output['clusters']:
        ssrs.append(cl['ssr_of_center'])
    #get ratios of seeds (meaning: how many of the seeds convered to a certain cluster)
    rofs=[]
    for cl in scan_output['clusters']:
        rofs.append(cl['ratio_of_seeds'])
    rofs=numpy.array(rofs)
    #get spreads of the parameter values (min and max)
    spreads_lower=[]
    spreads_upper=[]
    for i,cl in enumerate(scan_output['clusters_members']):
        spreads_lower.append(centers[i]-min(cl[:,pnumber]))
        spreads_upper.append(max(cl[:,pnumber])-centers[i])
    spreads=[spreads_lower,spreads_upper]
    spreads=numpy.array(spreads)
    #ploting circles with size related to the ratio of seeds, position=(center,ssr)
    plt.scatter(centers,ssrs,s=rofs*5000,c=colors,edgecolors='face')
    #plotting errorbars related to the spread
    plt.errorbar(centers,ssrs, xerr=spreads,fmt='.',color='black')
    #legend
    patches=[]
    for i,c in enumerate(colors):
        patches.append(mpatches.Patch(color=c, label=str(i)))
    plt.legend(handles=patches)
    plt.show()