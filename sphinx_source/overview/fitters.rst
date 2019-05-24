======================== 
Fitting
========================

To perform a fit you first need start parameters and lower and upper boundaries for the parameters. Usually, you will define them in the parameter file and read it while creating the parameter pool. Then you can get the three arrays like this:
    >>> start, lower, upper = simu.getStartLowerUpper()
    
If you want to have a look on how your simulation with the start values of the parameters compares to the data you can plot both:
    >>> simu.plotData(start)
    
For fitting you can use whatever routine you found to be good. E.g. in the package *scipy* there is a quite good least squares fitter. You can use it like this
    >>> import scipy.optimize
    >>> result = scipy.optimize.least_squares(simu.getResiduals, start, bounds=(l,u), method='trf', x_scale=numpy.array(u)-numpy.array(l), jac='3-point',verbose=2)
    >>> best = result.x

*best* is then the parameter array with the fitted values. You can plot your result:
    >>> simu.plotData(best)

You can write the result to a file:
    >>> pp.writeToFile("parameters_best.txt",best)
