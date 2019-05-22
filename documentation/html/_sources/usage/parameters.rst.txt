===================
Module *Parameters*
===================

The module :mod:`Parameters` is the heart of *PyXMRTool*. Its main purpose is to manage all the fit parameters which occur somewhere in model in such a way that the user has to care as less as possible.

The most important classes for the parameter managment are :class:`Parameters.ParameterPool` and :class:`Parameters.Fitparameter`.

At first create a parameter pool:

    >>> from PyXMRTool import Parameters
    >>> pp=Parameters.ParameterPool()

Then create some fitparameters (instances of :class:`Parameters.Fitparameter`)  with names inside of the parameter pool:
    
    >>> d=pp.newParameter("thickness")
    >>> sigma=pp.newParameter("roughness")
    
So, there are now two parameters inside the pool. So we need a parameter array of length 2 and each of the instances of :class:`Parameters.Fitparameter` knows how to pick its value from it:

    >>> fitpararray=[67.2 , 3.2]
    >>> d.getValue(fitpararray)
    67.2
    >>> sigma.getValue(fitpararray)
    3.2
    
In the modelling later on you only deal with the :class:`Parameters.Fitparameter` object and you dont't have to care about the parameter array and the values. 
There is a another class called :class:`Parameters.Parameter` which is actually the base class of :class:`Parameters.Fitparameter`. It is used as interface. Many classes and functions in :mod:`Experiment` und :mod:`SampleRepresentation` expect instances of 
:class:`Parameters.Parameter` or derived classes as input. So, if you don't want to have a fittable parameter at a certain position you can use the compatible :class:`Parameters.Parameter` instead:

    >>> atomic_layer_thickness=Parameters.Parameter(11.2)
    >>> atomic_layer_thickness.getValue(fitpararray)
    11.2

In this case *fitpararray* is not necessary:
    
    >>> atomic_layer_thickness.getValue()
    11.2
    


You can also perform basic math operations with instances of :class:`Parameters.Parameter` and :class:`Parameters.Fitparameter`. Like this you can create derived parameters which depend on the values of the original ones:
    
    >>> number_of_atomic_layers = d / atomic_layer_thickness
    >>> type(number_of_atomic_layers)
    <PyXMRTool.Parameters.Parameter object at 0x7f7ed70b5c50>
    >>> number_of_atomic_layers.getValue(fitpararray)
    6.000000000000001

More advanced derived parameters which are arbitrary functions of the original parameters can be created with :class:`Parameters.DerivedParameter`.

You can also create functions which depend on parameters, so called parametrized functions, with :class:`Parameters.ParametrizedFunction`.


The :class:`Parameters.Fitparameter` objects can also carry start values and lower and upper boundaries for fitting procedure and a flag if the corresponding parameter should be fixed during fitting. Usually, these are set via a parameter file. If you didn't set these values so far you can create a draft file containing all the created fit parameters like this:

    >>> pp.writeToFile('parameters.txt')

It gives you a blank file where you can enter the start values and so on. If you want to read the file either use from the beginning:

    >>> pp=Parameters.ParameterPool('parameters.txt')

Or if the pool is already created:
    
    >>> pp.readFromFile('parameters.txt')
    
When you need the start values and boundaries for the fitting you can get them (3 different arrays but with the same length) like this:

    >>> start, lower, upper = pp.getStartLowerUpper()