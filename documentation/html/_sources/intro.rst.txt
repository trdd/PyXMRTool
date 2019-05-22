==================
Introduction 
==================

*PyXMRTool* is a flexibel Python3 package for the analysis of Resonant X-ray Reflectivity measurements built on the package *Pythonreflectivity*. More specifically it allows to fit a model of a heterostructure to the reflectivity data.

The package has a strict object oriented design. It should enable the user to add together the building blocks in a script.
The package has a modular structure to account for different topics.

The module :mod:`SampleRepresentation` contains everything needed to build the model of the sample. In the end, an object of the class :class:`SampleRepresentation.Heterostructure` contains all the necessary information and serves as model representation.

The above mentioned module or its classes resp. use parameters which should be fitted to the data. To allow for a centralized handling of parameters instances of :class:`Parameters.Fitparameters` are organized within a parameter pool. Like this every parameter "knows" how to get its value from an parameter array. All this and some more can be found in the module :mod:`Parameters`. 

The module :mod:`Experiment` deals with the experimental setup, which polarizations are measured, parametrized background etc and it deals with the measured data. All this is represented by an instance of the class :class:`Experiment.ReflDataSimulator`. It also delivers the sum of squared residuals (SSR) or the residuals themself in dependence of a parameter array.

Fitting is then the procedure of minimizing this SSR by varying the parameter array. Self-written fitting routines can be found in the module :mod:`Fitters`, but it is better to use professional ones (see :ref:`hints-fitting-label`). But the module :mod:`Fitters` contains also a tool to scan for different fixpoints in the parameter range, to explore it using *scipy.optimize.least_squares*. See :func:`Fitters.Explore`.

.. figure:: module_scheme.png




