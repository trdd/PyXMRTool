===========
Overview
===========

The usual usage of PyXMRTool is to write a script which set up an instance of :class:`Experiment.ReflDataSimulator` and to minimize the value of its method :meth:`Experiment.ReflDataSimulator.getSRR` with respect to the parameter array.
The 4 modules of PyXMRTool can guide you through these steps: 

 * set up the parameter pool and maybe already some of the parameters: :mod:`Parameters`
 * set up the sample representation (the most extensive part): :mod:`SampleRepresentation`
 * set up the experiment and data representation: :mod:`Experiment`
 * fitting: :mod:`Fitters` and appropiate tools from numpy/scipy or similar
 
In the following chapter an overview over the possibility within these steps will be given. Examples can be found in the folders *tutorials* and *testing*. A more extensive documentation of the modules can be found in :doc:`api`.

.. toctree::
    :maxdepth: 1
    
    overview/parameters
    overview/samplerepresentation
    overview/experiment
    overview/fitters