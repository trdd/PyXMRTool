=====
Hints
=====


.. _hints-fitting-label:

Fitting
--------
The fitting algorithms in :mod:`Fitters` are not well develeped yet. It is better to use existing optimizers.

One option which delivers good performance is the least squares optimizer of scipy (*scipy.optimize.least_squares*) used in the following way (where *simu* is an instance of :class:`Experiment.ReflDataSimulator` and *pp* is an instance of :class:`Parameters.Parameterpool`)::

    >>> (start, l, u)=pp.getStartLowerUpper()
    >>> res= scipy.optimize.least_squares(simu.getResiduals, start, bounds=(l,u), method='trf', x_scale=u-l, jac='3-point',verbose=2)
    >>> best=res.x

With the given parameters a "trusted region reflective algorithm* (*method='trf'*) will be used. Each parameter is scaled by the difference between upper and lower boundary (*x_scale=u-l*). And for the approximation of the Jacobian 3 points are used (*jac='3-point'*).