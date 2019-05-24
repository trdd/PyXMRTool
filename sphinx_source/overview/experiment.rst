======================== 
Module *Experiment*
========================

Once you created the sample representation you have to set up the experiment and read the measured data. Here we will have a look on the example of the tutorial *LSMO_heterostructure*. But it is basically always the same procedure.

Instantiate experiment for linear polarization ("l") shown and fitted on a logarithmic scale ("L") and set length scale to Angstroem (1e-10 m).
    >>> simu=Experiment.ReflDataSimulator("lL",length_scale=1e-10)
    
The measured reflectivity data lies in the folder "Experiment" in the subfolders "pi" and "sigma" for pi and sigma polarization seperately. For each energy there is an file named like this "sro_lsmo_fixE_630_sigma.dat". The files contain two columns: q_z (instead of angle theta) and the reflectivity.

We need a function which extracts the energy from the filename (in general it is also possible to extract the angle from the filename, theirfore it must deliver a tupel of *(energy, angle)* of which one entry can be *None*).
    >>> namereader=lambda string: (float(string[:-4].split("_")[-2]), None)           #liefert energy aus den Dateinamen der verwendeten dateien

PyXMRTool expects angles not q_z values. So from the file we read q_z values as angles. This we can fix with a point modifier function. It takes one datapoint and returns a modified one. Datapoint means here a tuple like of two independent variables *energy* and *angle* and different reflectivity values as independent variables: *(energy,angle,rsigmag,rpi,rleft,rright,xmcd,total)*. Here we just replace the second entry.
    >>> def pointmodifier(point):        #berechnet winkel aus qz und energy und ersetzt qz dadurch. Alle anderen Werte des Datenpunktes bleiben unveraendert
    ...     point[1]=180.0/(numpy.pi)*numpy.arcsin( point[1]*simu.hcfactor/(2*numpy.pi)/(2*point[0]))
    ...     return point
    
Read measured data from files using *pointmodifier* and *namerreader*. Read pi and sigma polarization seperately. Use the funtion :class:`Experiment.ReflDataSimulator.createLinereader` to create a linereader function. This function returns an datapoint (as mentioned above) from a line of the data file. It is also possible to read pi and sigma polarization from just one file if the lines of the file look like this: :code:`angle, rpi, rsigma` or :code:`angle_pi, rpi, angle_sigma, rsigma`. Entries for the datapoint can also be *None* if the information is not present in a line of text.
    >>> simu.ReadData("Experiment/pi",simu.createLinereader(angle_column=0,rpi_column=1), pointmodifierfunction=pointmodifier , filenamereaderfunction=namereader)
    >>> simu.ReadData("Experiment/sigma",simu.createLinereader(angle_column=0,rsigma_column=1), pointmodifierfunction=pointmodifier , filenamereaderfunction=namereader)

Connect model of the sample with experiment. Thereby it is also posibble to define a reflection modifier function. It is applied to the simulated reflectivity and should reproduce e.g. arbitray factors or backgrounds present in the experiment.
    >>> b=pp.newParameter("background")
    >>> m=pp.newParameter("multiplier")
    >>> reflmodifier=lambda r, fitpararray: b.getValue(fitpararray) + r * m.getValue(fitpararray)
    >>> simu.setModel(hs,exp_energyshift=pp.newParameter("exp_Eshift"), exp_angleshift=pp.newParameter("exp_thetashift"),reflmodifierfunction=reflmodifier)

Now, *ReflDataSimulator* is set up completely. It is now possible to get the residuals between measured data and simulated data with parameter vector *fitpararray* like this:
    >>> residuals = simu.getResiduals(fitpararray)
    
Or the sum of squared residuals (SSR) like this:
    >>> ssr = simu.getSSR(fitpararray)

Or both:
    >>> residuals, ssr = simu.getResidualsSSR(fitpararray)

These functions are used in the next step to fit the model to the data.

You can also plot data and simulation for a certain parameter vector like this:
>>> simu.plotData(fitpararray)