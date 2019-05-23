=============================
Module *SampleRepresentation*
=============================

The most extensive part of the work is to create the model for the sample.

We start with a rough overview:
At first you create a heterostructure (:class:`SampleRepresentation.Heterostructure`) which consists of a certain number of layers.
Each layer is represented by an layer object which is pluged into the heterostructure object. There are different types but they are all derived from :class:`SampleRepresentation.LayerObject`. Each layer carries several properties. The common ones are *thickness* and *roughness*, which are just instances of :class:`Parameters.Parameter` and therefore fitable. More advanced is the energy-dependent :math:`\chi`-tensor (susceptibility tensor) describing the optical properties of the layer (The :math:`\chi`-tensor is used instead of the dielectric tensor as is close to zero instead of close to one. This gives higher numeric acuracy.) Its treatment differs for the different layer types:

    * :class:`SampleRepresentation.LayerObject`: Energy-independent susceptibility tensor given as array of instances of :class:`Parameters.Parameter`.
    * :class:`SampleRepresentation.MagneticLayerObject`: Energy-independent magnetic susceptibility tensor. Diagonal elements as instances of :class:`Parameters.Parameter`. Off-diagonal elements created from complex magnetic term and angles.
    * :class:`SampleRepresentation.ModelChiLayerObject`: Susceptibility tensor is user-defined parametrized function of energy. 
    * :class:`SampleRepresentation.AtomLayerObject`: Contains different atoms each with a formfactor and a density. The :math:`\chi`-tensor is result of a summation. 
    
The last layer object is the most advanced and complicated one. Moreover, the formfactors of the atoms are given as objects which are instances of :class:`SampleRepresentation.Formfactor` but there are many derived classes for different purposes or sources of the formfactor data resp. If you just want to get a formfactor from the Chantler tables it is enough to state the name of the atom instead of creating a formfactor object by yourself.

    * :class:`SampleRepresentation.Formfactor`: Abstract class which just severs as interface and is base class for all the other formfactor classes.
    * :class:`SampleRepresentation.FFfromFile`: Get the energy-dependent formfactor from a text file.
    * :class:`SampleRepresentation.FFfromChantler`: Get the energy-dependent formfactor from the Chantler tables.
    * :class:`SampleRepresentation.FFfromScaledAbsorption`: Formfactor from a XAS measurement but scalable.
    * :class:`SampleRepresentation.FFfromFitableModel`: Formfactor tensor is user-defined parametrized function of energy.
    * :class:`SampleRepresentation.MagneticFormfactor`: Off-diagonal elements only given by magnetization and angles. Energy-dependent magnetic termes are user-defined-parametrized functions.
    * :class:`SampleRepresentation.MFFfromXMCD`: Off-diagonal elements only given by magnetization and angles. Energy-dependent magnetic termes are taken from an XMCD measurement.
    
Also related to the layer type :class:`SampleRepresentation.AtomLayerObject` are the density profile classes :class:`SampleRepresentation.DensityProfile` and :class:`SampleRepresentation.DensityProfile_erf`. They can be used for the following scenario: You want to model a arbitrary concentration profiles of the atoms in your sample. Therefore you slice your sample in thin layers, each layer represented by an instance of :class:`SampleRepresentation.AtomLayerObject`. Then you can use the density profile classes to control the atom densitiy in each layer and by this create density profiles. See also tutorial *concentration_profile*.
    

Simple Example
==============
Here a short example shall be given. For more elaborate examples have a look in the folder *Tutorials*.

We want to create a model for a sample with two layers: one substrate with an energy-independent susceptibility and a layer of C\ :sub:`2`\ O.

As always start with the creation of the parameter pool:
    >>> from PyXMRTool import Parameters
    >>> from PyXMRTool import SampleRepresentation
    >>> pp=Parameters.ParameterPool()

Then create the heterostructure with to layers:
    >>> hs=SampleRepresentation.Heterostructure(2)

Create the substrate with an isotropic (same value on each diogonal entry) energy-indepedent suszeptibility, infinite thickness (0 equals infinity here), and a fitable roughness.
    >>> substrate = SampleRepresentation.LayerObject([pp.newParameter('substrate_chi')], Parameters.Parameter(0), pp.newParameter('substrate_sigma'))

Create the layer of C\ :sub:`2`\ O. Densities are here number densities. The layer has fitable thickness and sigma.
    >>> number_density_C2O = Parameters.Parameter(0.042)   #in mol/cm^3
    >>> layer1 = SampleRepresentation.AtomLayerObject( { "C" : 2*number_density_C2O, "O" : number_density_C2O}, pp.newParameter('layer_thickness'), pp.newParameter('layer_sigma'))
                                              
Plug the layers into the substrate object.                                        
    >>> hs.setLayer(0, substrate)
    >>> hs.setLayer(1, layer1)