=============================
Module *SampleRepresentation*
=============================

The most extensive part of the work is to create the model for the sample.

At first you create a heterostructure (:class:`SampleRepresentation.Heterostructure`) which consists of a certain number of layers.

Each layer is represented by an layer object which is pluged into the heterostructure object. There are different types but they are all derived from :class:`SampleRepresentation.LayerObject`. Each layer carries several properties. The common ones are *thickness* and *roughness*, which are just instances of :class:`Parameters.Parameter` and therefore fitable. More advanced is the energy-dependent :math:`\chi`-tensor describing the optical properties of the layer (The :math:`\chi`-tensor is used instead of the dielectric tensor as is close to zero instead of close to one. This gives higher numeric acuracy.) Its treatment differs for the different layer types:

    * :class:`SampleRepresentation.LayerObject`:
    * :class:`SampleRepresentation.MagneticLayerObject`:
    * :class:`SampleRepresentation.ModelChiLayerObject`:
    * :class:`SampleRepresentation.AtomLayerObject`: