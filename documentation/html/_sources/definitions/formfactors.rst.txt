=============================
Formfactor and Susceptibility
=============================

The simulation of reflectivities within PyXMRTool (based on Pythonreflectivity) relies on the energy-dependent susceptibities :math:`\chi(E)` of the layers. It is in general a 3 x 3 tensor and is related to the complex refractive index :math:`n` and the dielectric tensor :math:`\epsilon`:

.. math::
 \epsilon = \chi + 1 = n^2
 
In some cases the susceptibility :math:`\chi(E)` of one layer is calculated as a sum over the formfactors :math:`f_i(E)` of atoms contained in this layer.
In contrast to the susceptibility the formfactor is not an physical quantity and therefor different authors use different sign conventions.
Within PyXMRTool the following conventions are used:

The formfactor with real and imaginary part: 

.. math::
    f=f^\prime + \mathrm(i) f^{\prime\prime}

Relation to the absorption cross section: 

.. math::
    f^{\prime\prime}(E) = \frac{k}{4\pi}\sigma_{abs}(E)
    
with the vacuum wave vector :math:`k` of the incomming light. This means the imaginary part of the formfactor is always positiv. A resonance shows up as a "positiv peak".

The real part is the Kramers-Kronig transformation of the imaginary part:

.. math::
    f^\prime(E)= - \frac{2}{\pi}\mathrm{CH}\int_0^\infty \frac{\eta \cdot f^{\prime\prime}(\eta)}{\eta^2-E^2} \, d\eta
 
 
This means that :math:`f^\prime` is typically negative within the x-ray range. Only close to resonances it might be positiv. There it behaves similar to the derivative of :math:`f^{\prime\prime}` (concerning the slopes).
 
With the above given definitions the susceptibility is:

.. math::
 \chi(E) = \frac{4 \pi r_0}{k^2} \sum_j \rho_{at,j} f_j(E)
 
with
 * :math:`r_0=2.8179403227 \cdot 10^{-15} \, \mathrm{m}` the classical electron radius 
 * :math:`k=\frac{E}{\hbar c}` the vacuum wave vector of the incomming light
 * :math:`\rho_{at,j}` the number density of atom species j
 * :math:`f_j` the formfactor (tensor) of atom species j 
 
 
**BEWARE: The Chantler-Tables (https://dx.doi.org/10.18434/T4HS32) use a different sign convention. As long as you use the automatic lookup the conversion is done automatically by PyXMRTool. But as soon as you read formfactors from files or create formfactors in some other user-controlled way, you have to stick to the above mentioned sign conventions.**

This is how you have to transform them the formfactors from the Chantler tables to the ones used within PyXMRTool:

.. math::
 
 f^{\prime\prime}_{PyXMRTool} = f^{\prime\prime}_{Chantler} 
 
 f^{\prime}_{PyXMRTool} = - f^{\prime}_{Chantler} 
 


