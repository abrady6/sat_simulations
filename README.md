# sat_simulations

CONCEPTUAL OVERVIEW:

Code for simulating satellite-distributed entangled photons to ground-station pairs.  

Considers a polar-orbital constellation of satellites in space and an array of 
ground stations on the Earth's surface. Each satellite can distribute an entangled 
photon-pair to a ground-station pair, thus allowing the ground stations to share 
quantum entanglement. This is a quantum communication scenario over a pure-loss channel. 
Hence, the only input parameter is the transmittance, which is the probability that a single-photon 
will successfully be transmitted from satellite to ground. Since we are distributing to 
ground-station pairs, a transmittance value for each satellite-to-ground channel needs to be
simultaneously specified in time. This is done for every ground-station pair and every satellite.
Note that a higher transmittance value leads to higher quantum communication rates. Hence maximizing the
transmittance is important. Also, however, since we deal with a constellation of satellites, we 
must consider the number of satellites in the constellation. This number needs to be minimized 
in practice, since satellites (and going to space) are expensive. One can then optimize this entanglement 
distribution network by maximizing a figure of merit, which we define to be the average transmittance 
(averaged over a simulation period) divided by the total number of satellites in the constellation. 
Intuitively, this maximizes the 'average entanglement distribution rate per satellite' in the network. 
This maximization will depend on 1) the orbital altitude of the constellation and 2) the distance 
between ground station pairs. It will generally but only insensitively depend on the satellite 
constellation architecture as well. 
