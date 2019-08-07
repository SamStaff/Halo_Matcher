# Halo_Matcher
Python code to match halos, using output from GADGET and SUBFIND  
  
Return an array of group numbers, which have been matched across sims.  
Arguments:  
sims -- List of simulation directories to produce the matched halo catalogue for.  
  
Keyword arguments:  
SN -- snapshot tag required by readEagle when reading in simulation output. Default corresponds to  
      redshift 0 for a BAHAMAS simulation.  
n_bound -- Number of most bound particles that will be used when matching halos. Default is set to 50.    
tag -- Name for the simulation suite. Used when saving the output matched halo catalogue.  
output_dir -- Directory where the resultant halo catalogue will be output.  
Return:  
catalogues -- A dictionary object, with key: value corresponding to the snaps in SN: Halo  
              catalogue generated for that redshift. This halo catalogue is a numpy array with  
              all matched halo numbers wth shape (-1, len(sims)).  
Method:  
This halo matcher uses a bijective matching technique to match halos across simulations.  
It does this using particle IDs which encode the initial Lagrangian positions of the particles.  
It matches all halos which have more particles than n_bound from the reference simulation  
to each simulation in sims. Note, that the reference simulation is assumed to be the first element of  
the sims list. It then matches back from these simulations to the reference  
simulation, and keeps all halos which were able to be matched both backwards and forwards.  
The algorithm works primarily by setting up two arrays: temp_halo_catalogue_ref and  
temp_halo_catalogue_match. The former, holds halo numbers which have been matched from the  
reference simulation, to some simulation in sims. The latter holds halo numbers when matching  
back from the simulation to the reference simulation. The columns in these arrays correspond to:  
column 0: all halo numbers in the reference simulation.  
column 1: all halo numbers in the matched simulation.  
