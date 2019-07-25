# Halo_Matcher
Python code to match halos, using output from GADGET and SUBFIND

Input:
      tag:        Simulation tag to be saved with.
      SN:         SN of the simulation to be read in.
      redshift:   redshift of snapshot.
      N_bound:    number of most bound particles want to match to.
      N_part:     Number of particles that are in the simulation.
      Sims:       List of simulations in the suite.
      ref_sim:    simulation which particles in the rest of the suite will be matched to. Note if this is set to None, 
                  then the first simulation in the Sims list will be used.
      output_dir: where catalogue will be saved.
    

Requirements:
Numpy
Scipy
readEagle
