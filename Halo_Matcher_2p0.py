import numpy as np
import time
from tqdm import tqdm
import eagle as E
from scipy import stats

def GN_match(Ordered_GNs,IDs_array,N_bound):

	matched_Group, _ = stats.mode(Ordered_GNs[IDs_array].reshape(-1,N_bound),axis=1)

	return matched_Group

def most_bound(IDs, Group_Length, N_bound):

	split_bound = np.argmin(Group_Length>=N_bound)
	Group_Length = Group_Length[:split_bound]
	IDs = IDs[:np.sum(Group_Length)]
	index_bound = np.insert(np.cumsum(Group_Length)[:-1],0,0)[:, None] + np.arange(N_bound)

	return IDs[index_bound.flatten().astype(np.int64)]

def Halo_Matcher(sims, SN='033', redshifts=0.0, N_bound = 50, N_part=1024, tag='', output_dir='./'):

	'''Return an array of group numbers, which have been matched across sims.

	Arguments:
	sims -- List of simulation directories to produce the matched halo catalogue for.
	
	Keyword arguments:
	SN -- snapshot tag required by readEagle when reading in simulation output. Default corresponds to
	      redshift 0 for a BAHAMAS simulation.
	redshift -- the redshift of the simulation snapshot being used - must correspond to correct SN.
		    Default is redshift 0.
	N_bound -- Number of most bound particles that will be used when matching halos. Default is set to 50.
	N_part -- Cubed root of particles in the simulation. Default assumes an N1024 simulation.
	tag -- Name for the simulation suite. Used when saving the output matched halo catalogue.
	output_dir -- Directory where the resultant halo catalogue will be output.

	Return:
	Catalogues -- A dictionary object, with key: value corresponding to the Snaps in SN: Halo
			  catalogue generated for that redshift. This halo catalogue is a numpy array with
			  all matched halo numbers wth shape (-1, len(sims)).

	Method:
	This halo matcher uses a bijective matching technique to match halos across simulations.
	It does this using particle IDs which encode the initial Lagrangian positions of the particles.
	It matches all halos which have more particles than N_bound from the reference simulation
	to each simulation in sims. Note, that the reference simulation is assumed to be the first element of
	the sims list. It then matches back from these simulations to the reference
	simulation, and keeps all halos which were able to be matched both backwards and forwards.

	The algorithm works primarily by setting up two arrays: Temp_Halo_Catalogue_ref and
	Temp_Halo_Catalogue_match. The former, holds halo numbers which have been matched from the
	reference simulation, to some simulation in sims. The latter holds halo numbers when matching
	back from the simulation to the reference simulation. The columns in these arrays correspond to:
	column 0: all halo numbers in the reference simulation.
	column 1: all halo numbers in the matched simulation.
	'''

	if 'BAHAMAS' in tag:
		print('Looks like this is a BAHAMAS suite of sims, using different Snapshot notation')
		Snaps = np.array(['%03d'%i for i in np.arange(18,33)])[::-1] #snapshots out to z = 3
		redshift = np.array([3.0,2.75,2.5,2.25,2.0,1.75,1.5,1.25,1,0.75,0.5,0.375,0.25,0.125,0])[::-1]
	else:
		Snaps = np.array([])
		redshift = np.array([])
		Snaps = np.append(Snaps,SN)
		redshift = np.append(redshift, redshifts)

	# Set up a dictionary for snapshots
	Snapshots = {}
	for val_i, key in enumerate(Snaps):
		Snapshots[key] = str(redshift[val_i])

	assert len(sims) > 1, 'Need at least 2 simulations to form a match'
	sims = np.array(sims)
	sim_ref = sims[0]
	sims = sims[1:]

	if N_part > 1024:
		int_type = np.int64
	else:
		int_type = np.int32

	Catalogues = {}

	for SN_i in Snaps:

		# Check to see if this halo catalogue already exists:
		redshift = Snapshots[SN_i]
		try:
			Halo_Catalogue = np.load('{}Halo_Catalogue_{}_z_{}.npy'.format(output_dir, tag, redshift.replace('.','p')))
			Catalogues[SN_i] = Halo_Catalogue
			continue
		except(FileNotFoundError):
				if sims is None:
					raise ValueError('Please provide a valid tag, or a list of simulation directories.')
				else:
					print('No halo catalogue found with tag: {}, at redshift: {}'.format(tag, redshift))
					print('Generating new halo catalogue')

		ref_Ordered_GNs = np.ones(N_part**3, dtype=int_type) * -1

		GNs = np.abs(E.readArray('PARTDATA', sim_ref, SN_i, '/PartType1/GroupNumber', verbose=False)) -1
		IDs = E.readArray('PARTDATA', sim_ref, SN_i, '/PartType1/ParticleIDs', verbose=False) -1

		ref_Ordered_GNs[IDs] = GNs

		ref_IDs = E.readArray('SUBFIND_PARTICLES', sim_ref, SN_i, '/IDs/ParticleID', verbose=False) - 1
		ref_Group_Length = E.readArray('SUBFIND_GROUP', sim_ref, SN_i, '/FOF/GroupLength', verbose=False)

		# Find halos which contain at least 50 particles in Sim1
		ref_Group_Numbers = np.where(ref_Group_Length >= N_bound)[0]

		# Make array of N_bound most bound particle in Sim1
		ref_most_bound = most_bound(ref_IDs, ref_Group_Length, N_bound)

		Halo_Catalogue = np.ones((np.max(ref_Group_Numbers)+1, len(sims)+1), dtype=int_type) * - 1
		Halo_Catalogue[:,0] = np.arange(0, np.max(ref_Group_Numbers)+1)

		for i, sim in enumerate(tqdm(sims)):
			# Match from Sim1 to Sim2
			# Create array to store current Group Numbers matching from Sim1 to Sim2
			Temp_Halo_Catalogue_ref = np.ones((np.max(ref_Group_Numbers)+1, 2), dtype=int_type) * -1
			Temp_Halo_Catalogue_ref[:,0] = np.arange(0,np.max(ref_Group_Numbers)+1)

			# Set up an array to be filled in with ordered group numbers
			match_Ordered_GNs = np.ones(N_part**3, dtype=int_type) * -1

			# Read in the particle IDs and Group Numbers of Sim2
			match_GNs = np.abs(E.readArray('PARTDATA', sim, SN_i, '/PartType1/GroupNumber', verbose=False)) -1
			match_IDs = E.readArray('PARTDATA', sim, SN_i, '/PartType1/ParticleIDs', verbose=False) -1
			match_Ordered_GNs[match_IDs] = match_GNs # Ordered by ID

			# Match halos from Sim1 to Sim2 and store matches in temporary catalogue
			Matches = GN_match(match_Ordered_GNs, ref_most_bound, N_bound)
			Temp_Halo_Catalogue_ref[ref_Group_Numbers,1] = Matches[:,0]

			# Now match from Sim2 to Sim1
			# Read in particle information for Sim2
			match_IDs = E.readArray('SUBFIND_PARTICLES', sim, SN_i, '/IDs/ParticleID', verbose=False) - 1
			match_Group_Length = E.readArray('SUBFIND_GROUP', sim, SN_i, '/FOF/GroupLength', verbose=False)

			# Find halos which contain at least 50 particles in Sim2
			match_Group_Numbers = np.where(match_Group_Length>=N_bound)[0]

			# Create array to store current Group numbers and matched Group Numbers from Sim2 to Sim1
			Temp_Halo_Catalogue_match = np.ones((np.max(match_Group_Numbers) + 1, 2), dtype=int_type) * -1
			Temp_Halo_Catalogue_match[:,1] = np.arange(0,np.max(match_Group_Numbers)+1)

			# Make N_bound most bound particle array for halos in Sim2
			match_most_bound = most_bound(match_IDs, match_Group_Length, N_bound)

			# Match halos from Sim2 to Sim1 using most bound particles
			Bi_Matches = GN_match(ref_Ordered_GNs, match_most_bound, N_bound)

			# Store matches in temporary halo catalogue
			Temp_Halo_Catalogue_match[match_Group_Numbers,0] = Bi_Matches[:,0]

			# Match between the halo catalogues of Sim1 and Sim2
			# Remove all halos which do not have a match straight away
			Temp_Halo_Catalogue_ref = Temp_Halo_Catalogue_ref[Temp_Halo_Catalogue_ref[:,1] != -1]
			Temp_Halo_Catalogue_match = Temp_Halo_Catalogue_match[Temp_Halo_Catalogue_match[:,0] != -1]

			# Now find which of the reference halos are in both sets of matched halos
			index_both = np.isin(Temp_Halo_Catalogue_match[:,0], Temp_Halo_Catalogue_ref[:,0])
			Temp_Halo_Catalogue_match = Temp_Halo_Catalogue_match[index_both]

			# Construct a common set of halos across the two simulations
			index_array = np.searchsorted(Temp_Halo_Catalogue_ref[:,0], Temp_Halo_Catalogue_match[:,0])
			Temp_Halo_Catalogue_ref = Temp_Halo_Catalogue_ref[index_array]
			true_match = Temp_Halo_Catalogue_ref[:,1] == Temp_Halo_Catalogue_match[:,1]
			Temp_Halo_Catalogue = Temp_Halo_Catalogue_ref[true_match]

			# Fill in the Halo catalogue with the matched set
			Halo_Catalogue[Temp_Halo_Catalogue[:,0],i+1] = Temp_Halo_Catalogue[:,1]

		Halo_Catalogue = np.delete(Halo_Catalogue, np.where(Halo_Catalogue==-1)[0],axis=0)
		np.save('{}Halo_Catalogue_{}_z_{}'.format(output_dir, tag, redshift.replace('.','p')), Halo_Catalogue)

		Catalogues[SN_i] = Halo_Catalogue

	return Catalogues


if __name__ == '__main__':

	SN = ['033', '027', '023']
	z = [0.0, 1.0, 2.0]
	tag = 'L100N256'
	Sim_path = '/path/to/sim/suite/'
	suite_tags = ['sim1','sim2','sim3','sim4','sim5']

	HC = Halo_Matcher(tag, SN=SN, redshifts=z, Sims = [Sim_path+i+'/data/' for i in suite_tags], N_part=256, N_bound=50)
	print(HC[SN[0]])
