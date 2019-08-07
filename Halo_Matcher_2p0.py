import numpy as np
import time
from tqdm import tqdm
import eagle3 as E
from scipy import stats

def GN_match(ordered_GNs,IDs_array,n_bound):

	matched_group, _ = stats.mode(ordered_GNs[IDs_array].reshape(-1,n_bound),axis=1)

	return matched_group

def most_bound(IDs, group_length, n_bound):

	split_bound = np.argmin(group_length>=n_bound)
	group_length = group_length[:split_bound]
	IDs = IDs[:np.sum(group_length)]
	index_bound = np.insert(np.cumsum(group_length)[:-1],0,0)[:, None] + np.arange(n_bound)

	return IDs[index_bound.flatten().astype(np.int64)]

def halo_matcher(sims, SN='033', n_bound = 50, tag='', output_dir='./'):

	'''Return an array of group numbers, which have been matched across sims.
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
	'''

	snaps = np.array([])
	snaps = np.append(snaps,SN)

	assert len(sims) > 1, 'Need at least 2 simulations to form a match'
	sims = np.array(sims)
	sim_ref = sims[0]
	sims = sims[1:]

	# Read in number of particles in the simulation
	n_part = E.readAttribute('SNAPSHOT', sim_ref, SN[0], '/Header/NumPart_Total')[1]

	if n_part > np.power(1024,3):
		int_type = np.int64
	else:
		int_type = np.int32

	catalogues = {}

	for SN_i in snaps:

		#Read in redshift of Snapshot
		redshift = np.around(E.readAttribute('SNAPSHOT', sim_ref, SN_i, '/Header/Redshift'), 3)

		# Check to see if this halo catalogue already exists:
		try:
			halo_catalogue = np.load('{}Halo_Catalogue_{}_z_{}.npy'.format(output_dir, tag, str(redshift).replace('.','p')))
			catalogues[SN_i] = halo_catalogue
			continue
		except(FileNotFoundError):
				if sims is None:
					raise ValueError('Please provide a valid tag, or a list of simulation directories.')
				else:
					print('No halo catalogue found with tag: {}, at redshift: {}'.format(tag, redshift))
					print('Generating new halo catalogue')

		ref_ordered_GNs = np.ones(n_part, dtype=int_type) * -1

		GNs = np.abs(E.readArray('PARTDATA', sim_ref, SN_i, '/PartType1/GroupNumber', verbose=False)) -1
		IDs = E.readArray('PARTDATA', sim_ref, SN_i, '/PartType1/ParticleIDs', verbose=False) -1

		ref_ordered_GNs[IDs] = GNs

		ref_IDs = E.readArray('SUBFIND_PARTICLES', sim_ref, SN_i, '/IDs/ParticleID', verbose=False) - 1
		ref_group_length = E.readArray('SUBFIND_GROUP', sim_ref, SN_i, '/FOF/GroupLength', verbose=False)

		# Find halos which contain at least 50 particles in Sim1
		ref_group_numbers = np.where(ref_group_length >= n_bound)[0]

		# Make array of n_bound most bound particle in Sim1
		ref_most_bound = most_bound(ref_IDs, ref_group_length, n_bound)

		halo_catalogue = np.ones((np.max(ref_group_numbers)+1, len(sims)+1), dtype=int_type) * - 1
		halo_catalogue[:,0] = np.arange(0, np.max(ref_group_numbers)+1)

		for i, sim in enumerate(tqdm(sims)):
			# Match from Sim1 to Sim2
			# Create array to store current Group Numbers matching from Sim1 to Sim2
			temp_halo_catalogue_ref = np.ones((np.max(ref_group_numbers)+1, 2), dtype=int_type) * -1
			temp_halo_catalogue_ref[:,0] = np.arange(0,np.max(ref_group_numbers)+1)

			# Set up an array to be filled in with ordered group numbers
			match_ordered_GNs = np.ones(n_part, dtype=int_type) * -1

			# Read in the particle IDs and Group Numbers of Sim2
			match_GNs = np.abs(E.readArray('PARTDATA', sim, SN_i, '/PartType1/GroupNumber', verbose=False)) -1
			match_IDs = E.readArray('PARTDATA', sim, SN_i, '/PartType1/ParticleIDs', verbose=False) -1
			match_ordered_GNs[match_IDs] = match_GNs # Ordered by ID

			# Match halos from Sim1 to Sim2 and store matches in temporary catalogue
			matches = GN_match(match_ordered_GNs, ref_most_bound, n_bound)
			temp_halo_catalogue_ref[ref_group_numbers,1] = matches[:,0]

			# Now match from Sim2 to Sim1
			# First read in particle information for Sim2
			match_IDs = E.readArray('SUBFIND_PARTICLES', sim, SN_i, '/IDs/ParticleID', verbose=False) - 1
			match_group_length = E.readArray('SUBFIND_GROUP', sim, SN_i, '/FOF/GroupLength', verbose=False)

			# Find halos which contain at least 50 particles in Sim2
			match_group_numbers = np.where(match_group_length>=n_bound)[0]

			# Create array to store current Group numbers and matched Group Numbers from Sim2 to Sim1
			temp_halo_catalogue_match = np.ones((np.max(match_group_numbers) + 1, 2), dtype=int_type) * -1
			temp_halo_catalogue_match[:,1] = np.arange(0,np.max(match_group_numbers)+1)

			# Make n_bound most bound particle array for halos in Sim2
			match_most_bound = most_bound(match_IDs, match_group_length, n_bound)

			# Match halos from Sim2 to Sim1 using most bound particles
			bi_matches = GN_match(ref_ordered_GNs, match_most_bound, n_bound)

			# Store matches in temporary halo catalogue
			temp_halo_catalogue_match[match_group_numbers,0] = bi_matches[:,0]

			# Match between the halo catalogues of Sim1 and Sim2
			# Remove all halos which do not have a match straight away
			temp_halo_catalogue_ref = temp_halo_catalogue_ref[temp_halo_catalogue_ref[:,1] != -1]
			temp_halo_catalogue_match = temp_halo_catalogue_match[temp_halo_catalogue_match[:,0] != -1]

			# Now find which of the reference halos are in both sets of matched halos
			index_both = np.isin(temp_halo_catalogue_match[:,0], temp_halo_catalogue_ref[:,0])
			temp_halo_catalogue_match = temp_halo_catalogue_match[index_both]

			# Construct a common set of halos across the two simulations
			index_array = np.searchsorted(temp_halo_catalogue_ref[:,0], temp_halo_catalogue_match[:,0])
			temp_halo_catalogue_ref = temp_halo_catalogue_ref[index_array]
			true_match = temp_halo_catalogue_ref[:,1] == temp_halo_catalogue_match[:,1]
			temp_halo_catalogue = temp_halo_catalogue_ref[true_match]

			# Fill in the Halo catalogue with the matched set
			halo_catalogue[temp_halo_catalogue[:,0],i+1] = temp_halo_catalogue[:,1]

		halo_catalogue = np.delete(halo_catalogue, np.where(halo_catalogue==-1)[0],axis=0)
		np.save('{}Halo_Catalogue_{}_z_{}'.format(output_dir, tag, str(redshift).replace('.','p')), halo_catalogue)

		catalogues[SN_i] = halo_catalogue

	return catalogues


if __name__ == '__main__':
	SN = ['033', '027', '023']
	tag = 'L100N256'
	sim_path = '/path/to/sim/suite/'
	suite_tags = ['sim1','sim2','sim3','sim4','sim5']

	HC = halo_matcher(tag, SN=SN, sims = [sim_path+i+'/data/' for i in suite_tags], n_bound=50)
	print(HC[SN[0]])
