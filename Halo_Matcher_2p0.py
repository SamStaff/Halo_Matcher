import numpy as np
import time
from tqdm import tqdm
import eagle as E
from scipy import stats

def GN_match(IDs_array,N_bound):

	matched_Group, _ = stats.mode(Ordered_GNs[IDs_array].reshape(-1,N_bound),axis=1)

	return matched_Group

def most_bound(IDs, Group_Length, N_bound):

	split_bound = np.argmin(Group_Length>=N_bound)
	Group_Length = Group_Length[:split_bound]
	IDs = IDs[:np.sum(Group_Length)]
	index_bound = int_type(np.insert(np.cumsum(Group_Length)[:-1],0,0)[:, None] + np.arange(N_bound))

	return IDs[index_bound.flatten()]

def Halo_Matcher(tag, SN='033', redshift=0.0, N_bound = 50, N_part=1024, Sims=None, ref_sim=None, output_dir='./'):

	if 'BAHAMAS' in tag:
		print('Looks like this is a BAHAMAS suite of sims, using different Snapshot notation')
		Snaps = ['%03d'%i for i in np.arange(18,33)][::-1] #snapshots out to z = 3
		redshifts = np.array([3.0,2.75,2.5,2.25,2.0,1.75,1.5,1.25,1,0.75,0.5,0.375,0.25,0.125,0])[::-1]
	else:
		Snaps = [SN]
		redshift = np.array([redshift])

	# Set up a dictionary for snapshots
	Snapshots = {}
	for val_i, key in enumerate(Snaps):
		Snapshots[key] = str(redshift[val_i])

	# Check to see if this halo catalogue already exists:
	redshift = Snapshots[SN]
	try:
		Halo_Catalogue = np.load('{}Halo_Catalogue_{}_z_{}.npy'.format(output_dir, tag, redshift.replace('.','p')))
		return Halo_Catalogue
	except(FileNotFoundError):
			if Sims is None:
				raise ValueError('Please provide a valid tag, or a list of simulation directories.')
			else:
				print('No halo catalogue found with tag:{}'.format(tag))
				print('Generating new halo catalogue')

	if Sims is not None and ref_sim is None:
		assert len(Sims) > 1, 'Need at least 2 simulations to form a match'
		ref_sim = Sims[0]
		print('Note: ref_sim was not specified. Using first simulation in list.')
		Sims = Sims[1:]

	if N_part > 1024:
		int_type = np.int64
	else:
		int_type = np.int32

	ref_Ordered_GNs = np.ones(N_part**3, dtype=int_type) * -1

	GNs = np.abs(E.readArray('PARTDATA', ref_sim, SN, '/PartType1/GroupNumber', verbose=False)) -1
	IDs = E.readArray('PARTDATA', ref_sim, SN, '/PartType1/ParticleIDs', verbose=False) -1

	ref_Ordered_GNs[IDs] = GNs

	ref_IDs = E.readArray('SUBFIND_PARTICLES', ref_sim, SN, '/IDs/ParticleID', verbose=False) - 1
	ref_Group_Length = E.readArray('SUBFIND_GROUP',ref_sim, SN, '/FOF/GroupLength', verbose=False)

	# Find halos which contain at least 50 particles in Sim1
	ref_Group_Numbers = np.where(ref_Group_Length >= N_bound)[0]

	# Make array of N_bound most bound particle in Sim1
	ref_most_bound = most_bound(ref_IDs, ref_Group_Length, N_bound)

	Halo_Catalogue = np.ones((np.max(ref_Group_Numbers)+1, len(Sims)+1), dtype=int_type) * - 1
	Halo_Catalogue[:,0] = np.arange(0, np.max(ref_Group_Numbers)+1)

	for i, sim in enumerate(tqdm(Sims)):
		# Match from Sim1 to Sim2
		# Create array to store current Group Numbers matching from Sim1 to Sim2
		Temp_Halo_Catalogue_ref = np.ones((np.max(ref_Group_Numbers)+1, 2), dtype=int_type) * -1
		Temp_Halo_Catalogue_ref[:,0] = np.arange(0,np.max(ref_Group_Numbers)+1)

		# Set up an array to be filled in with ordered group numbers
		match_Ordered_GNs = np.ones(N_part**3, dtype=int_type) * -1

		# Read in the particle IDs and Group Numbers of Sim2
		global Ordered_GNs
		match_GNs = np.abs(E.readArray('PARTDATA', sim, SN, '/PartType1/GroupNumber', verbose=False)) -1
		match_IDs = E.readArray('PARTDATA', sim, SN, '/PartType1/ParticleIDs', verbose=False) -1
		match_Ordered_GNs[match_IDs] = match_GNs # Ordered by ID
		Ordered_GNs = match_Ordered_GNs

		# Match halos from Sim1 to Sim2 and store matches in temporary catalogue
		Matches = GN_match(ref_most_bound, N_bound)
		Temp_Halo_Catalogue_ref[ref_Group_Numbers,1] = Matches[:,0]

		# Now match from Sim2 to Sim1
		# Read in particle information for Sim2
		match_IDs = E.readArray('SUBFIND_PARTICLES', sim, SN, '/IDs/ParticleID', verbose=False) - 1
		match_Group_Length = E.readArray('SUBFIND_GROUP',sim, SN, '/FOF/GroupLength', verbose=False)

		# Find halos which contain at least 50 particles in Sim2
		match_Group_Numbers = np.where(match_Group_Length>=N_bound)[0]

		# Create array to store current Group numbers and matched Group Numbers from Sim2 to Sim1
		Temp_Halo_Catalogue_match = np.ones((np.max(match_Group_Numbers) + 1, 2), dtype=int_type) * -1
		Temp_Halo_Catalogue_match[:,1] = np.arange(0,np.max(match_Group_Numbers)+1)

		# Make N_bound most bound particle array for halos in Sim2
		match_most_bound = most_bound(match_IDs, match_Group_Length, N_bound)

		# Match halos from Sim2 to Sim1 using most bound particles
		Ordered_GNs = ref_Ordered_GNs
		Bi_Matches = GN_match(match_most_bound, N_bound)

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
	np.save('{}Halo_Catalogue_{}_z_{}_test'.format(output_dir, tag, redshift.replace('.','p')), Halo_Catalogue)

	return Halo_Catalogue


if __name__ == '__main__':

	tag = 'test'
	HC = Halo_Matcher(tag, N_part=1024, N_bound=50)
	print(HC)