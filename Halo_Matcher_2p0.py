import numpy as np
import time
from tqdm import tqdm
import eagle3 as E
from scipy import stats
from sim_info import get_sim_info
from Sim_Tools import duplicate_remover

def GN_match(IDs_array):

	matched_Group, _ = stats.mode(Ordered_GNs[IDs_array].reshape(-1,50),axis=1)

	return matched_Group

def most_bound(IDs, Group_Length, GNs, N_bound):

	split_IDs = np.split(IDs, np.cumsum(Group_Length))[:-1]
	split_IDs = [split_IDs[GN] for GN in GNs]

	Group_Length = Group_Length[GNs]
	IDs_most_bound = np.zeros(len(Group_Length)*N_bound, dtype=np.int32)

	for j, ids in enumerate(tqdm(split_IDs)):

		index = j*N_bound
		IDs_most_bound[index:index+N_bound] = ids[:N_bound]

		del index

	del split_IDs
	del Group_Length

	return IDs_most_bound

def Halo_Matcher(tag, SN='033', N_bound = 50, N_part=1024, Sims=None, ref_sim=None, HYDRO=False):

	output_dir = '/home/astsstaf/Documents/Work/Simulations/Halo_Catalogues/Halo_Matcher_2p0/'

	BAHAMAS = 'BAHAMAS' in tag

	if BAHAMAS is True:

		print('Looks like this is a BAHAMAS suite of sims, using different Snapshot notation')

		Snaps = ["%03d"%i for i in np.arange(18,33)][::-1] #snapshots out to z = 3
		redshifts = np.array([3.0,2.75,2.5,2.25,2.0,1.75,1.5,1.25,1,0.75,0.5,0.375,0.25,0.125,0])[::-1]
	else:
		# # Set up a dictionary for snapshots
		Snaps = ["%03d"%i for i in np.arange(19,34)][::-1] #snapshots out to z = 3
		redshifts = np.array([3.0,2.75,2.5,2.25,2.0,1.75,1.5,1.25,1,0.75,0.5,0.375,0.25,0.125,0])[::-1]

	Snapshots = {}
	for val_i, key in enumerate(Snaps):
		Snapshots[key] = str(redshifts[val_i])

	if Sims is None:
		try:
			print('Simulation tag is: {}'.format(tag))
			Sims, SN, colours, sim_labels, plot_labels = get_sim_info(tag)
		except:
			raise ValueError('Please provide a valid tag, or a list of Simulation directories.')

	if Sims is not None and ref_sim is None:

		assert len(Sims) > 1, 'Need at least 2 simulations to form a match'

		ref_sim = Sims[0]
		print('Note: a reference simulation has not been specified. Using first simulation in list.')
		Sims = Sims[1:]

	# Check to see if this halo catalogue already exists:
	data_avail = False
	redshift = Snapshots[SN]
	try:
		Halo_Catalogue = np.load('{}Halo_Catalogue_{}_z_{}.npy'.format(output_dir, tag, redshift.replace('.','p')))
		return Halo_Catalogue
	except:
		data_avail = False

	if data_avail is False:

		h = E.readAttribute('PARTDATA', ref_sim, SN, '/Header/HubbleParam')
		M_p = E.readAttribute('PARTDATA', ref_sim, SN, '/Header/MassTable')[1] * 1e10 * h**(-1)
		M = (N_bound + (0.1*N_bound))*M_p

		ref_Ordered_GNs = np.ones(N_part**3, dtype=np.int32) * -1

		GNs = np.abs(E.readArray("PARTDATA", ref_sim, SN, '/PartType1/GroupNumber', verbose=False)) -1
		IDs = E.readArray("PARTDATA", ref_sim, SN, '/PartType1/ParticleIDs', verbose=False) -1

		ref_Ordered_GNs[IDs] = GNs
		del GNs
		del IDs

		ref_IDs = E.readArray('SUBFIND_PARTICLES', ref_sim, SN, '/IDs/ParticleID', verbose=False) - 1
		ref_Group_Length = E.readArray('SUBFIND_GROUP',ref_sim, SN, '/FOF/GroupLength', verbose=False)
		ref_M200 = E.readArray('SUBFIND_GROUP', ref_sim, SN, '/FOF/Group_M_Crit200', verbose=False) * 1e10

		index_M200 = np.where(ref_M200>=M)[0]
		ref_Group_Numbers = index_M200
		del index_M200

		# Make 50 most bound particle array for all halos above mass cut in reference sim
		ref_most_bound = most_bound(ref_IDs, ref_Group_Length, ref_Group_Numbers, N_bound)

		Halo_Catalogue = np.ones((np.max(ref_Group_Numbers)+1, len(Sims)+1), dtype=np.int32) * - 1
		Halo_Catalogue[:,0] = np.arange(0, np.max(ref_Group_Numbers)+1)

		for i, sim in enumerate(tqdm(Sims)):

			# Create array to store current Group numbers matching from Sim1 to Sim2
			Temp_Halo_Catalogue_ref = np.ones((np.max(ref_Group_Numbers)+1, 2), dtype=np.int32) * -1
			Temp_Halo_Catalogue_ref[:,0] = np.arange(0,np.max(ref_Group_Numbers)+1)

			# Set up an array to be filled in with ordered group numbers
			match_Ordered_GNs = np.ones(N_part**3, dtype=np.int32) * -1

			# Read in the particle IDs and Group Numbers of second sim to and fill them in to waiting array
			global Ordered_GNs
			match_GNs = np.abs(E.readArray("PARTDATA", sim, SN, '/PartType1/GroupNumber', verbose=False)) -1
			match_IDs = E.readArray("PARTDATA", sim, SN, '/PartType1/ParticleIDs', verbose=False) -1
			match_Ordered_GNs[match_IDs] = match_GNs # Ordered by ID
			Ordered_GNs = match_Ordered_GNs

			# Match halos from Sim 1 to Sim 2 and store matches in temporary catalogue
			Matches = GN_match(ref_most_bound)
			Temp_Halo_Catalogue_ref[ref_Group_Numbers,1] = Matches[:,0]

			del match_GNs
			del match_IDs
			del match_Ordered_GNs
			del Ordered_GNs
			del Matches

			#####################################################################################
			# Now match from Sim2 to Sim1

			# Read in particle information
			match_IDs = E.readArray('SUBFIND_PARTICLES', sim, SN, '/IDs/ParticleID', verbose=False) - 1
			match_Group_Length = E.readArray('SUBFIND_GROUP',sim, SN, '/FOF/GroupLength', verbose=False)
			match_M200 = E.readArray('SUBFIND_GROUP', sim, SN, '/FOF/Group_M_Crit200', verbose=False) * 1e10

			# Find halos above the minimum mass which will contain atleast 50 particles
			match_Group_Numbers = np.where(match_M200>=M)[0]

			# Create array to store current Group numbers and matched group numbers from Sim2 to Sim1
			Temp_Halo_Catalogue_match = np.ones((np.max(match_Group_Numbers) + 1, 2), dtype=np.int32) * -1
			Temp_Halo_Catalogue_match[:,1] = np.arange(0,np.max(match_Group_Numbers)+1)

			# Make 50 most bound particle array for halos in match simulation
			match_most_bound = most_bound(match_IDs, match_Group_Length, match_Group_Numbers, N_bound)

			# Match halos from Sim2 to ref Sim using most bound particles
			Ordered_GNs = ref_Ordered_GNs
			Bi_Matches = GN_match(match_most_bound)

			# Store matches in temporary halo catalogue
			Temp_Halo_Catalogue_match[match_Group_Numbers,0] = Bi_Matches[:,0]

			# Remove all halos which do not have a match straight away
			Temp_Halo_Catalogue_ref = Temp_Halo_Catalogue_ref[Temp_Halo_Catalogue_ref[:,1] != -1]
			Temp_Halo_Catalogue_match = Temp_Halo_Catalogue_match[Temp_Halo_Catalogue_match[:,0] != -1]

			#Sort Matched halos by halo number matched to in reference simulation, thus aligning all duplicate matches
			index_sort = np.argsort(Temp_Halo_Catalogue_match[:,0])
			Temp_Halo_Catalogue_match = Temp_Halo_Catalogue_match[index_sort]

			#Remove these duplicates from the array to be dealt with later
			arr, duplicate_matches, index_dupes = duplicate_remover(Temp_Halo_Catalogue_match[:,0], return_dup=True, return_index=True)
			duplicates_match = Temp_Halo_Catalogue_match[index_dupes]

			#Find these matched halos in the reference matching
			dup_index_ref = np.where(np.isin(Temp_Halo_Catalogue_ref[:,0], Temp_Halo_Catalogue_match[index_dupes,0]))[0]
			duplicates_ref = Temp_Halo_Catalogue_ref[dup_index_ref]

			#Now delete from the two matched sets, duplicates from the match, and the corresponding halos in the ref match
			Temp_Halo_Catalogue_ref = np.delete(Temp_Halo_Catalogue_ref, dup_index_ref, axis=0)
			Temp_Halo_Catalogue_match = np.delete(Temp_Halo_Catalogue_match, index_dupes, axis=0)
			del index_dupes
			del dup_index_ref
			del arr
			del index_sort

			# Now find which of the reference halos are in both sets of matched halos
			index_both = np.isin(Temp_Halo_Catalogue_ref[:,0], Temp_Halo_Catalogue_match[:,0])
			Temp_Halo_Catalogue_ref = Temp_Halo_Catalogue_ref[index_both]
			index_both = np.isin(Temp_Halo_Catalogue_match[:,0], Temp_Halo_Catalogue_ref[:,0])
			Temp_Halo_Catalogue_match = Temp_Halo_Catalogue_match[index_both]

			# Find where these arrays are now different and thus have not matched to the same halo Discount these
			diff = np.subtract(Temp_Halo_Catalogue_ref, Temp_Halo_Catalogue_match)
			index_diff = np.where(diff[:,1] == 0)[0]

			#Create new catalogue with the complete list
			Temp_Halo_Catalogue = Temp_Halo_Catalogue_ref[index_diff]
			del diff
			del index_diff
			del index_both

			# Now deal with the removed duplicates, set up a new reference temp catalogue, so it has same length as duplicates array
			index_dupes = np.isin(duplicates_match[:,0], duplicates_ref[:,0])
			duplicates_match = duplicates_match[index_dupes]
			index_dupes = np.searchsorted(duplicates_ref[:,0], duplicates_match[:,0])
			duplicates_ref = duplicates_ref[index_dupes]

			# Find the indeces where the groups are matched the same bijectively
			true_match = duplicates_ref[:,1] == duplicates_match[:,1]
			duplicates = duplicates_ref[true_match]

			# Update the Temporary halo catalogue to include all halos
			Temp_Halo_Catalogue = np.vstack((Temp_Halo_Catalogue, duplicates))

			# Fill in the Halo catalogue with the matched set
			Halo_Catalogue[Temp_Halo_Catalogue[:,0],i+1] = Temp_Halo_Catalogue[:,1]

			del match_IDs
			del match_Group_Length
			del match_M200
			del match_Group_Numbers
			del match_most_bound
			del Ordered_GNs
			del Bi_Matches
			del Temp_Halo_Catalogue_match
			del Temp_Halo_Catalogue_ref
			del Temp_Halo_Catalogue
			del duplicates
			del true_match
			del duplicates_ref
			del index_dupes

		Halo_Catalogue = np.delete(Halo_Catalogue, np.where(Halo_Catalogue==-1)[0],axis=0)
		np.save('{}Halo_Catalogue_{}_z_{}'.format(output_dir, tag, redshift.replace('.','p')), Halo_Catalogue)

		return Halo_Catalogue
if __name__ == '__main__':

	HC = Halo_Matcher('nrun', N_bound=50)