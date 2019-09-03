import numpy as np
import time
from tqdm import tqdm
import eagle3 as E
from scipy import stats

def GN_match(ordered_GNs,IDs_array,n_bound):

    '''Function to return the matching group number to a halo.'''

    matched_group, _ = stats.mode(ordered_GNs[IDs_array].reshape(-1,n_bound),axis=1)

    return matched_group

def most_bound(IDs, group_length, n_bound):

    '''Return an array containing the n_bound particles for all halos being matched.
    Arguments:
    IDs -- list of DM particle IDs that will be matched, in order of most bound to
           least bound.
    group_length -- total number of particles in each group.
    n_bound -- total number of particles that will be selected from the IDs array
               in order to match the halos.
    '''

    split_bound = np.argmin(group_length>=n_bound)
    group_length = group_length[:split_bound]
    IDs = IDs[:np.sum(group_length)]
    index_bound = np.insert(np.cumsum(group_length)[:-1],0,0)[:, None] + np.arange(n_bound)

    return IDs[index_bound.flatten().astype(np.int64)]

def halo_matcher(ref_group_numbers, ref_ordered_GNs, ref_most_bound, match_ordered_GNs, sim_match, SN_i, n_bound, verbose, redshift_tracker):

    '''Main function, called to return a matched halo catalogue from sim 1
    to sim 2.
    Arguments:
    ref_group_numbers -- group numbers from sim 1 for which a match is going
                         to be searched for in sim 2.
    ref_ordered_GNs -- numpy array with length n_part, filled in with the group
                       each particle from sim 1 belongs too.
    ref_most_bound --  numpy array of length n_bound*len(ref_group_numbers)
                       particle IDs are organised by binding energy from most to least
                       bound.
    match_ordered_GNs -- numpy array with length n_part, filled in with the group
                       each particle from sim 2 belongs too.
    sim_match -- simulation for which particles from sim 1 will be matched.
    SN_i -- snapshot of simulation matching too. This allows one to match back in redshift.
    n_bound -- number of most bound particles that will be used when matching halos. Default
               is set to 50.
    verbose -- output information on reading in particle data.
    redshift_tracker -- Flag, telling the code one is matching one simulation
                        backwards through redshift. This is needed as the way it
                        is set up, in order to minimise reading in time, the function
                        will return the matched_ordered_GNs array, which will then be
                        used as the ref_ordered_GNs for the next iteration.'''

    # Create array to store current Group Numbers matching from Sim1 to Sim2
    temp_halo_catalogue_ref = np.ones((np.max(ref_group_numbers)+1, 2), dtype=int_type) * -1
    temp_halo_catalogue_ref[:,0] = np.arange(0,np.max(ref_group_numbers)+1)

    # Read in the particle IDs and Group Numbers of Sim2
    match_GNs = np.abs(E.readArray('PARTDATA', sim_match, SN_i, '/PartType1/GroupNumber', verbose=verbose)) -1
    match_IDs = E.readArray('PARTDATA', sim_match, SN_i, '/PartType1/ParticleIDs', verbose=verbose) -1
    match_ordered_GNs[match_IDs] = match_GNs # Ordered by ID

    # Match halos from Sim1 to Sim2 and store matches in temporary catalogue
    matches = GN_match(match_ordered_GNs, ref_most_bound, n_bound)
    temp_halo_catalogue_ref[ref_group_numbers,1] = matches[:,0]

    # Now match from Sim2 to Sim1
    # First read in particle information for Sim2
    match_IDs = E.readArray('SUBFIND_PARTICLES', sim_match, SN_i, '/IDs/ParticleID', verbose=verbose) - 1
    match_group_length = E.readArray('SUBFIND_GROUP', sim_match, SN_i, '/FOF/GroupLength', verbose=verbose)

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

    if redshift_tracker is False:
        return temp_halo_catalogue
    elif redshift_tracker is True:
        return temp_halo_catalogue, match_ordered_GNs, match_IDs, match_group_length

def Halo_Matcher(sims, SN='033', n_bound = 50, tag='', output_dir='./', redshift_tracker = False, verbose=False):

    '''Return an array of group numbers, which have been matched across sims.
    Arguments:
    sims -- List of simulation directories to produce the matched halo catalogue for.

    Keyword arguments:
    SN -- snapshot tag required by readEagle when reading in simulation output. Default corresponds to
          redshift 0 for a BAHAMAS simulation. Note, if redshift_tracker is true. SN is expected to be
          of the form: [start_SN, end_SN], and will be used create a range of snaps to match across. I.e.
          if input was: ['025', '030'] will create redshift catalogue for halos across snaps:
          ['025', '026', '027', '028', '029', '030'].
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

    if redshift_tracker is False:
        # Read in number of particles in the simulation
        n_part = E.readAttribute('SNAPSHOT', sims[0], snaps[0], '/Header/NumPart_Total')[1]
    elif redshift_tracker is True:
        n_part = E.readAttribute('SNAPSHOT', sims, snaps[0], '/Header/NumPart_Total')[1]

    global int_type
    if n_part > np.power(1024,3):
    	int_type = np.int64
    else:
    	int_type = np.int32


    if redshift_tracker is False:
        catalogues = {}

        assert len(sims) > 1, 'Need at least 2 simulations to form a match'
        sims = np.array(sims)
        sim_ref = sims[0]
        sims = sims[1:]

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
                    raise ValueError('Please provide a list of simulation directories.')
                else:
                    print('No halo catalogue found with tag: {}, at redshift: {}'.format(tag, redshift))
                    print('Generating new halo catalogue')

            ref_ordered_GNs = np.ones(n_part, dtype=int_type) * -1
            # Set up an array to be filled in with ordered group numbers
            match_ordered_GNs = np.ones(n_part, dtype=int_type) * -1

            GNs = np.abs(E.readArray('PARTDATA', sim_ref, SN_i, '/PartType1/GroupNumber', verbose=verbose)) -1
            IDs = E.readArray('PARTDATA', sim_ref, SN_i, '/PartType1/ParticleIDs', verbose=verbose) -1

            ref_ordered_GNs[IDs] = GNs

            ref_IDs = E.readArray('SUBFIND_PARTICLES', sim_ref, SN_i, '/IDs/ParticleID', verbose=verbose) - 1
            ref_group_length = E.readArray('SUBFIND_GROUP', sim_ref, SN_i, '/FOF/GroupLength', verbose=verbose)

            # Find halos which contain at least 50 particles in Sim1
            ref_group_numbers = np.where(ref_group_length >= n_bound)[0]

            # Make array of n_bound most bound particle in Sim1
            ref_most_bound = most_bound(ref_IDs, ref_group_length, n_bound)

            halo_catalogue = np.ones((np.max(ref_group_numbers)+1, len(sims)+1), dtype=int_type) * - 1
            halo_catalogue[:,0] = np.arange(0, np.max(ref_group_numbers)+1)

            for i, sim in enumerate(tqdm(sims)):

                temp_halo_catalogue = halo_matcher(ref_group_numbers, ref_ordered_GNs, \
                    ref_most_bound, match_ordered_GNs, sim, SN_i, n_bound, verbose, redshift_tracker)
                halo_catalogue[temp_halo_catalogue[:,0],i+1] = temp_halo_catalogue[:,1]

            halo_catalogue = np.delete(halo_catalogue, np.where(halo_catalogue==-1)[0],axis=0)
            np.save('{}Halo_Catalogue_{}_z_{}'.format(output_dir, tag, str(redshift).replace('.','p')), halo_catalogue)

        catalogues[SN_i] = halo_catalogue

        return catalogues

    elif redshift_tracker is True:
        assert len(snaps) == 2, 'redshift_tracker is True, please input a start and end snapshot \
                              in order to create a redshift catalogue'
        SN_ref = snaps[1] # will match to the end snapshot, so tracking back in redshift
        snaps = np.array(["%03d" %i for i in range(int(snaps[0]), int(snaps[1]))])[::-1]

        # Check to see if this halo catalogue already exists:
        try:
            redshift_catalogue = np.load('{}Redshift_Catalogue_{}_SN_{}-{}.npy'.format(output_dir, tag, snaps[0], snaps[-1]))
            return redshift_catalogue
        except(FileNotFoundError):
            print('No redshift catalogue found with tag: {}, for sim: {}'.format(tag, sims))
            print('Generating new redshift catalogue')

        for i, SN_i in enumerate(tqdm(snaps)):

            match_ordered_GNs = np.ones(n_part, dtype=int_type) * -1

            if i == 0:
                '''For the first element, need to define the reference IDs and GNs which are going
                to be used to make the matches. These correspond to the first snapshot.'''

                # Set up arrays to be filled in with ordered group numbers
                ref_ordered_GNs = np.ones(n_part, dtype=int_type) * -1

                # Read in the reference particles information, which are going to be matched to
                GNs = np.abs(E.readArray('PARTDATA', sims, SN_ref, '/PartType1/GroupNumber', verbose=verbose)) -1
                IDs = E.readArray('PARTDATA', sims, SN_ref, '/PartType1/ParticleIDs', verbose=verbose) -1
                ref_ordered_GNs[IDs] = GNs

                ''' Read in the reference particles information. This reads particles in so that they are ordered
                by their group number, i.e. biggest group to smallest group, and from most bound to least bound
                inside a group'''
                ref_IDs = E.readArray('SUBFIND_PARTICLES', sims, SN_ref, '/IDs/ParticleID', verbose=verbose) - 1
                ref_group_length = E.readArray('SUBFIND_GROUP', sims, SN_ref, '/FOF/GroupLength', verbose=verbose)

                # Find halos which contain at least 50 particles in Sim1
                ref_group_numbers = np.where(ref_group_length >= n_bound)[0]

                # Make array of n_bound most bound particle in Sim1
                ref_most_bound = most_bound(ref_IDs, ref_group_length, n_bound)

                ''' Set up the redshift catalogue, which will be updated upon each iteration.
                This will have length equal to the maximum group number in the reference simulation.'''
                redshift_catalogue = np.ones((np.max(ref_group_numbers)+1, len(snaps)+1), dtype=int_type) * - 1
                redshift_catalogue[:,0] = np.arange(0, np.max(ref_group_numbers)+1)

                temp_redshift_catalogue, ref_ordered_GNs, ref_IDs, ref_group_length = halo_matcher(ref_group_numbers, \
                    ref_ordered_GNs, ref_most_bound, match_ordered_GNs, sims, SN_i, n_bound, verbose, redshift_tracker)
                redshift_catalogue[temp_redshift_catalogue[:,0],i+1] = temp_redshift_catalogue[:,1]
                '''Note, this returns the matched_ordered_GNs array, and replaces the
                ref_ordered_GNs to be given to the matcher for next iteration'''
                continue

            '''if not first snapshot update the reference group numbers to groups
            successfully matched at this redshift.'''
            ref_group_numbers = redshift_catalogue[:,i]

            # Make array of n_bound most bound particle in Sim1
            ref_most_bound = most_bound(ref_IDs, ref_group_length, n_bound)

            # Now only extract the particles of the groups we matched from previous iteration
            index = ref_group_numbers[:, None]*50 + np.arange(n_bound)[None, :]
            ref_most_bound = np.concatenate(ref_most_bound[index])

            temp_redshift_catalogue, ref_ordered_GNs, ref_IDs, ref_group_length = halo_matcher(ref_group_numbers, \
                ref_ordered_GNs, ref_most_bound, match_ordered_GNs, sims, SN_i, n_bound, verbose, redshift_tracker)

            '''Now fill in the redshift catalogue. To do this, need to find which halos have been matched
            and where in the catalogue they are positioned.'''
            group_numbers_cur = temp_redshift_catalogue[:,0]
            index = np.where(np.isin(ref_group_numbers, group_numbers_cur))[0]

            redshift_catalogue[index,i+1] = temp_redshift_catalogue[:,1]

        np.save('{}Redshift_Catalogue_{}_SN_{}-{}.npy'.format(output_dir, tag, snaps[0], snaps[-1]), redshift_catalogue)

    return redshift_catalogue


if __name__ == '__main__':
	SN = ['033', '027', '023']
	tag = 'L100N256'
	sim_path = '/path/to/sim/suite/'
	suite_tags = ['sim1','sim2','sim3','sim4','sim5']

	'''To run and match a reference set of halos to other simulations in a suite, a command
	similar to this would be used.'''
	HC = halo_matcher(sims = [sim_path+i+'/data/' for i in suite_tags], SN[0], n_bound=50, tag=tag)
	print(HC[SN[0]])
	
	'''You can also make a redshift catalogue for a halo, which will give you back the group number
	of the progenitor which contained the most of n_bound particles. This can be done via the following:'''
	sim_dirs = [sim_path+i+'/data/' for i in suite_tags]
    for i, sim in enumerate(tqdm(sim_dirs)):
	''' Here you need to enable the redshift_tracker flag, to tell the matcher you are matching back in redshift.
	You also need to give it a start and an end snapshot.'''
        HC = Halo_Matcher(sim, ['023','033'], n_bound=50, tag = tags[i],redshift_tracker=True)
