from configs import *
from changepoint.subjects import subject_from_experiment
from changepoint.tasks import ChangepointTask

def concatenate_blocks(subj):
    '''Concatenates blocks for each subject in the list.'''

    # Get fields to concatenate
    behav_flds = list(subj[0][0]['behaviour'].keys())
    state_flds = list(subj[0][0]['cpt']['state'].keys())
    param_flds = list(subj[0][0]['cpt']['params'].keys())

    # Loop over subjects, copying data
    for i, subject in enumerate(subj):

        # Initialize with first block values
        sdata = {'behaviour': {}, 'cpt': {'state': {}, 'params': {}}}

        for f in behav_flds:
            sdata['behaviour'][f] = subject[0]['behaviour'][f]
        for f in state_flds:
            sdata['cpt']['state'][f] = subject[0]['cpt']['state'][f]
        for f in param_flds:
            sdata['cpt']['params'][f] = subject[0]['cpt']['params'][f]

        # Concatenate remaining blocks
        for j in range(1, len(subject)):

            # Concatenate behaviour fields
            for f in behav_flds:
                sdata['behaviour'][f] = np.concatenate((sdata['behaviour'][f], subject[j]['behaviour'][f]))

            # Concatenate state fields
            for f in state_flds:
                sdata['cpt']['state'][f] = np.concatenate((sdata['cpt']['state'][f], subject[j]['cpt']['state'][f]))

            # Concatenate param fields that are arrays, keep scalars from first block
            for f in param_flds:
                val = sdata['cpt']['params'][f]
                if isinstance(val, np.ndarray):
                    sdata['cpt']['params'][f] = np.concatenate((val, subject[j]['cpt']['params'][f]))

        # Add ntrials to params
        sdata['cpt']['params']['ntrials'] = len(sdata['cpt']['state']['obs'])

        # Only modify subject data if completed properly
        subj[i] = sdata

    return subj


def read_experiment_matfile(file = 'McGuireNassar2014data.csv', catblks = True):
    """
    Produces standardized subject data format from .mat files.

    Parameters:
        file (str)    - Specifies .mat file to read and process. Either 'MN', 'JN', or 'PP'.
        catblks (int) - Concatenate blocks? If 1, returns single sequences for each output field.

    Returns:
        subj (list)   - List of subject dictionaries with fields pred, obs, pe, update, cpt.
                        Field 'cpt' is itself a dictionary with fields state, noise_sd, and hazard.
                        If catblks is true, this is simplified to a single list of subjects.
    """
    subj = []

    # Select which data to process
    if file == 'McGuireNassar2014data.csv' or file == 'MN':
        data = pd.read_csv(SUBJ_DATA_DIR + SUBJ_DATA_FILE)

        # Get subject numbers & initialize subjects array
        snums = np.unique(data['subjNum'])

        # Subject will be a list of blocks
        subj = [[] for _ in range(len(snums))]

        # Extract subject data
        for i in snums:

            # Subject mask into full data
            smsk = data['subjNum'] == i

            # List of blocks subject performed
            blks = np.unique(data['blkNum'][smsk])

            # Loop over blocks
            for j in blks:
                bmsk = data['blkNum'] == j
                msk = bmsk & smsk

                # Temporary home for subject data
                cs = {}

                # Extract predictions and updates
                pred   = data['currentPrediction'][msk].to_numpy()
                update = data['currentUpdate'    ][msk].to_numpy()
                obs    = data['currentOutcome'   ][msk].to_numpy()

                # Compute prediction errors and re-index updates
                pe = obs - pred
                update[:-1] = update[1:]
                update[-1] = 0

                # Try to assess joystick failures
                tol = 30
                lb = np.append(np.nan, np.minimum(pred[:-1], obs[:-1]) - tol)
                ub = np.append(np.nan, np.maximum(pred[:-1], obs[:-1]) + tol)
                jsfail = (pred < lb) | (pred > ub)

                # Package behaviour
                cs['behaviour'] = {
                    'pred':   pred,
                    'pe':     pe,
                    'update': update,
                    'jsfail': jsfail,
                }

                # Extract the changepoint task that they saw
                cs['cpt'] = {
                    'state': {
                        'obs':     obs,
                        'state':   data['currentMean'   ][msk].to_numpy(),
                        'cp':      data['isChangeTrial' ][msk].to_numpy(),
                        'new_blk': np.eye(sum(msk), 1).flatten(),
                    },
                    'params': {
                        'noise_sd': data['currentStd'    ][msk].to_numpy(),
                        'hazard':   np.ones(sum(msk)) * data['currentHazard'][np.where(msk)[0][0]],
                        'bnds_ls':  [25, 275],
                        'bnds_obs': [0, 300],
                    }
                }

                # Package
                subj[int(i) - 1].append(cs)

    elif file == 'JN':
        data = sp.io.loadmat('./data/mat-files/jNeuroBehav_toSend.mat')['jNeuroBehavData']

        # Each session is a subject - blocks are "within session".
        snums = np.unique(data['session'][0][0])
        subj = [[] for _ in range(len(snums))]

        for i in snums:
            smsk = data['session'][0][0] == i
            blks = np.unique(data['Block'][0][0][smsk])

            for j in blks:
                bmsk = data['Block'][0][0] == j
                msk = (smsk & bmsk).flatten()

                # Temporary home for current subject data
                cs = {}

                # Extract predictions and observations
                pred = data['Prediction'][0][0][msk].flatten().astype(float)
                obs  = data['outcome'][0][0][msk].flatten().astype(float)

                # Compute prediction errors and updates
                pe     = obs - pred
                update = np.append(np.diff(pred), 0)

                # Package behaviour
                cs['behaviour'] = {
                    'pred':   pred,
                    'pe':     pe,
                    'update': update,
                }

                # CPT parameters
                cs['cpt'] = {
                    'state': {
                        'obs':     obs,
                        'state':   data['distMean'][0][0][msk].flatten().astype(float),
                        'new_blk': np.eye(sum(msk), 1).flatten().astype(float),
                        'cp':      np.append(0, np.diff(data['distMean'][0][0][msk].flatten()) != 0).astype(int).astype(float),
                    },
                    'params': {
                        'noise_sd': data['standDev'][0][0][msk].flatten().astype(float),
                        'hazard':   np.ones(sum(msk)) * 0.125,
                        'bnds_ls':  [25, 275],
                        'bnds_obs': [0, 300],
                    }
                }

                # Package into array
                #subj[int(i) - 1].setdefault('blk', {})[int(j)] = cs
                subj[int(i) - 1].append(cs)

    elif file == 'PP':
        raw = sp.io.loadmat('./data/mat-files/pupilPaperBehavData_toSend.mat')['pupilBehavData']
        data = {name: raw[name][0, 0] for name in raw.dtype.names}

        # Get subject numbers & initialize subjects array
        snums = np.unique(data['trialNumber'][:, 1])
        subj = [[] for _ in range(len(snums))]

        for i in snums:
            smsk = data['trialNumber'][:, 1] == i
            trls = data['trialNumber'][smsk, 0].astype(int)

            # Find block boundaries where trial numbers decrease
            resets = np.where(np.diff(trls) < 0)[0] + 1
            blk_starts = np.concatenate([[0], resets])
            blk_ends = np.concatenate([resets, [len(trls)]])

            for b in range(len(blk_starts)):
                msk = np.zeros_like(smsk, dtype=bool)
                subj_indices = np.where(smsk)[0][blk_starts[b]:blk_ends[b]]
                msk[subj_indices] = True

                # Temporary home for subject data
                cs = {}

                # Extract predictions and observations
                pred = data['prediction'][msk, 0].astype(float)
                obs  = data['outcome'][msk, 0].astype(float)

                # Compute prediction errors and updates
                pe     = obs - pred
                update = np.append(np.diff(pred), 0)

                # Package behaviour
                cs['behaviour'] = {
                    'pred':   pred,
                    'pe':     pe,
                    'update': update,
                }

                # Compute changepoints from state changes
                cp = np.append(0, np.diff(data['mean'][msk, 0]) != 0).astype(float)
                new_blk = np.zeros(sum(msk))
                new_blk[0] = 1.0

                cs['cpt'] = {
                    'state': {
                        'obs':     obs,
                        'state':   data['mean'][msk, 0].astype(float),
                        'cp':      cp,
                        'new_blk': new_blk,
                    },
                    'params': {
                        'noise_sd': data['stdDev'][msk, 0].astype(float),
                        'hazard':   np.ones(sum(msk)) * np.sum(cp) / sum(msk),
                        'bnds_ls':  [25, 275],
                        'bnds_obs': [0, 300],
                    }
                }

                # Package
                subj[int(i) - 1].append(cs)

    # Concatenate blocks if requested
    if catblks:
        subj = concatenate_blocks(subj)

    return subj


def read_experiment(file=SUBJ_DATA_FILE, max_subj=None):
    """Read experimental data and return lists of Subject and ChangepointTask objects."""

    # Read raw data
    raw_data = read_experiment_matfile(file=file, catblks=True)

    # Limit subjects if requested
    if max_subj is not None:
        raw_data = raw_data[0:max_subj]

    # Build subjects and tasks lists
    subjects, tasks = [], []
    for data in raw_data:
        behav  = data['behaviour']
        cpt    = data['cpt']
        state  = cpt['state']
        params = cpt['params']

        # Create task
        task = ChangepointTask(state=state, params=params)
        tasks.append(task)

        # Create subject
        subj = subject_from_experiment(
            pred    = behav['pred'],
            pe      = behav['pe'],
            update  = behav['update'],
            obs     = state['obs'],
            new_blk = state['new_blk'],
            hazard  = params['hazard'],
            noise_sd = params['noise_sd'],
        )
        subjects.append(subj)

    return subjects, tasks
