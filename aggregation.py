"""Functions for aggregating and subsetting subject and task data."""
from configs import *


def get_relinds(cp, endpoint=4):
    """
    Assigns each trial two indices, locating it relative to two proximal CPs.

    Parameters:
        cp: Array of boolean changepoint indicators

    Returns:
        pre: Array of indices relative to next changepoint.
        pst: Array of indices relative to previous changepoint.
    """
    pre = np.full_like(cp, np.nan, dtype=float)
    pst = np.full_like(cp, np.nan, dtype=float)

    # Change-point indices themselves
    change_indices = np.where(cp == 1)[0]

    # Determine how far each peri-cp trial is before and after each CP
    for idx in change_indices:

        # Every trial before and at a CP gets a pre index
        pre[idx-1:idx+1] = [-1, 0]

        # Every trial after a CP gets a post index (posts never overlap CP)
        end = min(idx + endpoint, len(cp))
        pst[idx:end] = np.arange(0, endpoint)[:end-idx]

    return pre, pst


def align_trials(task, endpoint=4):
    """Builds a matrix assigning each CP (row) a vector of proximal trial indices (columns)"""

    # Peri-changepoint trial positions to consider
    pcp_trials = np.arange(-1, endpoint)

    # Get pre/post indices for each trial
    pre, pst = get_relinds(task.cp, endpoint=endpoint)

    # Count changepoints (where pre == 0)
    n_cpts = np.sum(pre == 0)

    # Build index matrix
    aligned_inds = np.full((n_cpts, len(pcp_trials)), np.nan)
    row = -1

    for t in range(len(pre)):
        pre_ind = pre[t]
        pst_ind = pst[t]

        is_pre = not np.isnan(pre_ind)
        is_pst = not np.isnan(pst_ind)

        # Check post first, since we'll increment row if pre
        if is_pst:
            aligned_inds[row, int(pst_ind)+1] = t

        # If it's a pre-ind, we'll need to increment row
        if is_pre:
            if int(pre_ind) == -1:
                row += 1
            aligned_inds[row, int(pre_ind)+1] = t

    return aligned_inds, pcp_trials


def subset(subj_orig, task_orig, inds):
    """Subset one subject and one task to specific trial indices."""
    from subjects import Subject, Responses, Beliefs
    from tasks import ChangepointTask

    # Create new subject with subsetted responses and beliefs
    responses = Responses(len(inds))
    for attr in vars(subj_orig.responses):
        val = getattr(subj_orig.responses, attr)
        if isinstance(val, np.ndarray):
            setattr(responses, attr, val[inds])

    beliefs = Beliefs(len(inds))
    for attr in vars(subj_orig.beliefs):
        val = getattr(subj_orig.beliefs, attr)
        if isinstance(val, np.ndarray):
            setattr(beliefs, attr, val[inds])

    subj = Subject(responses=responses, beliefs=beliefs)

    # Create new task with subsetted trial arrays
    task = ChangepointTask()
    task.ntrials = len(inds)
    for attr in vars(task_orig):
        val = getattr(task_orig, attr)
        if isinstance(val, np.ndarray) and len(val) == task_orig.ntrials:
            setattr(task, attr, val[inds])

    # Always pretend subdivision produces a new block
    task.new_blk[0] = 1

    # Never count first trial as a CP
    task.cp[0] = 0

    # If inds are not continuous, set item after jump to new block
    is_jump = np.where(np.array(inds[1:]) != np.array(inds[0:-1]) + 1)[0]+1
    task.new_blk[is_jump] = 1
    task.cp[is_jump] = 0

    return subj, task


def split_within_subjects(subjs, tasks):
    """Iterate over subject-task pairs and split each in half internally."""

    # Initialize split halves
    subjs_a, subjs_b, tasks_a, tasks_b = [], [], [], []

    # Iterate through subjects and tasks
    for i in range(len(subjs)):

        # Find number of trials and split index
        ntrials = tasks[i].ntrials
        split_ind = np.random.randint(ntrials)

        inds_a = np.arange(split_ind - np.floor(ntrials/2), split_ind).astype(int)
        inds_b = np.arange(split_ind, split_ind + np.floor(ntrials/2)).astype(int)

        # Wrap any indices beyond end
        inds_b[inds_b >= ntrials] -= ntrials

        # Wrap any indices earlier than start
        inds_a[inds_a < 0] += ntrials

        # Sort to correct order
        inds_a = np.sort(inds_a)
        inds_a = np.sort(inds_a)

        # Split each half
        subj_a, task_a = subset(subjs[i], tasks[i], inds_a)
        subj_b, task_b = subset(subjs[i], tasks[i], inds_b)

        # Append results to lists
        subjs_a.append(subj_a)
        subjs_b.append(subj_b)
        tasks_a.append(task_a)
        tasks_b.append(task_b)

    return subjs_a, subjs_b, tasks_a, tasks_b


def subset_within_subjects(subjs, tasks, frac=0.5):
    """Iterate over subject-task pairs and subset each to a fixed fraction of random trials."""

    # Initialize split halves
    subjs_a, tasks_a = [], []

    # Iterate through subjects and tasks
    for i in range(len(subjs)):

        # Find number of trials and split index
        ntrials = tasks[i].ntrials
        nselect = int(np.floor(ntrials*frac))
        ind_beg = np.random.randint(ntrials-nselect)

        # Subset
        inds = range(ind_beg, ind_beg + nselect)
        subj_a, task_a = subset(subjs[i], tasks[i], inds)

        # Append results to lists
        subjs_a.append(subj_a)
        tasks_a.append(task_a)

    return subjs_a, tasks_a


def subset_between_subjects(subjs, tasks, frac=0.5):
    """Split paired lists of subjects and tasks into two random subsets."""

    # Initialize split halves
    subjs_a, tasks_a = [], []

    # Find number of trials and split index
    nsubjs = len(subjs)
    indices = np.random.permutation(nsubjs)
    nselect = int(np.floor(nsubjs*frac))

    # Subset
    inds_a = indices[0:nselect]
    inds_b = indices[nselect:]

    subjs_a = [subjs[i] for i in inds_a]
    tasks_a = [tasks[i] for i in inds_a]

    subjs_b = [subjs[i] for i in inds_b]
    tasks_b = [tasks[i] for i in inds_b]

    return subjs_a, subjs_b, tasks_a, tasks_b
