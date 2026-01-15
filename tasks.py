import numpy as np

class ChangepointTask:
    def __init__(self, params=None, state=None):
        """Container for changepoint task data."""

        # Parameter fields
        self.ntrials  = None
        self.noise_sd = None
        self.hazard   = None
        self.bnds_ls  = None
        self.bnds_obs = None

        # State fields
        self.obs     = None
        self.cp      = None
        self.state   = None
        self.new_blk = None
        self.slist   = None

        # Parameter inputs
        if params is not None:
            for key, val in params.items():
                if hasattr(self, key):
                    setattr(self, key, val)

        # State inputs
        if state is not None:
            for key, val in state.items():
                if hasattr(self, key):
                    setattr(self, key, val)

    def check_params(self):
        """Check that all necessary parameters are set."""
        required_params = ['ntrials', 'noise_sd', 'hazard', 'bnds_ls', 'bnds_obs']
        for param in required_params:
            if getattr(self, param) is None:
                raise ValueError(f"Missing required parameter: {param}")


def simulate_cpt(params):
    """Simulates a changepoint task using input parameters."""

    # Initialize task container
    task = ChangepointTask(params = params)

    # Check that params included all fields necessary
    task.check_params()

    # Set block flag
    task.new_blk = np.zeros(task.ntrials, dtype=int)
    task.new_blk[0] = 1

    # Interpret noise_sd
    if np.isscalar(task.noise_sd):
        task.noise_sd = np.ones(task.ntrials) * task.noise_sd

    # Get random changepoints
    task.cp = np.random.rand(task.ntrials) < task.hazard
    ncp = np.sum(task.cp)

    # Get latent states
    range_ls = task.bnds_ls[1] - task.bnds_ls[0]
    task.slist = np.random.rand(ncp + 1) * range_ls + task.bnds_ls[0]

    # Pre-allocation for observations and latent states
    task.obs = np.full(task.ntrials, np.nan)
    task.state = np.full(task.ntrials, np.nan)

    # Set observations and save latent state for each trial
    snum = 0
    for i in range(task.ntrials):
        
        # Set new block if noise_sd changes
        if i > 0:
            if task.noise_sd[i] != task.noise_sd[i-1]:
                task.new_blk[i] = 1

        # Update latent state if changepoint and set observation
        snum += task.cp[i]
        task.obs[i] = np.round(np.random.normal(task.slist[snum], task.noise_sd[i]))
        task.state[i] = task.slist[snum]

    # Bound observations
    task.obs[task.obs < task.bnds_obs[0]] = task.bnds_obs[0]
    task.obs[task.obs > task.bnds_obs[1]] = task.bnds_obs[1]

    return task


