from configs import *

DEFAULT_PARAMS_SUBJ = {
    'mix':       1.0,
    'beta_pe':   0.0,
    'beta_cpp':  1.0,
    'beta_ru':   1.0,
    'hazard':    0.1,
    'noise_sd': 10.0,
    'loc':       0.0,
    'unc':       1.0,
    'drift':     0.0,
    'init_state_est': 150.0,
    'init_runlen_est': 10.0,
    'noise_sd_update': 2.0,
    'limit_updates': True,
    'clip': True
}


class Responses:
    """Container for subject responses, prediction errors, updates, and learning rates."""
    def __init__(self, n_trials):
        self.pred     = np.full(n_trials + 1, np.nan)
        self.pe       = np.full(n_trials, np.nan)
        self.update   = np.full(n_trials, np.nan)
        self.lr       = np.full(n_trials, np.nan)

class Beliefs:
    """Container for storing beliefs about latent states, uncertainties, and change-point probabilities."""
    def __init__(self, n_trials):
        # Estimated subject beliefs
        self.runlen   = np.full(n_trials + 1, np.nan)
        self.relunc   = np.full(n_trials, np.nan)
        self.cpp      = np.full(n_trials, np.nan)
        self.obs_sd   = np.full(n_trials, np.nan)
        self.state_sd = np.full(n_trials, np.nan)
        self.model_lr = np.full(n_trials, np.nan)

class Subject:
    """Behavior from a real or simulated subject, with inference and simulation methods."""
    def __init__(self, responses=None, beliefs=None):
        self.beliefs   = beliefs
        self.responses = responses


def subject_from_experiment(pred, pe, update, obs, new_blk, hazard, noise_sd, params=None):
    """Create a Subject instance from imported experimental data."""
    # Use default params if not provided
    if params is None:
        params = DEFAULT_PARAMS_SUBJ

    # Initialize subject
    subj = Subject()

    # Set responses from experimental data
    subj.responses = set_responses(pred, pe, update)

    # Infer beliefs
    subj.beliefs = get_beliefs(subj.responses, obs, new_blk, hazard, noise_sd, params=params)

    return subj


# Method for simulating a subject's responses
def simulate_subject(subj, obs, params=DEFAULT_PARAMS_SUBJ):
    """Performs approximate Bayesian latent state inference for the change-point task."""
    n_trials = len(obs)

    # Initialize beliefs and responses
    responses = Responses(n_trials)
    beliefs = Beliefs(n_trials)

    # Initialize
    responses.pred[0] = params['init_state_est']
    beliefs.runlen[0] = params['init_runlen_est']
    
    # Bounds on location of latent state (and observations)
    bnds = [0, 300]

    # Convert noise SD to variance
    noise_var = params['noise_sd'] ** 2

    # If noise_sd is a scalar, convert to vector
    if np.isscalar(noise_var):
        noise_var = np.full(n_trials, noise_var)

    # If hazard rate is a scalar, convert to vector
    hazard = params['hazard']
    if np.isscalar(hazard):
        hazard = np.full(n_trials, hazard)

    # Loop through the observations, making sequential predictions
    for t in range(n_trials):
        # Part 1: get the expected distribution of observations

        # Latent state uncertainty without drift
        static_var = noise_var[t] / beliefs.runlen[t]

        # Latent state uncertainty considering drift
        beliefs.state_sd[t] = np.sqrt(static_var + params['drift'])

        # Updated belief in how long runs should last
        beliefs.runlen[t] = noise_var[t] / beliefs.state_sd[t]**2

        # Uncertainty on the next observation
        beliefs.obs_sd[t] = np.sqrt(noise_var[t] + beliefs.state_sd[t]**2)

        # Part 2: calculate probability of latent state change
        beliefs.cpp[t] = get_cpp(obs[t], responses.pred[t], beliefs.obs_sd[t], bnds, hazard[t], mix=params['mix'])

        # Part 3: Update belief about mean

        # Now run_len_est can be really small if there is a big drift...
        beliefs.relunc[t] = 1 / (beliefs.runlen[t] + 1)

        # Find learning rate
        responses.lr[t] = get_learning_rate(
            beliefs, 
            t,
            bpe  = params['beta_pe'],
            bcpp = params['beta_cpp'],
            bru  = params['beta_ru'],
            clip = params['clip'],
            noise_sd = params['noise_sd_update']
        )

        # Set state prediction error
        responses.pe[t] = obs[t] - responses.pred[t]

        # Update state estimate
        noise_update = np.random.normal(0, params['noise_sd_update'])
        responses.update[t] = responses.lr[t] * responses.pe[t] + noise_update
        responses.pred[t + 1] = responses.pred[t] + responses.update[t]

        # Apply limits if task has them
        if params['limit_updates']:
            min_pred = min(responses.pred[t], obs[t])
            max_pred = max(responses.pred[t], obs[t])
            responses.pred[t + 1] = np.clip(responses.pred[t + 1], min_pred, max_pred)

        # Part 4: Update run length
        beliefs.runlen[t + 1] = get_runlen(obs[t], responses.pred[t], responses.pe[t], beliefs.cpp[t], beliefs.relunc[t], beliefs.runlen[t], noise_var[t])

    # Lots of calcs are easier without the final runlen/relunc number
    beliefs.runlen = beliefs.runlen[:-1]

    # Save beliefs and responses to subject
    subj.beliefs   = beliefs
    subj.responses = responses

    return subj


def set_responses(pred, pe, update):
    """Save responses for the subject from imported experimental data."""
    
    # Initialize responses data class
    responses = Responses(len(pred))

    # Set individual response fields
    responses.pred   = pred
    responses.pe     = pe
    responses.update = update
    
    # Find any locations where pe is zero and convert to 1
    augmented_pe = responses.pe.copy()
    augmented_pe[augmented_pe == 0] = 1.0

    # Compute raw learning rate - might encounter divide by zero        
    responses.lr = responses.update/augmented_pe

    # Clip any wrong (unstably estimated) learning rates
    bad_inds = np.abs(responses.lr) > 5

    # Set a derived lr field with nan for bad values
    responses.lr[bad_inds] = np.nan

    return responses


# Method for inferring beliefs from real subject data
def get_beliefs(responses, obs, new_blk, hazard, noise_sd, params=None, init_ru=0.1, ud=1.0, bnds=[0,300]):
    """Returns inferred subjective quantities from combination of observable subject data and task information."""
    ntrials = len(obs)

    # Use default params if not provided
    if params is None:
        params = DEFAULT_PARAMS_SUBJ

    # Initialize subject beliefs
    beliefs = Beliefs(ntrials)

    # Convert hazard to vector if scalar
    if np.isscalar(hazard):
        hazard = np.full(ntrials, hazard)

    # Compute both CPP and relative uncertainty according to subject prediction errors
    for t in range(ntrials):

        # Reset relative uncertainty at new blocks
        if new_blk[t]:
            beliefs.relunc[t] = init_ru
        else:
            beliefs.relunc[t] = get_relunc(beliefs.relunc[t-1], responses.pe[t-1], beliefs.cpp[t-1], noise_sd[t], ud)

        # Compute total uncertainty
        tot_unc = (noise_sd[t] ** 2) / (1 - beliefs.relunc[t])

        # Compute error based CPP for subject data
        beliefs.cpp[t] = get_cpp(obs[t], responses.pred[t], np.sqrt(tot_unc), bnds, hazard[t], mix=params['mix'])

        # Compute learning rate predicted
        beliefs.model_lr[t] = get_learning_rate(
            beliefs, t,
            bpe=params['beta_pe'],
            bcpp=params['beta_cpp'],
            bru=params['beta_ru'],
            clip=params['clip'],
            noise_sd=params['noise_sd_update']
        )

    # Compute the update we would predict from these beliefs
    beliefs.model_up = beliefs.model_lr * responses.pe

    # Make sure beliefs are never NaN
    assert(not np.isnan(beliefs.model_up).any())
    assert(not np.isnan(beliefs.model_lr).any())

    return beliefs


def get_learning_rate(beliefs, t, bpe = 0.0, bcpp=1.0, bru=1.0, clip=False, noise_sd=0.0):
    """Computes learning rate. This is shared between simulation and inference."""
    # Learning rate as weighted sum
    lr = bpe + bcpp*beliefs.cpp[t] + bru*beliefs.relunc[t]*(1-beliefs.cpp[t])
        
    # Limit learning rate to [0,1] if specified
    lr = np.clip(lr, 0.0, 1.0) if clip else lr

    # Add learning rate noise, but don't flip sign
    proposal = lr + np.random.normal(0, noise_sd)
    proposal = proposal if np.sign(proposal) == np.sign(lr) else 0

    return proposal


def get_runlen(obs, pred, pe, cpp, relunc, runlen_est, noise_var):
    """Estimate the current run length given the observation, prediction, and other parameters."""
    # Opposite of change-point probability, for convenience
    stay_prob = 1 - cpp

    # Calculate the sum of squares (ss) term
    ss = (
        cpp * noise_var
        + stay_prob * noise_var / (runlen_est + 1)
        + cpp * stay_prob * ((pred + relunc * pe) - obs) ** 2
    )

    # Update run length estimate
    runlen_est = noise_var / ss

    return runlen_est


def get_relunc(relunc, pe, cpp, noise_sd, ud):
    '''Estimate relative uncertainty.'''

    # Transform to variance
    noise_var = noise_sd**2

    # Numerator from equation 6 in heliFMRI paper:
    num = (cpp*noise_var)+((1-cpp)*relunc*noise_var) + cpp*(1-cpp)*(pe*(1-relunc))**2

    num    = num/ud;          # Divide uncertainty about mean by constant
    denom  = num + noise_var  # Denominator is just numerator plus noise_sd variacne
    relunc = num/denom;       # RU is just the fraction

    # Make sure relunc is never NaN or negative
    assert(not (np.isnan(relunc) or relunc < 0.0))

    return relunc


def get_cpp(obs, pred, obs_sd, bnds, hazard, mix=1.0):
    """Calculate a change-point probability estimate."""
    
    # Probability of the data given our state estimate and observation SD estimate
    obs_prob = sp.stats.norm.pdf(obs, pred, obs_sd)

    # Warn if obs_prob is zero
    if obs_prob == 0.0:
        #print(f"Zero probability of observation {obs} given prediction {pred} and SD {obs_sd}")
        #print('Setting to 1e-10.')
        obs_prob = 1e-10

    # Normalize to correct for probability outside of range
    cdf_min = sp.stats.norm.cdf(bnds[0], pred, obs_sd)
    cdf_max = sp.stats.norm.cdf(bnds[1], pred, obs_sd)
    obs_prob = obs_prob / (cdf_max - cdf_min)

    # Uniform prior likelihood of a change point
    like_cp = 1 / (bnds[1] - bnds[0])

    # Likelihood ratio in favor of being a change point
    like_ratio = like_cp / obs_prob

    # Hazard ratio
    hr = hazard / (1 - hazard)

    # Bayes optimal change-point probability estimate
    change_ratio = np.exp(mix * np.log(like_ratio) + np.log(hr))
    cpp = ratio_to_prob(change_ratio)

    # Ensure that the change-point probability is within bounds
    assert(cpp >= 0 and cpp <= 1 and not np.isnan(cpp))

    return cpp

def ratio_to_prob(ratio):
    """Convert a likelihood ratio to a probability."""
    return ratio / (1 + ratio)

def get_cpp_sigmoid(pe, obs_sd=25, hazard=0.1, lp=0, up=1, N=300, loc=0, unc=1, mix=1, la=0, ua=1, debug=False):
    """
    Direct parameterization of the sigmoid function determining CPP.

    Parameters:
        pe (float)       - Prediction error, obs - pred
        obs_sd (float)   - Observation uncertainty, as a standard deviation (r+1)/r * noise_sd in Bayes case
        hazard (float)   - Hazard rate as probability
        lp (float)       - Observation probability under Gaussian which is truncated by lower bound on bin range
        up (float)       - Observation probability which is truncated by upper bound
        N (int)          - Number of bins over which uniform probability is distributed for post-CP observation
        loc (float)      - Location change parameter for sigmoid
        unc (float)      - Uncertainty multiplier for sigmoid (acts on slope)
        mix (float)      - Mixture parameter affecting the sigmoid curve
        la (float)       - Lower asymptote of the sigmoid
        ua (float)       - Upper asymptote of the sigmoid

    Returns:
        p (float)        - Probability from the sigmoid function
    """
    # Check that every argument is a scalar
    if debug:
        if not all(isinstance(arg, (int, float)) for arg in [pe, obs_sd, hazard, lp, up, N, loc, unc, mix, la, ua]):
            raise ValueError("All arguments to sigmoid must be scalar values.")
    
    # Inverse of usual hazard ratio
    ihr = (1 - hazard) / hazard

    # Sigmoid uncertainty (slope) as a scaling of the Bayes uncertainty
    unc = obs_sd * unc

    # Location as shift in P(CP) = 0.5
    loc = max(abs(pe) - loc, 0)

    # Likelihood without change-point
    gauss = (1 / (up - lp)) * np.sqrt(1 / (2 * np.pi * unc**2)) * np.exp(-0.5 * (loc / unc)**2)

    # Likelihood under uniform distribution
    unif = 1 / N

    # Likelihood ratio (no priors)
    lr = gauss / unif

    # Probability from the sigmoid function
    sigmoid =  1 / (1 + lr**mix * ihr)

    # Parameterize the asymptotes
    return la + (ua - la) * sigmoid