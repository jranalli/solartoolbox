import numpy as np
from scipy.optimize import minimize_scalar, leastsq

from solarspatialtools.signalproc import compute_delays, compute_delays_tf
from solarspatialtools.spatial import project_vectors, compute_vectors
from solarspatialtools.spatial import pol2rect, magnitude, dot

from scipy.optimize import linear_sum_assignment, shgo
from scipy.stats import linregress

from enum import Enum
from collections import Counter
import itertools


class Flag(Enum):
    GOOD = 0
    NOSIGNAL = 1
    LOWVAR_S = 2
    LOWCCF = 3
    OVERCAST = 4
    NOLAG = 5
    LOWCORR = 6
    STATS = 7
    VELLIMIT = 8
    ALLBAD = 9
    TREND = 10
    SAME = 11

    DESCS = {
        GOOD: "good data",
        NOSIGNAL: "one signal has zero variance, likely bogus",
        LOWVAR_S: "Low variation ratio in a signal",
        LOWCCF: "Low cloud cover fraction in a signal (i.e. too clear)",
        OVERCAST: "low overall kt in both signals",
        NOLAG: "no lag computed (infinite velocity)",
        LOWCORR: "low lagging correlation (<0.8)",
        VELLIMIT: "Jamaly and Kleissl Low velocity limit",

        ALLBAD: "no good datapoints",
        TREND: "trend",
        SAME: "pair is same site"}


class WindspeedData:
    """
    A holder data structure
    """
    corr_raw = None  # Zero-lag cross-correlations relative to ref point
    corr_lag = None  # Max-lag cross-correlation
    allpairs = None  # Points used for windspeed measurement
    pair_lag = None  # Point-wise maximum lag
    pair_coh = None  # Point-wise mean coherence (TF backend only)
    pair_dists = None  # Point-wise distance to reference in m
    wind_speed = None  # Wind speed in m/s
    wind_angle = None  # Wind dir in radians
    pair_flag = None
    flag = None  # A data quality flag
    method_data = None  # A holder for method-specific data

    _fields = (
        'corr_raw',
        'corr_lag',
        'allpairs',
        'pair_lag',
        'pair_coh',
        'pair_dists',
        'wind_speed',
        'wind_angle',
        'pair_flag',
        'flag',
        'method_data',
    )

    def to_dict(self):
        """Return WindspeedData members as a dictionary."""
        return {field: getattr(self, field, None) for field in self._fields}

    def __repr__(self):
        return repr(self.to_dict())


def _get_pairs(all_points, must_contain=None, replacement=True):
    """
    Build the array of all possible site pairs from a given list of points.
    Works by generating number of combinations (with replacement)

    Parameters
    ----------
    all_points : iterable
        List of all point ids

    must_contain : iterable (optional, default: None)
        List of points that must be contained within all entries of the final
        list. Use for example to force all combinations to contain a reference
        site.

    replacement : boolean (optional, default: True)
        Should the list of pairs be executed with replacement? True will result
        in the identity pair (i.e. 1-1) being generated, while False will lead
        to only non-identical pairs.

    Returns
    -------
    pairs : np.array
        An N x 2 numpy array containing all the combination pairs.

    """

    # # Old Method - All possible permutations (2-way)
    # pairs_all = list(itertools.permutations(all_points, 2))
    # # Permutations with replacement
    # pairs_all = list(itertools.product(all_points, repeat=2))

    # New method, combination not permutation
    if replacement:
        pairs_all = list(
            itertools.combinations_with_replacement(all_points, 2))
    else:
        pairs_all = list(itertools.combinations(all_points, 2))

    if must_contain is not None:
        # Downselect to those that contain the parts
        pairs = []
        for pair in pairs_all:
            if pair[0] in must_contain or pair[1] in must_contain:
                pairs.append(pair)
        pairs = np.array(pairs)
        return np.array(pairs)
    else:
        return np.array(pairs_all)


def compute_cmv(timeseries, positions, reference_id=None, method="jamaly",
                options=None, delay_backend='xcorr'):
    """
    Find Cloud Motion Vector based on clear sky index timeseries from a cluster
    of sensors.

    Default uses the method by Jamaly and Kleissl [1]. An alternate method
    described by Gagne [2] is available. Optionally computes relative
    to a single reference point rather than a global computation across all
    possible site pairs.

    Parameters
    ----------
    timeseries : pandas.DataFrame
        dataframe of kt timeseries for all sensors. Columns should be labelled
        with the sensor id. All sensors should share a uniform time index.

    positions : pandas.DataFrame
        dataframe of the site positions. Index must be the site IDs. The first
        column should be the x coordinate of each site, and the second column
        should be the y coordinate. Coordinates must represent a rectilinear
        coordinate system, measured in meters. Consider converting to UTM using
        spatial.latlon2utm().

    reference_id : numeric, str, iterable, or None (default None)
        The identifier of a single reference site id within the dataframe.
        OR
        a list/tuple of identifiers of references within the dataframe.
        OR
        None to signify that all possible site pairs should be considered.

    method : str (default 'jamaly')
        Method to use. Currently accepted methods are 'jamaly' and 'gagne'

    options : dict (default {})
        Dictionary of detailed QC arguments for the methods.
            All Methods:
                ktlim : float (default 0.4)
                    Minimum permissible clear sky index for time period
            Jamaly:
                minvelocity : float (default 0 m/s)
                    Minimum permissible pairwise velocity in m/s
                maxvelocity : float (default 70 m/s)
                    Maximum permissible pairwise velocity in m/s
                var_s_lim : float (default 0.05)
                    Minimum permissible pairwise variation ratio in signals
                mincorr : float (default 0.8)
                    Minimum permissible correlation coefficient for site pairs
            Gagne:
                No method specific options
    delay_backend : str (default 'xcorr')
        Delay estimation backend. Options are:
            'xcorr':
                Uses cross-correlation to estimate delay and QC metrics.
            'tf':
                Uses transfer-function phase for delay estimation with fixed
                settings (navgs=5, coh_limit=0.6, freq_limit=0.02,
                method='multi', overlap=0.5, window='hamming'), and computes
                cross-correlation QC metrics at the TF-estimated delay.

    Returns
    -------
    cmv_vel : numeric
        The cloud motion vector magnitude in meters per second

    cmv_theta : numeric
        The cloud motion direction as an angle measured CCW from east in rads

    outdata : WindspeedData
        An object containing detailed data about the individual sites

    [1] M. Jamaly and J. Kleissl, “Robust cloud motion estimation by
    spatio-temporal correlation analysis of irradiance data,” Solar
    Energy, vol. 159, pp. 306–317, Jan. 2018. [Online]. Available:
    https://www.sciencedirect.com/science/article/pii/S0038092X17309556

    [2] A. Gagné, N. Ninad, J. Adeyemo, D. Turcotte, and S. Wong, “Directional
    Solar Variability Analysis,” in 2018 IEEE Electrical Power and Energy
    Conference (EPEC) (2018) pp. 1–6, iSSN: 2381-2842
    https://www.researchgate.net/publication/330877949_Directional_Solar_Variability_Analysis
    """

    # Validate method
    methods = ['jamaly', 'gagne']
    method = method.lower()
    if method not in methods:
        raise ValueError('Method must be one of: ' + str(methods) + '.')
    delay_backend = delay_backend.lower()
    if delay_backend not in ['xcorr', 'tf']:
        raise ValueError("delay_backend must be one of: ['xcorr', 'tf'].")

    # Ignore some numpy printouts, we'll deal with them manually
    np.seterr(divide='ignore')
    np.seterr(invalid='ignore')

    # Find all the needed info
    nsamples, _ = timeseries.shape
    point_ids = positions.index.to_list()

    # Set ref_ids to a list of ref_ids that should be used
    if reference_id is not None:
        # Check if it's already an iterable (not raw str though)
        if (hasattr(reference_id, "__iter__") and
                (not isinstance(reference_id, str))):
            ref_ids = reference_id
        else:
            ref_ids = [reference_id]
    else:
        # If it's none, get the full list
        ref_ids = point_ids.copy()

    # Get all possible point pairs
    pairs = _get_pairs(point_ids, ref_ids, replacement=True)

    # Get some parameters about the time series
    ts = timeseries  # Shorten the name for simplicity

    # This is the alternative to looping over all the pairs
    A = ts[pairs[:, 0]]
    B = ts[pairs[:, 1]]

    # We always need correlation-coefficient QC metrics for pairwise QC.
    if delay_backend == 'xcorr':
        delay, extras = compute_delays(A, B, 'loop', scaling='coeff')
    else:
        delay, extras = compute_delays_tf(A, B, navgs=5, coh_limit=0.6,
                                          freq_limit=0.02, method='multi',
                                          overlap=0.5, window='hamming',
                                          compute_xcorr=True)
    corr_lag = extras['peak_corr']
    corr_mean = extras['mean_corr']

    # Vectors
    vectors_cart = (positions.loc[pairs[:, 1]].values
                    - positions.loc[pairs[:, 0]].values)

    # Pairwise QC
    pair_flags, method_out = _pairwise_qc(pairs, magnitude(vectors_cart.T),
                                          [A, B], delay, corr_lag, corr_mean,
                                          method, options)

    # Perform the Gagne QC, which looks for correlation on the whole set to
    # decide which pairs are worth using
    overall_flag = Flag.GOOD
    if method == 'gagne':
        corr_inds = corr_lag >= 0.99
        if sum(corr_inds[pair_flags == Flag.GOOD]) < 12:
            corr_inds = corr_lag >= 0.95
        if sum(corr_inds[pair_flags == Flag.GOOD]) < 12:
            corr_inds = corr_lag >= 0.9
        if sum(corr_inds[pair_flags == Flag.GOOD]) < 12:
            overall_flag = Flag.ALLBAD
        pair_flags[np.bitwise_not(corr_inds)] = Flag.LOWCORR

    # Jump out if no pairs passed QC
    if np.sum(pair_flags == Flag.GOOD) <= 1:  # they all fail QC
        overall_flag = Flag.ALLBAD
        w = WindspeedData()
        w.allpairs = pairs
        w.flag = overall_flag
        w.pair_flag = pair_flags
        return 0, 0, w

    # Select the subsets of delay and vectors to use
    delay_good = delay[pair_flags == Flag.GOOD]
    vectors_good = vectors_cart[pair_flags == Flag.GOOD]

    if method == 'jamaly':
        # Function that we'll minimize
        def vel_variance(theta, lag, vecs):
            # Project each vector into the direction of theta
            dir_dist = project_vectors(vecs, np.array(pol2rect(1, theta)))

            # Compute wind speed for each pair, ignoring infinites
            vel = dir_dist / lag
            vel[vel == np.inf] = np.nan
            vel[vel == -np.inf] = np.nan

            # compute the variance for this value of theta
            variance = np.nanvar(vel)
            return variance

        # Minimize the variance in the computed velocities to get theta. Use
        # only the Flag.Good values in delay and the vectors
        opt = minimize_scalar(vel_variance, args=(delay_good, vectors_good))
        cmv_theta = opt.x

        # Compute the velocity for all the useful pairs at this theta. The
        # actual velocity is the median over all useful pairs
        cmv_dir_dist = project_vectors(vectors_good,
                                       np.array(pol2rect(1, cmv_theta)))
        velocity = cmv_dir_dist / delay_good
        velocity[velocity == np.inf] = np.nan
        velocity[velocity == -np.inf] = np.nan
        cmv_vel = np.nanmedian(velocity)

        # QC - Compute the error index from Shinozaki et al (Jamaly eq 3)
        v60 = np.nanpercentile(velocity, 60)
        v40 = np.nanpercentile(velocity, 40)
        err = (v60 - v40) / cmv_vel
        if np.abs(err) > 0.4:  # Set flag if appropriate
            overall_flag = Flag.TREND

        method_out['v60'] = v60
        method_out['v40'] = v40
        method_out['error_index'] = err
        method_out['velocity'] = velocity

        # Compute a few extra statistics for the good pairs
        try:
            _, _, r_corr, _, std_err = linregress(delay_good, cmv_dir_dist)
        except ValueError:
            r_corr = 0
            std_err = 0
        ngood = len(vectors_good)
        method_out['r_corr'] = np.abs(r_corr)
        method_out['stderr_corr'] = std_err
        method_out['ngood'] = ngood

        good_pairs = [pair for pair, keep in zip(pairs, pair_flags==Flag.GOOD) if keep]
        degree = list((Counter(sensor for pair in good_pairs for sensor in pair)).values())
        method_out['degree_good'] = np.max(degree)/np.sum(degree)

    elif method == 'gagne':
        # Function to apply least squares to
        def resid(p, lag, coeff_vecs):
            dx, dy = coeff_vecs[:, 0], coeff_vecs[:, 1]
            ax, ay = p[0], p[1]
            # Residual as defined in Gagne 2018 equation 3
            return lag - (dx * ax + dy * ay)

        [ax_opt, ay_opt], pcov = leastsq(resid, np.array([1, 1]),
                                         args=(delay_good, vectors_good))
        method_out['pcov'] = pcov

        # Gagne 2018 equation 4
        vx = 1 / (ax_opt ** 2 + ay_opt ** 2) * ax_opt
        vy = 1 / (ax_opt ** 2 + ay_opt ** 2) * ay_opt

        cmv_vel = magnitude([vx, vy])
        cmv_theta = np.arctan2(vy, vx)
    else:
        raise ValueError('Invalid method: ' + str(method) + '.')

    # Make velocity always positive and change distances to accommodate
    if cmv_vel < 0:
        cmv_vel *= -1
        cmv_theta += np.pi
    cmv_theta %= (2*np.pi)
    cmv_dir_dist = project_vectors(vectors_cart,
                                   np.array(pol2rect(1, cmv_theta)))

    # Create the output data object
    outdata = WindspeedData()
    outdata.flag = overall_flag
    outdata.pair_flag = pair_flags
    outdata.pair_lag = delay
    outdata.pair_coh = extras.get('mean_coh', None)
    outdata.allpairs = pairs
    outdata.corr_raw = extras.get('zero_corr', None)
    outdata.corr_lag = corr_lag
    outdata.pair_dists = cmv_dir_dist
    outdata.wind_angle = cmv_theta
    outdata.wind_speed = cmv_vel
    outdata.method_data = method_out

    return cmv_vel, cmv_theta, outdata


def optimum_subset(cmvx, cmvy, n=10):
    """
    Chooses a subset of vectors from within a full set of vectors, based on
    optimizing the most diverse angles available. Operates in 2 quadrants only,
    subject to the assumption that anti-parallels are also undesirable.

    This is useful for getting vectors that represent a variety of directions.
    For example, given all the CMVs for a whole year, this will provide a list
    of the time periods with the most varied cloud motion directions. One use
    case is in field analysis where the most diverse set of CMVs is needed to
    get the highest probability of perpendicular vector pairs.

    Parameters
    ----------
    cmvx : pd.Series
        A list of x-components of the CMVs

    cmvy : pd.Series
        A list of y-components of the CMVs

    n : int
        The number of vectors to select

    Returns
    -------
    indices : list
        The indices of the CMVs that are most diverse. Indices are positional
        within the input arrays. For example, could be used to reduce a
        DataFrame to the most diverse CMVs by using df.iloc[indices]
    """

    # Compute unit vectors representing the CMVs
    cld_spd = []
    for x, y in zip(cmvx, cmvy):
        cld_spd.append(magnitude((x, y)))
    cld_spd = np.array(cld_spd)
    cmv_x = cmvx / cld_spd
    cmv_y = cmvy / cld_spd
    cmv_vecs = np.array([cmv_x, cmv_y]).T

    def calc_cost(ang_0):
        """
        Compute the cost function associated with using each CMV as a member of
        the set relative to the ideal set of equally spaced vectors.

        Parameters
        ----------
        ang_0 : the rotation angle of the ideal vectors in radians

        Returns
        -------
        cost : the total cost of the assignment
        indices : the indices of the CMVs in the optimal assignment
        """
        # Compute unit vectors equally distributed around 180 deg.
        ideal_angs = np.arange(0, n) / n * np.pi + ang_0
        ideal_vecs = np.array([np.cos(ideal_angs), np.sin(ideal_angs)]).T

        # Compute cost as dot products between each CMV and each ideal vector
        # Absolute value used because both parallel and anti-parallel are bad
        # for our case.
        dots = -np.array([np.abs(dot(cmv_vecs.T, ideal_vec))
                          for ideal_vec in ideal_vecs])

        # Compute the optimal assignment of CMVs to maximize alignment with the
        # ideal vectors. Relative cost could be used to compare multiple zero
        # angles.
        r_ind, c_ind = linear_sum_assignment(dots)
        cost = dots[r_ind, c_ind].sum()
        return cost, c_ind

    def cost_wrapper(ang_0):
        """
        The minimizer only works on a single output, so in this case we wrap it
        """
        return calc_cost(ang_0)[0]  # return only the cost

    # The bounds of the optimization are limited by the spacing between ideal
    # vectors.
    bounds = [(0, np.pi / n)]

    #  What we're optimizing here is a rotation angle for the set of ideal vecs
    y = shgo(cost_wrapper, bounds)

    # Use the optimized angle to figure out which CMVs to use
    final_cost, indices = calc_cost(y.x[0])

    return indices


def _pairwise_qc(pairs, spacing, sigs, delay, peak_corr, mean_corr,
                 method, options=None):
    """
    Perform pairwise QC on the CMV signals according to method

    Parameters
    ----------
    pairs : np.array
        An N x 2 numpy array containing all the combination pairs.
    spacing : np.array
        An array of the distances between each pair in meters
    sigs : list of np.array
        A list of the two signals to be compared
    delay : np.array
        An array of the delays between each pair in seconds
    peak_corr : np.array
        An array of the peak cross-correlation between each pair
    mean_corr : np.array
        An array of the mean cross-correlation between each pair
    method : str
        The method to use for QC. Currently accepted methods are 'jamaly' and
        'gagne'
    options : dict or None
        A dictionary of options for the QC. See _validate_method_options for
        details
    """
    # Validate the options
    options, method_out = _validate_method_options(method, options)

    A, B = sigs

    # Initialize the flags for all the pairs
    npts = A.shape[1]
    pair_flags = np.empty(npts, dtype=Flag)
    pair_flags.fill(Flag.GOOD)

    # Global QC Common to all Methods

    # Same sensor lag and correlation is meaningless
    same_ind = pairs[:, 0] == pairs[:, 1]
    pair_flags[same_ind] = Flag.SAME

    # Base STD rejects faulty sensors with no variance at all, possibly bad
    # sensor data?
    siginds = np.bitwise_or(np.nanstd(A, axis=0) < 0.001,
                            np.nanstd(B, axis=0) < 0.001)
    pair_flags[siginds] = Flag.NOSIGNAL

    # Zero lag Reject sensor pairs with no delay
    pair_flags[delay == 0] = Flag.NOLAG

    # Reject sensor pairs with nan correlation
    pair_flags[np.isnan(peak_corr)] = Flag.LOWCORR
    pair_flags[np.isnan(mean_corr)] = Flag.LOWCORR

    # Signals are extremely overcast
    ktlim = options['ktlim']
    ktinds = np.bitwise_or(np.nanmax(A, axis=0) < ktlim,
                           np.nanmax(B, axis=0) < ktlim)
    pair_flags[ktinds] = Flag.OVERCAST

    # Method-specific QC
    if method == 'jamaly':
        # Jamaly and Kleissl - low variation ratio
        var_s_lim = options['var_s_lim']
        var_s0 = 1 - np.nanmean(A, axis=0) / np.nanmax(A, axis=0)
        var_s1 = 1 - np.nanmean(B, axis=0) / np.nanmax(B, axis=0)
        method_out['var_s'] = [var_s0, var_s1]
        var_s_inds = np.bitwise_or(var_s0 < var_s_lim, var_s1 < var_s_lim)
        pair_flags[var_s_inds] = Flag.LOWVAR_S

        # # Jamaly and Kleissl - low cloud cover fraction (i.e. too clear)
        # ccf_lim = 0.1  # More than 10% of points should have kt < 0.85
        # ccf0 = np.nansum(sig0 < 0.85) / len(sig0)  # Fraction of kt<0.85
        # ccf1 = np.nansum(sig1 < 0.85) / len(sig1)
        # if ccf0 < ccf_lim or ccf1 < ccf_lim:
        #     pair_flags[pair_ind] = Flag.LOWCCF
        #     continue

        # Jamaly and Kleissl - require minimum lagged xcorr and minimum
        # cross correlation ratio
        r_qc = 1 - mean_corr / peak_corr
        corr_min = options['mincorr']
        method_out['r_qc'] = r_qc
        corr_inds = np.bitwise_or(peak_corr < corr_min, r_qc < corr_min)
        pair_flags[corr_inds] = Flag.LOWCORR

        # Jamaly and Kleissl distance limit is confusing; I modified here
        # Original appears to require that the max velocity be less than
        # 1 m/s I implement it as a literal velocity limit rather than dist
        min_v = options['minvelocity']  # m/s (could be introduced)
        max_v = options['maxvelocity']  # m/s (70 m/s, around 160 mph)
        v = spacing/delay
        vind = np.bitwise_not(np.bitwise_and(min_v < np.abs(v),
                                             np.abs(v) < max_v))
        pair_flags[vind] = Flag.VELLIMIT

    elif method == 'gagne':
        pass  # No pairwise QC

    return pair_flags, method_out


def _validate_method_options(method, options):
    """
    Validate and fill defaults for an options dictionary for the various CMV
    methods.

    Parameters
    ----------
    method : str
        The method to validate options for. Currently accepted methods are:
        'jamaly' and 'gagne'.
    options : dict
        The options dictionary to validate. If None, a default dictionary will
        be created.

    Returns
    -------
    options : dict
        The validated options dictionary filled with defaults as applicable.
    method_out : dict
        A dictionary of method-specific output data. This is used to store
        intermediate data for debugging purposes.
    """
    # Validate options
    if options is None:
        options = {}
    # Specify default options by method
    if method == 'jamaly':
        defaults = {'minvelocity': 0,  # m/s
                    'maxvelocity': 70,  # m/s (about 160 mph)
                    'var_s_lim': 0.05,  # 0.10 in original paper
                    'ktlim': 0.4,
                    'mincorr': 0.8
                    }
        method_out = {
            'error_index': None,  # The Error Index
            'velocity': None,  # Velocity for each pair
            'v60': None,  # 60th percentile velocity
            'v40': None,  # 40th percentile velocity
            'r_qc': [],  # Ratio of peak to mean xcorr
            'var_s': [],  # Variation ratio of signals [s0, s1]
            'r_corr': None,  # Corr. coeff. for dist/delay for good pairs
            'stderr_corr': None,  # Std. error for dist/delay for good pairs
            'ngood': None,  # Number of good pairs
        }
    elif method == 'gagne':
        defaults = {
            'ktlim': 0.4,
        }
        method_out = {
            'pcov': None,  # Covariance matrix from least squares
        }
    else:
        raise ValueError('Invalid method: ' + str(method) + '.')

    # Validate option keys are valid for method
    for key in options:
        if key not in defaults:
            raise ValueError(f'Invalid option for {method}: ' + str(key) + '.')

    # Apply defaults if option not specified
    for key in defaults:
        if key not in options:
            options[key] = defaults[key]

    return options, method_out
