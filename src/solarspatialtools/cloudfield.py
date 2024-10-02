import numpy as np
from scipy.interpolate import RegularGridInterpolator
# from scipy.ndimage import map_coordinates

import matplotlib.pyplot as plt
from scipy.ndimage import sobel, uniform_filter

from solarspatialtools.stats import variability_score


def _random_at_scale(rand_size, final_size, plot=False):
    """
    Generate a 2D array of random. Generate initially at the size of rand_size
    and then linearly interpolate to the size of final_size.

    Parameters
    ----------
    rand_size : tuple
        The size of the random array to generate (rows, cols)
    final_size : tuple
        The size of the final array to return (rows, cols)

    Returns
    -------
    np.ndarray
        A 2D array of random values
    """

    # Generate random values at the scale of rand_size
    random = np.random.rand(rand_size[0], rand_size[1])

    # Linearly interpolate to the final size
    x = np.linspace(0, 1, rand_size[0])
    y = np.linspace(0, 1, rand_size[1])

    xnew = np.linspace(0, 1, final_size[0])
    ynew = np.linspace(0, 1, final_size[1])

    # # New Scipy Method
    interp_f = RegularGridInterpolator((x, y), random, method='linear')
    Xnew, Ynew = np.meshgrid(xnew, ynew, indexing='ij')
    random_new = interp_f((Xnew, Ynew))

    # # # Alternate Scipy Method
    # interp_f = RectBivariateSpline(x, y, random, kx=1, ky=1)
    # # interp_ft = lambda xnew, ynew: interp_f(xnew, ynew).T
    # random_new = interp_f(xnew, ynew)

    # # # Different potentially faster Scipy Method
    # Xnew, Ynew = np.meshgrid(xnew, ynew, indexing='ij')
    # Xnew = Xnew*len(x)
    # Ynew = Ynew*len(y)
    # random_new = map_coordinates(random, [Xnew.ravel(), Ynew.ravel()], order=1, mode='nearest').reshape(final_size)

    if plot:
        # generate side by side subplots to compare
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(random, extent=(0, rand_size[1], 0, rand_size[0]))
        axs[0].set_title('Original Random')
        axs[1].imshow(random_new, extent=(0, final_size[1], 0, final_size[0]))
        axs[1].set_title('Interpolated Random')
        plt.show()

    return random, random_new


def _calc_vs_weights(scales, vs):
    """
    Calculate the weight for each scale

    Parameters
    ----------
    scales : int
        The space scale
    vs : float
        The variability score

    Returns
    -------

    """
    VS1 = -np.log(1-vs/180)/0.6
    weight = scales ** (1/VS1)
    return weight / np.sum(weight)


def _calc_wavelet_weights(waves):
    """
    Calculate the weights for each wavelet

    Parameters
    ----------
    waves : np.ndarray
        The wavelet coefficients

    Returns
    -------

    """
    scales = np.nanmean(waves**2, axis=1)
    return scales / np.sum(scales)



def space_to_time(pixres=1, cloud_speed=50):

    # Existing code uses 1 pixel per meter, and divides by cloud speed to get X size
    # does 3600 pixels, but assumes the clouds move by one full timestep.

    # ECEF coordinate system (Earth Centered Earth Fixed) is used to convert lat/lon to x/y.
    # This seems really weird. https://en.wikipedia.org/wiki/Geographic_coordinate_conversion

    # [X, Y] = geod2ecef(Lat_to_Sim, Lon_to_Sim, zeros(size(Lat_to_Sim)));
    # ynorm1 = Y - mean(Y);
    # xnorm1 = X - mean(X);
    # ynorm=round((ynorm1-min(ynorm1))./Cloud_Spd(qq))+1;
    # xnorm=round((xnorm1-min(xnorm1))./Cloud_Spd(qq))+1;
    # Xsize=60*60+max(xnorm);
    # Ysize=max(ynorm);
    # Xsize and Ysize are the pixel sizes generated

    # Extracting a time series has us loop through the entire size of X pixels, and choose a window 3600 pixels wide, and multiply by GHI_cs
    # GHI_syn(i,hour(datetime2(GHI_timestamp))==h1)=clouds_new{h1}(ynorm(i),xnorm(i):xnorm(i)+3600-1)'.*GHI_clrsky(hour(datetime2(GHI_timestamp))==h1);

    pixjump = cloud_speed / pixres
    # n space
    # dx space
    # velocity
    # dt = dx / velocity
    # max space = n * dx
    # max time = max space / velocity


def stacked_field(vs, size, weights=None, scales=(1, 2, 3, 4, 5, 6, 7), plot=False):

    field = np.zeros(size, dtype=float)

    if weights is None:
        weights = _calc_vs_weights(scales, vs)
    else:
        assert len(weights) == len(scales)

    for scale, weight in zip(scales, weights):
        prop = 2**(-scale+1)  # proportion for this scale
        _, i_field = _random_at_scale((int(size[0]*prop), int(size[1]*prop)), size)
        field += i_field * weight

    # Scale it zero to 1??
    field = (field - np.min(field))
    field = field / np.max(field)
    assert np.min(field) == 0
    assert np.max(field) == 1

    if plot:
        # Plot the field
        plt.imshow(field, extent=(0, size[1], 0, size[0]))
        plt.show()

    return field

def _clip_field(field, clear_frac=0.5, plot=False):
    """
    Find the value in the field that will produce an X% clear sky mask. The
    mask is 1 where clear, and 0 where cloudy.

    Parameters
    ----------
    field
    clear_frac
    plot

    Returns
    -------

    """
    # Zero where clouds, 1 where clear

    # clipping needs to be based on pixel fraction, which thus needs to be
    # done on quantile because the field has a normal distribution
    quant = np.quantile(field, clear_frac)

    # Find that quantile and cap it
    field_out = np.ones_like(field)
    field_out[field > quant] = 0

    # Test to make sure that we're close to matching the desired fraction
    assert (np.isclose(clear_frac, np.sum(field_out) / field.size, rtol=1e-3))

    if plot:
        plt.imshow(field_out, extent=(0, field.shape[1], 0, field.shape[0]))
        plt.show()

    return field_out

def _find_edges(base_mask, size, plot=False):
    """
    Find the edges of the field using a sobel filter and then smooth it with a
    Parameters
    ----------
    size
    plot

    Returns
    -------

    """

    # This gets us roughly 50% overlapping with mask and 50% outside
    edges = np.abs(sobel(base_mask))
    smoothed = uniform_filter(edges, size=size)

    # We want to binarize it
    smoothed[smoothed < 1e-5] = 0  # Zero out the small floating point values
    # Calculate a threshold based on quantile, because otherwise we get the whole clouds
    baseline = np.quantile(smoothed[smoothed>0], 0.5)
    smoothed_binary = smoothed > baseline

    if plot:
        # Compare the edges and uniform filtered edges side by side
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(edges, extent=(0, ysiz, 0, xsiz))
        axs[0].set_title('Edges')
        axs[1].imshow(smoothed_binary, extent=(0, ysiz, 0, xsiz))
        axs[1].set_title('Uniform Filtered Edges')
        plt.show()

    return edges, smoothed_binary

def shift_mean_lave(field, ktmean, max_overshoot=1.4, ktmin=0.2, min_quant=0.005, max_quant=0.995, plot=True):

    # ##### Shift values of kt to range from 0.2 - 1

    # Calc the "max" and "min", excluding clear values
    field_max = np.quantile(field[field < 1], max_quant)
    field_min = np.quantile(field[field < 1], min_quant)

    # Scale it between ktmin and max_overshoot
    field_out = (field - field_min) / (field_max - field_min) * (1-ktmin) + ktmin

    # # Clip limits to sensible boundaries
    field_out[field_out > 1] = 1
    field_out[field_out < 0] = 0

    # ##### Apply multiplier to shift mean to ktmean #####

    # Rescale the mean
    tgtsum = field_out.size * ktmean  # Mean scaled over whole field
    diff_sum = tgtsum - np.sum(field_out == 1)  # Shifting to exclude fully clear values
    tgt_mean = diff_sum / np.sum(field_out < 1)  # Recalculating the expected mean of the cloudy-only aareas
    current_cloud_mean = np.mean(field_out[field_out < 1]) # Actual cloud mean

    if diff_sum > 0:
        field_out[field_out!=1] = tgt_mean / current_cloud_mean * field_out[field_out!=1]

    # print(diff_sum)
    # print(current_cloud_mean)
    print(f"Desired Mean: {ktmean}, actual global mean {np.mean(field_out)}.")


    if plot:
        plt.hist(field_out[field_out<1].flatten(), bins=100)
        plt.show()

        # plot field and field_out side by side
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(field, extent=(0, ysiz, 0, xsiz))
        axs[0].set_title('Original Field')
        axs[1].imshow(field_out, extent=(0, ysiz, 0, xsiz))
        axs[1].set_title('Shifted Field')
        plt.show()
    return field_out


def lave_scaling_exact(field, clear_mask, edge_mask, ktmean, ktmax=1.4, kt1pct=0.2, max_quant=0.99, plot=True):

    # ##### Shift values of kt to range from 0.2 - 1

    # Calc the "max" and "min", excluding clear values
    field_max = np.quantile(field[clear_mask == 0], max_quant)
    print(f"Field Max: {field_max}")
    print(f"kt1pct: {kt1pct}")

    # Create a flipped version of the distribution that scales between slightly below kt1pct and bascially (1-field_min)
    # I think the intent here would be to make it vary between kt1pct and 1, but that's not quite what it does.
    clouds3 = 1 - field*(1-kt1pct)/field_max


    # # Clip limits to sensible boundaries
    clouds3[clouds3 > 1] = 1
    clouds3[clouds3 < 0] = 0

    # ##### Apply multiplier to shift mean to ktmean #####
    mean_c3 = np.mean(clouds3)
    nmin_c3 = np.min(clouds3)/mean_c3
    nrange_c3 = np.max(clouds3)/mean_c3-nmin_c3
    ce = 1+ (clouds3/mean_c3-nmin_c3)/nrange_c3*(ktmax-1)

    # Rescale one more time to make the mean of clouds3 match the ktmean from the timeseries
    cloud_mask = np.bitwise_or(clear_mask>0, edge_mask) == 0  # Where is it neither clear nor edge
    tgtsum = field.size * ktmean  # Mean scaled over whole field
    diff_sum = tgtsum - np.sum(clear_mask) - np.sum(ce[np.bitwise_and(edge_mask > 0, clear_mask==0)])  # Shifting target to exclude fully clear values and the cloud enhancement
    tgt_cloud_mean = diff_sum / np.sum(cloud_mask)  # Find average required in areas where it's neither cloud nor edge
    current_cloud_mean = np.mean(clouds3[cloud_mask]) # Actual cloud mean

    if diff_sum > 0:
        clouds4 = tgt_cloud_mean / current_cloud_mean * clouds3
    else:
        clouds4 = clouds3.copy()

    clouds5 = clouds4.copy()

    # Edges then clear means that the clearsky overrides the edge enhancement
    clouds5[edge_mask] = ce[edge_mask > 0]
    clouds5[clear_mask > 0] = 1
    print(f"Desired Mean: {ktmean}, actual global mean {np.mean(clouds5)}.")


    if plot:
        plt.hist(ce.flatten(), bins=100)
        plt.hist(clouds3.flatten(), bins=100, alpha=0.5)
        plt.hist(clouds4.flatten(), bins=100, alpha=0.5)
        plt.hist(clouds5.flatten(), bins=100, alpha=0.5)
        plt.hist(field.flatten(), bins=100, alpha=0.5)
        plt.legend(["Cloud Enhancement", "1st Scaled Cloud Distribution", "2nd Scaled Cloud Distribution", "Fully Remapped Distribution",
                    "Original Field Distribution"])

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(field, extent=(0, ysiz, 0, xsiz))
        axs[0].set_title('Original Field')
        axs[1].imshow(clouds5, extent=(0, ysiz, 0, xsiz))
        axs[1].set_title('Shifted Field')
        plt.show()

    return clouds5


def get_settings_from_timeseries(kt_ts, clear_threshold=0.95, plot=True):
    # Get the mean and standard deviation of the time series
    ktmean = np.mean(kt_ts)  # represents mean of kt
    ktstd = np.std(kt_ts)
    ktmax = np.max(kt_ts)  # represents peak cloud enhancement
    ktmin = np.min(kt_ts)

    kt1pct = np.nanquantile(kt_ts, 0.01)  # represents "lowest" kt

    # Get the fraction of clear sky with a threshold
    frac_clear = np.sum(kt_ts > clear_threshold) / kt_ts.size

    vs = variability_score(kt) * 1e4

    # Compute the wavelet weights
    # should be the mean(wavelet squared) for all modes except the steady mode
    waves, tmscales = pvlib.scaling._compute_wavelet(kt_ts)

    if plot:
        # create a plot where each of the timeseries in waves is aligned vertically in an individual subplot
        fig, axs = plt.subplots(len(waves), 1, figsize=(10, 2 * len(waves)), sharex=True)
        for i, wave in enumerate(waves):
            axs[i].plot(wave)
            axs[i].set_title(f'Wavelet {i+1}')
        plt.show()

    waves = waves[:-1, :]  # remove the steady mode
    tmscales = [i+1 for i, _ in enumerate(tmscales[:-1])]
    weights = _calc_wavelet_weights(waves)

    return ktmean, kt1pct, ktmax, frac_clear, vs, weights, tmscales




if __name__ == '__main__':

    import pandas as pd
    import pvlib

    datafn = "../../demos/data/hope_melpitz_1s.h5"
    twin = pd.date_range('2013-09-08 9:15:00', '2013-09-08 10:15:00', freq='1s', tz='UTC')
    data = pd.read_hdf(datafn, mode="r", key="data")
    data = data[40]
    # plt.plot(data)
    # plt.show()

    pos = pd.read_hdf(datafn, mode="r", key="latlon")
    loc = pvlib.location.Location(np.mean(pos['lat']), np.mean(pos['lon']))
    cs_ghi = loc.get_clearsky(data.index, model='simplified_solis')['ghi']
    cs_ghi = 1000/max(cs_ghi) * cs_ghi  # Rescale (possible scaling on
    kt = pvlib.irradiance.clearsky_index(data, cs_ghi, 2)

    # plt.plot(data)
    # plt.plot(cs_ghi)
    # plt.show()
    #
    # plt.plot(kt)
    # plt.show()

    # plt.hist(kt, bins=100)
    # plt.show()

    ktmean, kt1pct, ktmax, frac_clear, vs, weights, scales = get_settings_from_timeseries(kt, plot=False)

    print(f"Clear Fraction is: {frac_clear}")

    np.random.seed(42)  # seed it for repeatability

    xsiz = 2**12
    ysiz = 2**14

    cfield = stacked_field(vs, (xsiz, ysiz), weights, scales)

    clear_mask = stacked_field(vs, (xsiz, ysiz), weights, scales)
    clear_mask = _clip_field(clear_mask, frac_clear, plot=False)  # 0 is cloudy, 1 is clear


    # Clear Everywhere
    out_field = np.ones_like(cfield)
    # Where it's cloudy, mask in the clouds
    out_field[clear_mask == 0] = cfield[clear_mask == 0]

    # plt.imshow(out_field, extent=(0, ysiz, 0, xsiz))
    # plt.show()

    edges, smoothed = _find_edges(clear_mask, 3, plot=False)

    # field_final = shift_mean_lave(out_field, ktmean)
    field_final = lave_scaling_exact(cfield, clear_mask, smoothed, ktmean, ktmax, kt1pct, plot=False)

    plt.plot(field_final[1,:])
    plt.show()

    plt.hist(kt, bins=50)
    plt.hist(field_final[1,:], bins=50, alpha=0.5)
    plt.show()

    plt.hist(np.diff(kt), bins=50)
    plt.hist(np.diff(field_final[1,:]), bins=50, alpha=0.5)
    plt.show()

    # assert np.all(r == rnew)