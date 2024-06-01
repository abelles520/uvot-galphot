import numpy as np
import matplotlib.pyplot as plt

import aplpy
from astropy.io import fits
from astropy.stats import biweight_location, sigma_clip
from reproject import reproject_interp

import pdb

def make_aperture_image(label, filter_list,
                            center_ra, center_dec, major_diam, minor_diam, pos_angle):
    """
    Make a picture of the galaxy with the apertures overlaid

    Currently just does one given aperture, but should eventually do the various
    annuli for each filter

    Parameters
    ----------
    label : string
        label associated with the galaxy, both for finding image/data files and
        saving the aperture image (e.g., 'ngc24_offset_')

    filter_list : list of strings
        filters for the galaxy

    center_ra, center_dec : float
        coordinates of the center of the galaxy (degrees)

    major_diam, minor_diam : float
        major and minor axes for the galaxy ellipse (arcsec)

    pos_angle : float
        position angle of the galaxy ellipse ("position angle increases
        counterclockwise from North (PA=0)")

    """
    counts_im = label + 'sk.fits'
    exp_im = label + 'ex.fits'

    # get the image HDUs
    hdu_list = []
    for filt in filter_list:
        with fits.open(label+filt+'_sk.fits') as hdu_counts, fits.open(label+filt+'_ex.fits') as hdu_ex:
            hdu_list.append(fits.ImageHDU(data=hdu_counts[1].data/hdu_ex[1].data,
                                                header=hdu_counts[1].header))

    # if there's more than one filter, do reprojection
    if len(filter_list) > 1:
        for f in range(1,len(filter_list)):
            new_array, _ = reproject_interp(hdu_list[f], hdu_list[0].header)
            hdu_list[f] = fits.ImageHDU(data=new_array, header=hdu_list[0].header)

    # normalize the images
    for f in range(len(filter_list)):

        # subtract mode
        # - do a sigma clip
        pix_clip = sigma_clip(hdu_list[f].data, sigma=2.5, maxiters=3)
        # - calculate biweight
        biweight_clip = biweight_location(pix_clip.data[~pix_clip.mask])
        # - subtraction
        new_array = hdu_list[f].data - biweight_clip

        # set anything below 0 to 0
        new_array[new_array < 0] = 0

        # set 95th percentile to 1
        new_array = new_array/np.nanpercentile(new_array, 95)

        # save it
        hdu_list[f].data = new_array


    # add the images together
    im_sum = np.mean([hdu_list[f].data for f in range(len(filter_list))], axis=0)

    # make it into an HDU
    hdu_sum = fits.ImageHDU(data=log_image(im_sum, 0, np.nanpercentile(im_sum, 99.5)),
                                header=hdu_list[0].header)

    # make an image
    fig = aplpy.FITSFigure(hdu_sum)
    fig.show_grayscale()
    fig.axis_labels.hide_x()
    fig.axis_labels.hide_y()
    fig.tick_labels.hide_x()
    fig.tick_labels.hide_y()
    fig.frame.set_linewidth(0)

    # aperture ellipses
    fig.show_ellipses(center_ra, center_dec,
                          major_diam/3600, minor_diam/3600,
                          angle=90+pos_angle,
                          edgecolor='red', linewidth=2)

    fig.save(label+'aperture_image.pdf')





def log_image(image, min_val, max_val, ds9_a=1000):
    """
    Use the ds9 formalism to calculate a log (since aplpy log isn't playing nicely with negatives)
    """

    # set everything below/above min/max to min/max
    image[image < min_val] = min_val
    image[image > max_val] = max_val

    # normalize to 0-1
    image = (image - min_val)/(max_val - min_val)

    # take log
    return np.log10(ds9_a * image + 1)/np.log10(ds9_a)
