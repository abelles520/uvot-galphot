import numpy as np
import matplotlib.pyplot as plt

from photutils import SkyEllipticalAperture, SkyEllipticalAnnulus, aperture_photometry, EllipticalAnnulus
from astropy.io import fits
from astropy import units as u
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.stats import biweight_location, sigma_clip
from regions import read_ds9

import pdb


def surface_phot(label, center_ra, center_dec, major_diam, minor_diam, pos_angle,
                     ann_width, zeropoint, mask_file=None, offset_file=False,
                     verbose=False):
    """
    Do surface brightness photometry within annuli

    Parameters
    ----------
    label : string
        label associated with the galaxy, both for finding image files and
        saving results (e.g., 'ngc24_offset_w2_')

    center_ra, center_dec : float
        coordinates of the center of the galaxy (degrees)

    major_diam, minor_diam : float
        major and minor axes for the galaxy ellipse (arcsec)

    pos_angle : float
        position angle of the galaxy ellipse ("position angle increases
        counterclockwise from North (PA=0)")

    ann_width : float
        width of annuli (arcsec)

    zeropoint : float
        conversion from counts/sec into magnitude units
        AB_mag = -2.5*log10(counts/sec) + zeropoint

    mask_file : string (default=None)
        path+name of ds9 region file with masks

    offset_file : boolean (default=False)
        if True, the file label+'sk_off.fits' is used to show what offsets (in
        counts) have already been added to the counts images

    verbose : boolean (default=False)
        if True, print progress

    """

    # read in the images
    counts_im = label + 'sk.fits'
    exp_im = label + 'ex.fits'
    offset_im = label + 'sk_off.fits'

    with fits.open(counts_im) as hdu_counts, fits.open(exp_im) as hdu_ex:

        # if mask file is provided, make a mask image
        if mask_file is not None:
            mask_image = make_mask_image(hdu_counts[1], mask_file)

        # if offset file is set, save it into an array
        if offset_file == True:
            with fits.open(label+'sk_off.fits') as hdu_off:
                counts_off_array = hdu_off[1].data

        # WCS for the images
        wcs_counts = wcs.WCS(hdu_counts[1].header)
        arcsec_per_pix = wcs_counts.wcs.cdelt[1] * 3600
        
        # ellipse center
        #ellipse_center = SkyCoord(ra=center_ra*u.deg, dec=center_dec*u.deg)
        ellipse_center = wcs_counts.wcs_world2pix([[center_ra,center_dec]], 0)
        
        # array of annuli over which to do photometry
        annulus_array = np.arange(0, major_diam*1.2, ann_width)# * u.arcsec 


        
        # -------------------------
        # sky background
        # -------------------------

        # size of sky annulus
        sky_in = annulus_array[-1]
        sky_ann_width = ann_width * 10
        sky_out = sky_in + sky_ann_width
        
        # define aperture object
        aperture = EllipticalAnnulus(tuple(ellipse_center[0]),
                                         a_in=sky_in/arcsec_per_pix,
                                         a_out=sky_out/arcsec_per_pix,
                                         b_out=sky_out/arcsec_per_pix * minor_diam/major_diam,
                                         theta=(90+pos_angle)*np.pi/180)
        # make an ApertureMask object with the aperture
        annulus_mask = aperture.to_mask(method='exact')
        # turn aperture into an image
        annulus_im = annulus_mask[0].to_image(hdu_counts[1].data.shape)

        # make masked version using input ds9 file
        if mask_file is not None:
            annulus_im = annulus_im * mask_image

        # plot things
        #annulus_data = annulus_mask[0].multiply(hdu_counts[1].data)
        #plt.imshow(annulus_mask[0])
        #plt.imshow(annulus_data, origin='lower')
        #plt.imshow(annulus_im, origin='lower')
        #plt.colorbar()

        # list of values within aperture
        nonzero_annulus = np.where(annulus_im > 1e-5)
        annulus_list = annulus_im[nonzero_annulus]
        counts_list = hdu_counts[1].data[nonzero_annulus]
        exp_list = hdu_ex[1].data[nonzero_annulus]
        if offset_file == True:
            counts_off_list = counts_off_array[nonzero_annulus]

        # calculate background
        if offset_file == True:
            sky_phot = do_phot(annulus_list, counts_list, exp_list, offset_list=counts_off_list, sig_clip=2)
        else:
            sky_phot = do_phot(annulus_list, counts_list, exp_list, sig_clip=2)
        

        # -------------------------
        # sky background variation
        # -------------------------

        # define theta around the sky annulus
        delta_x = nonzero_annulus[1] - ellipse_center[0][0]
        delta_y = nonzero_annulus[0] - ellipse_center[0][1]
        theta = np.arccos(delta_x/np.sqrt(delta_x**2 + delta_y**2))
        # go from 0->2pi instead of 0->pi and pi->0 (yay arccos?)
        theta[delta_y < 0] = np.pi + (np.pi - theta[delta_y < 0])
        # convert to degrees
        theta_deg = theta * 180/np.pi
        # shift starting point to match position angle of galaxy
        theta_deg = (theta_deg + (90-pos_angle)) % 360
        #pdb.set_trace()

        # increments of theta for 8 equal-area segments
        delta_theta = np.arctan(minor_diam/major_diam) * 180/np.pi
        delta_list = [delta_theta, 90-delta_theta, 90-delta_theta, delta_theta,
                          delta_theta, 90-delta_theta, 90-delta_theta, delta_theta]
        theta_start = 0

        # array to save results
        seg_phot = np.zeros(len(delta_list))
        seg_phot_err = np.zeros(len(delta_list))
        
        
        for i in range(len(delta_list)):
            # indices of the current segment
            seg = np.where((theta_deg >= theta_start) & (theta_deg < theta_start+delta_list[i]))
            ind = (nonzero_annulus[0][seg[0]], nonzero_annulus[1][seg[0]])
            # list of values within segment
            annulus_list = annulus_im[ind]
            counts_list = hdu_counts[1].data[ind]
            exp_list = hdu_ex[1].data[ind]
            if offset_file == True:
                counts_off_list = counts_off_array[ind]
            # do photometry
            if offset_file == True:
                temp = do_phot(annulus_list, counts_list, exp_list, offset_list=counts_off_list, sig_clip=2)
            else:
                temp = do_phot(annulus_list, counts_list, exp_list, sig_clip=2)
            # save it
            seg_phot[i] = temp['count_rate_per_pix']
            seg_phot_err[i] = temp['count_rate_err_per_pix']
            # next segment
            theta_start += delta_list[i]

        

        pdb.set_trace()

        # -------------------------
        # photometry for each aperture
        # -------------------------

        # initialize a table (or, rather, the rows... turn into table later)
        cols = ['radius','count_rate','count_rate_err',
                    'count_rate_err_bg','count_rate_err_poisson',
                    'mu','mu_err','n_pix']
        units = ['arcsec','counts/sec','counts/sec',
                     'counts/sec','counts/sec',
                     'mag/arcsec2','mag/arcsec2','']
        phot_table = {key:np.zeros(len(annulus_array)-1) for key in cols}

        #for i in range(len(annulus_array)-1):
        for i in [30]:
            
            # define aperture object
            aperture = EllipticalAnnulus(tuple(ellipse_center[0]),
                                             a_in=annulus_array[i]/arcsec_per_pix,
                                             a_out=annulus_array[i+1]/arcsec_per_pix,
                                             b_out=annulus_array[i+1]/arcsec_per_pix * minor_diam/major_diam,
                                             theta=(90+pos_angle)*np.pi/180)
            # make an ApertureMask object with the aperture
            annulus_mask = aperture.to_mask(method='exact')
            # turn aperture into an image
            annulus_im = annulus_mask[0].to_image(hdu_counts[1].data.shape)

            # get total number of pixels
            tot_pix = np.sum(annulus_im)
            
            # make masked version using input ds9 file
            if mask_file is not None:
                annulus_im = annulus_im * mask_image

            # plot things
            #annulus_data = annulus_mask[0].multiply(hdu_counts[1].data)
            #plt.imshow(annulus_mask[0])
            #plt.imshow(annulus_data, origin='lower')
            #plt.imshow(annulus_im, origin='lower')
            #plt.colorbar()

            # get total number of pixels post-mask
            tot_pix_mask = np.sum(annulus_im)

            # list of values within aperture
            nonzero_annulus = np.where(annulus_im > 1e-5)
            annulus_list = annulus_im[nonzero_annulus]
            counts_list = hdu_counts[1].data[nonzero_annulus]
            exp_list = hdu_ex[1].data[nonzero_annulus]
            if offset_file == True:
                counts_off_list = counts_off_array[nonzero_annulus]
            pdb.set_trace()

            # do photometry
            if offset_file == True:
                ann_temp = do_phot(annulus_list, counts_list, exp_list, offset_list=counts_off_list)
            else:
                ann_temp = do_phot(annulus_list, counts_list, exp_list)

            # subtract background
            ann_phot = ann_temp['count_rate_per_pix'] - sky_phot['count_rate_per_pix']
            ann_phot_err = sqrt(ann_temp['count_rate_err_per_pix']**2 +
                                    sky_phot['count_rate_err_per_pix']**2 +
                                    np.std(seg_phot)**2 )

            # multiply by the number of pixels in the annulus to get the total count rate
            
            
            pdb.set_trace()




def make_mask_image(hdu, mask_file):
    """
    Make a mask file (foreground stars, background galaxies) that can be
    applied to each aperture image

    Parameters
    ----------
    hdu : astropy hdu object
        An HDU with a reference image

    mask_file : string
        path+name of ds9 region file with masks


    Returns
    -------
    mask_image : array
        an image of 1s and 0s, where 0s represent masked pixels

    """

    # make a 1s image
    mask_image = np.ones(hdu.data.shape)

    # read in the ds9 file
    regions = read_ds9(mask_file)
    # get ra/dec/radius (all in degrees)
    reg_ra = np.array( [regions[i].center.ra.deg for i in range(len(regions))] )
    reg_dec = np.array( [regions[i].center.dec.deg for i in range(len(regions))] )
    reg_rad_deg = np.array( [regions[i].radius.value for i in range(len(regions))] )/3600
        
    # convert to x/y
    im_wcs = wcs.WCS(hdu.header)
    reg_x, reg_y = im_wcs.wcs_world2pix(reg_ra, reg_dec, 1)
    reg_rad_pix = reg_rad_deg / wcs.utils.proj_plane_pixel_scales(im_wcs)[0]

    # go through the regions and mask
    height, width = mask_image.shape
    y_grid, x_grid = np.ogrid[:height, :width]

    for i in range(len(reg_x)):
        dist_from_center = np.sqrt((x_grid - reg_x[i])**2 + (y_grid-reg_y[i])**2)
        mask = dist_from_center <= reg_rad_pix[i]
        mask_image[mask] = 0

    # return final image
    return mask_image



def do_phot(annulus_list, counts_list, exp_list,
                offset_list=None, sig_clip=None):
    """
    Do photometry!  Given counts/exposure time for a list of pixels (and
    partial pixels), calculate total counts, count rate, and uncertainties.
    If this is for a sky estimate, sigma clipping might be useful.

    Parameters
    ----------
    annulus_list : array
        covering fraction of each pixel (between 0 and 1)

    counts_list : array
        counts in each pixel (not accounting for covering fraction)

    exp_list : array
        exposure time in each pixel (seconds)

    offset_list : array (default=None)
        the offset (in counts) already applied to the counts image

    sig_clip : float (default=None)
        if set, apply a N iterations of sigma clipping to count rate values
        (useful for estimating sky background)

    Returns
    -------
    phot : dictionary
        results of the photometry
        keys are 'count_rate', 'count_rate_err'

    """

    # account for poisson errors from counts and any offset errors
    if offset_list is not None:
        pois_err = np.sqrt(counts_list - offset_list)
        off_err = np.sqrt(np.abs(offset_list))
    else:
        pois_err = np.sqrt(counts_list)
        off_err = np.zeros(counts_list.shape)
    # the combined error   
    counts_err = np.sqrt( pois_err**2 + off_err**2 )

    # multiply by covering fraction
    counts = counts_list * annulus_list
    counts_err *= annulus_list
    pois_err *= annulus_list
    off_err *= annulus_list

    # count rate
    count_rate = counts / exp_list
    count_rate_err = counts_err / exp_list
    count_rate_pois_err = pois_err / exp_list
    count_rate_off_err = off_err / exp_list

    # do sigma clipping, if chosen
    if sig_clip is not None:
        pix_clip = sigma_clip(count_rate, sigma=3, iters=sig_clip)
        count_rate = count_rate[~pix_clip.mask]
        count_rate_err = count_rate_err[~pix_clip.mask]
        count_rate_pois_err = count_rate_pois_err[~pix_clip.mask]
        count_rate_off_err = count_rate_off_err[~pix_clip.mask]


    # total count rate
    tot_count_rate = np.sum(count_rate)
    tot_count_rate_err = np.sqrt(np.sum( count_rate_err**2 ))
    tot_count_rate_pois_err = np.sqrt(np.sum( count_rate_pois_err**2 ))
    tot_count_rate_off_err = np.sqrt(np.sum( count_rate_off_err**2 ))

    # total count rate per pixel
    tot_count_rate_per_pix = tot_count_rate/len(count_rate)
    tot_count_rate_per_pix_err = tot_count_rate_err/len(count_rate)
    tot_count_rate_per_pix_pois_err = tot_count_rate_pois_err/len(count_rate)
    tot_count_rate_per_pix_off_err = tot_count_rate_off_err/len(count_rate)

    pdb.set_trace()

    # return results
    return {'count_rate_per_pix':tot_count_rate_per_pix,
                'count_rate_err_per_pix':tot_count_rate_per_pix_err,
                'count_rate_pois_err_per_pix':tot_count_rate_per_pix_pois_err,
                'count_rate_off_err_per_pix':tot_count_rate_per_pix_off_err }
