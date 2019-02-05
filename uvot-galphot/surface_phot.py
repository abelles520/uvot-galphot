import numpy as np
import matplotlib.pyplot as plt

from photutils import SkyEllipticalAperture, SkyEllipticalAnnulus, aperture_photometry, EllipticalAnnulus, EllipticalAperture
from astropy.io import fits
from astropy import units as u
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.stats import biweight_location, sigma_clip
from regions import read_ds9

import pdb


def surface_phot(label, center_ra, center_dec, major_diam, minor_diam, pos_angle,
                     ann_width, zeropoint, zeropoint_err=0.0,
                     mask_file=None, offset_file=False,
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

    zeropoint_err : float (default=0)
        uncertainty for the zeropoint

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

            # for some unknown reason (uvotimsum bug?), the offset file could have NaNs
            # -> mask them
            if np.sum(~np.isfinite(counts_off_array)) > 0:
                bad_pix = np.where(np.isfinite(counts_off_array) == 0)
                # either add to existing mask
                if mask_file is not None:
                    mask_image[bad_pix] = 0
                # or make new mask
                if mask_file is None:
                    mask_image = np.ones(counts_off_array.shape)
                    mask_image[bad_pix] = 0
                    mask_file = 'mask_from_offset_nans'

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
        sky_seg_phot = np.zeros(len(delta_list))
        sky_seg_phot_err = np.zeros(len(delta_list))
        
        
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
            sky_seg_phot[i] = temp['count_rate_per_pix']
            sky_seg_phot_err[i] = temp['count_rate_err_per_pix']
            # next segment
            theta_start += delta_list[i]

        

        # -------------------------
        # photometry for each annulus
        # -------------------------

        # initialize a table (or, rather, the rows... turn into table later)
        cols_ann = ['radius','count_rate','count_rate_err',
                    'count_rate_err_poisson','count_rate_err_bg',
                    'mu','mu_err','n_pix']
        units_ann = ['arcsec','cts/sec','cts/sec',
                     'cts/sec','cts/sec',
                     'ABmag/arcsec2','ABmag/arcsec2','']
        dtypes = ['%9.3f','%9f','%9f',
                      '%9f','%9f',
                      '%9f','%9f','%9f']
        phot_dict_ann = {key:np.zeros(len(annulus_array)-1) for key in cols_ann}

        for i in range(len(annulus_array)-1):
        #for i in range(0,5):

            # save radius
            phot_dict_ann['radius'][i] = annulus_array[i+1]

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

            # get total number of pixels (using ellipse areas, in case some of the aperture is off the image)
            #tot_pix = np.sum(annulus_im)
            area_out = np.pi * annulus_array[i+1]/arcsec_per_pix * annulus_array[i+1]/arcsec_per_pix * minor_diam/major_diam
            area_in = np.pi * annulus_array[i]/arcsec_per_pix * annulus_array[i]/arcsec_per_pix * minor_diam/major_diam
            tot_pix = area_out - area_in
            tot_arcsec2 = tot_pix * arcsec_per_pix**2
            phot_dict_ann['n_pix'][i] = tot_pix
            
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

            # do photometry
            if offset_file == True:
                ann_temp = do_phot(annulus_list, counts_list, exp_list, offset_list=counts_off_list)
            else:
                ann_temp = do_phot(annulus_list, counts_list, exp_list)

            # subtract background
            ann_phot_per_pix = ann_temp['count_rate_per_pix'] - sky_phot['count_rate_per_pix']
            ann_phot_per_pix_err = np.sqrt(ann_temp['count_rate_err_per_pix']**2 +
                                            sky_phot['count_rate_err_per_pix']**2 +
                                            np.std(sky_seg_phot)**2 )
            ann_phot_per_pix_pois_err = ann_temp['count_rate_pois_err_per_pix']
            ann_phot_per_pix_bg_err = np.sqrt(ann_temp['count_rate_off_err_per_pix']**2 +
                                                sky_phot['count_rate_err_per_pix']**2 +
                                                np.std(sky_seg_phot)**2 )

            # multiply by the number of pixels in the annulus to get the total count rate
            ann_phot = ann_phot_per_pix * tot_pix
            ann_phot_err = ann_phot_per_pix_err * tot_pix
            ann_phot_pois_err = ann_phot_per_pix_pois_err * tot_pix
            ann_phot_bg_err = ann_phot_per_pix_bg_err * tot_pix

            phot_dict_ann['count_rate'][i] = ann_phot
            phot_dict_ann['count_rate_err'][i] = ann_phot_err
            phot_dict_ann['count_rate_err_poisson'][i] = ann_phot_pois_err
            phot_dict_ann['count_rate_err_bg'][i] = ann_phot_bg_err

            # convert to surface brightness
            # - counts/sec/arcsec2
            ann_phot_arcsec2 = ann_phot / tot_arcsec2
            ann_phot_arcsec2_err = ann_phot_err / tot_arcsec2
            # - mag/arcsec2
            mag_arcsec2 = -2.5 * np.log10(ann_phot_arcsec2) + zeropoint
            mag_arcsec2_err = np.sqrt( ( 2.5/np.log(10) * ann_phot_arcsec2_err/ann_phot_arcsec2 )**2 +
                                           zeropoint_err**2 )

            phot_dict_ann['mu'][i] = mag_arcsec2
            phot_dict_ann['mu_err'][i] = mag_arcsec2_err

            
            #[print(k+': ', phot_dict_ann[k][i]) for k in cols_ann]
            #pdb.set_trace()


        # make big numpy array
        data_array = np.column_stack(tuple([phot_dict_ann[key] for key in cols_ann]))
        # save it to a file
        np.savetxt(label+'phot_annprofile.dat', data_array,
                       header=' '.join(cols_ann) + '\n' + ' '.join(units_ann),
                       delimiter='  ', fmt=dtypes)


        # -------------------------
        # total photometry within each radius
        # -------------------------

        # initialize a table (or, rather, the rows... turn into table later)
        cols_tot = ['radius','count_rate','count_rate_err',
                    'count_rate_err_poisson','count_rate_err_bg',
                    'mag','mag_err','n_pix']
        units_tot = ['arcsec','cts/sec','cts/sec',
                     'cts/sec','cts/sec',
                     'ABmag','ABmag','']
        dtypes = ['%9.3f','%9f','%9f',
                      '%9f','%9f',
                      '%9f','%9f','%9f']
        phot_dict_tot = {key:np.zeros(len(annulus_array)-1) for key in cols_tot}

        for i in range(len(annulus_array)-1):
        #for i in range(0,5):

            # save radius
            phot_dict_tot['radius'][i] = annulus_array[i+1]

            # define aperture object
            aperture = EllipticalAperture(tuple(ellipse_center[0]),
                                             a=annulus_array[i+1]/arcsec_per_pix,
                                             b=annulus_array[i+1]/arcsec_per_pix * minor_diam/major_diam,
                                             theta=(90+pos_angle)*np.pi/180)
            # make an ApertureMask object with the aperture
            annulus_mask = aperture.to_mask(method='exact')
            # turn aperture into an image
            annulus_im = annulus_mask[0].to_image(hdu_counts[1].data.shape)
            
            # get total number of pixels (using ellipse areas, in case some of the aperture is off the image)
            #tot_pix = np.sum(annulus_im)
            tot_pix = np.pi * annulus_array[i+1]/arcsec_per_pix * annulus_array[i+1]/arcsec_per_pix * minor_diam/major_diam
            tot_arcsec2 = tot_pix * arcsec_per_pix**2
            phot_dict_tot['n_pix'][i] = tot_pix
            
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

            # do photometry
            if offset_file == True:
                tot_temp = do_phot(annulus_list, counts_list, exp_list, offset_list=counts_off_list)
            else:
                tot_temp = do_phot(annulus_list, counts_list, exp_list)

            # subtract background
            tot_phot_per_pix = tot_temp['count_rate_per_pix'] - sky_phot['count_rate_per_pix']
            tot_phot_per_pix_err = np.sqrt(tot_temp['count_rate_err_per_pix']**2 +
                                            sky_phot['count_rate_err_per_pix']**2 +
                                            np.std(sky_seg_phot)**2 )
            tot_phot_per_pix_pois_err = tot_temp['count_rate_pois_err_per_pix']
            tot_phot_per_pix_bg_err = np.sqrt(tot_temp['count_rate_off_err_per_pix']**2 +
                                                sky_phot['count_rate_err_per_pix']**2 +
                                                np.std(sky_seg_phot)**2 )

            # multiply by the number of pixels in the annulus to get the total count rate
            tot_phot = tot_phot_per_pix * tot_pix
            tot_phot_err = tot_phot_per_pix_err * tot_pix
            tot_phot_pois_err = tot_phot_per_pix_pois_err * tot_pix
            tot_phot_bg_err = tot_phot_per_pix_bg_err * tot_pix

            phot_dict_tot['count_rate'][i] = tot_phot
            phot_dict_tot['count_rate_err'][i] = tot_phot_err
            phot_dict_tot['count_rate_err_poisson'][i] = tot_phot_pois_err
            phot_dict_tot['count_rate_err_bg'][i] = tot_phot_bg_err

            # convert to magnitudes
            mag = -2.5 * np.log10(tot_phot) + zeropoint
            mag_err = np.sqrt( ( 2.5/np.log(10) * tot_phot_err/tot_phot )**2 +
                                           zeropoint_err**2 )

            phot_dict_tot['mag'][i] = mag
            phot_dict_tot['mag_err'][i] = mag_err

            
            #[print(k+': ', phot_dict_tot[k][i]) for k in cols_tot]
            #pdb.set_trace()


        # make big numpy array
        data_array = np.column_stack(tuple([phot_dict_tot[key] for key in cols_tot]))
        # save it to a file
        np.savetxt(label+'phot_totprofile.dat', data_array,
                       header=' '.join(cols_tot) + '\n' + ' '.join(units_tot),
                       delimiter='  ', fmt=dtypes)



        # -------------------------
        # calculate magnitudes
        # -------------------------

        # asymptotic: plot accumulated flux vs gradient of accumulated flux, then get y-intercept
        # (see Gil de Paz et al 2007, section 4.3)
        
        # - grab points with the last part of flux accumulation
        use_ind = np.where(phot_dict_tot['count_rate'] >= 0.9 * np.max(phot_dict_tot['count_rate']))
        use_rad = phot_dict_tot['radius'][use_ind]
        use_cr = phot_dict_tot['count_rate'][use_ind]
        use_cr_err = phot_dict_tot['count_rate_err'][use_ind]
        grad = np.diff(use_cr) / np.diff(use_rad)
        
        # - bootstrap linear fit
        fit_boot = boot_lin_fit(grad, use_cr[1:], use_cr_err[1:])

        # - convert flux to mags
        asym_mag = -2.5 * np.log10(fit_boot['int']) + zeropoint
        asym_mag_err = np.sqrt( ( 2.5/np.log(10) * fit_boot['int_err']/fit_boot['int'] )**2 +
                                    zeropoint_err**2 )

        # - save it
        np.savetxt(label+'phot_asymmag.dat',
                       np.array([[fit_boot['int'], fit_boot['int_err'], asym_mag, asym_mag_err]]),
                       header='count_rate count_rate_err mag mag_err\ncts/sec cts/sec ABmag ABmag',
                       delimiter='  ', fmt=['%9f','%9f','%9f','%9f'])


        # - make plots
        if False:
            fig = plt.figure(figsize=(6,5), num='flux gradient stuff')
            plt.errorbar(grad, use_cr[1:],
                         yerr=use_cr_err[1:],
                         marker='.', color='black', ms=5, mew=0,
                         linestyle='-', ecolor='black', capsize=0)
            plt.plot(np.linspace(0,np.max(grad),50),
                     fit_boot['slope']*np.linspace(0,np.max(grad),50) + fit_boot['int'],
                     marker='.', ms=0, mew=0,
                     color='dodgerblue', linestyle='-')
            ax = plt.gca()
            ax.set_ylabel('Accumulated Flux (counts/sec)')
            ax.set_xlabel('Gradient of Accumulated Flux')
            plt.tight_layout()
            pdb.set_trace()
        
            fig = plt.figure(figsize=(6,5), num='flux stuff')
            plt.errorbar(use_rad/60, use_cr,
                         yerr=use_cr_err,
                         marker='.', color='black', ms=5, mew=0,
                         linestyle='-', ecolor='black', capsize=0)
            ax = plt.gca()
            ax.set_ylabel('Accumulated Flux (counts/sec)')
            ax.set_xlabel('Radius (arcmin)')
            plt.tight_layout()
            pdb.set_trace()



        # total: outermost annular point with S/N > 2 -> get accumulated flux within that radius
        sn = phot_dict_ann['count_rate'] / phot_dict_ann['count_rate_err']
        ind = np.nonzero(sn > 2)[0][-1]
        max_radius = phot_dict_tot['radius'][ind]
        tot_mag = phot_dict_tot['mag'][ind]
        tot_mag_err = phot_dict_tot['mag_err'][ind]
        # save it
        np.savetxt(label+'phot_totmag.dat',
                       np.array([[phot_dict_tot['count_rate'][ind], phot_dict_tot['count_rate_err'][ind], tot_mag, tot_mag_err]]),
                       header='count_rate count_rate_err mag mag_err\ncts/sec cts/sec ABmag ABmag',
                       delimiter='  ', fmt=['%9f','%9f','%9f','%9f'])
        

        # return various useful info
        return {'phot_dict_ann':phot_dict_ann, 'cols_ann':cols_ann, 'units_ann':units_ann,
                    'phot_dict_tot':phot_dict_tot, 'cols_tot':cols_tot, 'units_tot':units_tot,
                    'sky_phot':sky_phot,
                    'sky_seg_phot':sky_seg_phot, 'sky_seg_phot_err':sky_seg_phot_err,
                    'asym_mag':asym_mag, 'asym_mag_err':asym_mag_err,
                    'tot_mag':tot_mag, 'tot_mag_err':tot_mag_err}



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
    tot_count_rate_per_pix = tot_count_rate / np.sum(annulus_list)
    tot_count_rate_per_pix_err = tot_count_rate_err / np.sum(annulus_list)
    tot_count_rate_per_pix_pois_err = tot_count_rate_pois_err / np.sum(annulus_list)
    tot_count_rate_per_pix_off_err = tot_count_rate_off_err / np.sum(annulus_list)

    # return results
    return {'count_rate_per_pix':tot_count_rate_per_pix,
                'count_rate_err_per_pix':tot_count_rate_per_pix_err,
                'count_rate_pois_err_per_pix':tot_count_rate_per_pix_pois_err,
                'count_rate_off_err_per_pix':tot_count_rate_per_pix_off_err }



def boot_lin_fit(x_in, y_in, y_err_in, n_boot=500):
    """
    Linear fit with bootstrap!
    """

    # initialize arrays
    slope_array = np.zeros(n_boot)
    int_array = np.zeros(n_boot)

    for b in range(n_boot):

        # draw a set of new y values
        y_new = np.random.normal(y_in, y_err_in)

        # linear fit
        fit = np.polyfit(x_in, y_new, 1)

        # save results
        slope_array[b] = fit[0]
        int_array[b] = fit[1]

    # extract best fits and errors
    s16, s50, s84 = np.percentile(slope_array, [16,50,84])
    i16, i50, i84 = np.percentile(int_array, [16,50,84])
    best_fit = {'slope':s50, 'slope_err':(s84-s16)/2,
                    'int':i50, 'int_err':(i84-i16)/2}


    return best_fit
