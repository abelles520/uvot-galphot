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
        counts) have already been applied to the counts images

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
        
        # WCS for the images
        wcs_counts = wcs.WCS(hdu_counts[1].header)
        arcsec_per_pix = wcs_counts.wcs.cdelt[1] * 3600
        
        # ellipse center
        #ellipse_center = SkyCoord(ra=center_ra*u.deg, dec=center_dec*u.deg)
        ellipse_center = wcs_counts.wcs_world2pix([[center_ra,center_dec]], 0)
        
        # array of annuli
        annulus_array = np.arange(0, major_diam*1.2, ann_width)# * u.arcsec 


        # initialize a table (or, rather, the rows... turn into table later)
        table_rows = []
        
        
        # go through each aperture
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
            nonzero_annulus = np.where(annulus_im > 0)
            annulus_list = annulus_im[nonzero_annulus]
            counts_list = hdu_counts[1].data[nonzero_annulus]
            exp_list = hdu_ex[1].data[nonzero_annulus]
            if offset_file == True:
                with fits.open(label+'sk_off.fits') as hdu_off:
                    counts_off_list = hdu_off[1].data[nonzero_annulus]
            pdb.set_trace()
            
            # do some photometry
            phot_table = aperture_photometry(hdu_counts[1].data, aperture)


            
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
