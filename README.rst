Photometry of UVOT galaxy images
--------------------------------

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

Do surface brightness photometry for galaxy images taken with UVOT.
This assumes images have already been stacked/processed (e.g., with 
`uvot-mosaic <https://github.com/UVOT-data-analysis/uvot-mosaic>`_).

License
-------

This project is Copyright (c) Lea Hagen and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.


Documentation
-------------
There are three main files used to do surface brightness photometry of Swift/UVOT
galaxy images: surface_phot.py, phot_plot.py, make_aperture_image.py.

An example of the workflow can be seen in the gal_phot_pipeline.py file. The basic workflow is as follows:

#.  Step 0: Process raw UVOT data

    *  This is done using the uvot-mosiac repository. 

    *  The summed and stacked images and exposure maps are needed to run the photometry code. 

#.  Step 1: surface_phot.py

    *  Main function here is surface_phot, which does the photometry calculation. The photometry is calculated in two ways, total/aperture photometry and asymptotic, where the surface brightness profile is extrapolated to a aperture which no longer accumulates galaxy flux. 
   
    *  Arguments: label, center_ra, center_dec, major_diam, minor_diam, pos_angle, ann_width, zeropoint, zeropoint_err=0.0, aperture_factor=1.0, sky_aperture_factor=1.0, mask_file=None, offset_file=False, verbose=False
   
    *  label is a string with the naming convention for your processed data files, the galaxy position arguments are fairly straightforward but note that the surface_phot function takes the SEMI major/minor axes, the ann_width is the width of the annuli used, zeropoint and err are self-explanatory, aperture factors are a multiplicative factor that can resize the aperture parameters passes (this is useful for diagnosing problematic photometry), the mask_file is the name of the DS9 region file masking foreground stars, offset_file is the name of the offset file if the offset correction is applied, and verbose is a boolean can toggle extra information printed to the terminal

    *  A more complete explanation of arguments can be found in the docstrings.  

    *  Returns: The main output of this function are a text files ending with ''

#.  Step 2: phot_plot.py

    *  This function plots the surface brightness profiles from surface_phot. 
   
    *  Arguments:
   
    *  Returns: 

#.  Step 3 (optional): make_aperture_image.py

    *  This step is optional. In order to visualize the aperture and sky annulus used in the photometry, this code will plot the sky aperture in red and the outer edge of the sky annulus in blue. By default, the sky annulus starts at the end of the galaxy's aperture. The relevant function in this file is make_aperture_image.  
   
    *  An important note: this code takes the major and minor axes as arguments. This is different compared to  
    the functions in surface_phot.py.


Contributing
------------

New contributions and contributors are always welcome!  Contact
@lea-hagen for more information.
