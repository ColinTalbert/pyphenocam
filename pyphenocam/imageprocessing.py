import os as _os

import numpy as _np

from datetime import datetime as _datetime
import math as _math

import skimage as _skimage
import skimage.io as _skimageio

import utils
import headerextraction

__all__ = ['get_corrected_ndvi']


def get_photo_array(fname):
    """returns a numpy array with the data from a photo
    """
    return _skimage.io.imread(fname)


def get_boolean_photo_array(fname):
    """given a filename with an ROI returns a boolean array
    """
    roi_data = get_photo_array(fname).astype('bool')
    return roi_data


def get_corrected_ndvi(fname):
    """Returns an array of the corrected NDVI from a set of phenocam images 
    calculated according to the methodology of Petach et al
    """

    fname_ir = utils.convert_fname_to_ir(fname)
    temperature, exposure = headerextraction.get_temp_exposure(
        fname)
    irtemperature, irexposure = headerextraction.get_temp_exposure(
        fname_ir)

    return _get_corrected_ndvi(fname, fname_ir, exposure, irexposure)


def get_roi_contours(roi):
    return _skimage.measure.find_contours(roi, 0.5)


def _get_corrected_ndvi(fname, fname_ir, exposure, nir_exposure):
    """Returns an array of the corrected NDVI from a set of phenocam images 
    calculated according to the methodology of Petach et al
    """

    r, g, b = [_np.squeeze(a)
               for a in _np.split(_skimageio.imread(fname), 3, axis=2)]
    z = _skimageio.imread(fname_ir)[:, :, 0].astype(float)
    y = 0.3 * r + 0.59 * g + 0.11 * b
    x = z - y

    z_prime = z / (_math.sqrt(nir_exposure))
    r_prime = r / (_math.sqrt(exposure))
    y_prime = y / (_math.sqrt(exposure))
    x_prime = z_prime - y_prime
    # x_prime = (x_prime+255)/2

    ndvi = ((z - r) / (z + r))
    ndvi_c = ((x_prime - r_prime) / (x_prime + r_prime))

    a = 0.53
    b_ = 0.83
    ndvi_c2 = a * ndvi_c + b_

    ndvi_c2[ndvi_c2 > 1.0] = 1.0
    ndvi_c2[ndvi_c2 < -1.0] = -1.0

    return ndvi_c2


def process_files(dname, rois=None):

    results = {}
    matched_fnames = utils.get_matched_fnames(dname)

    for fname, irfname in matched_fnames:
        short_fname = _os.path.split(fname)[1]
        short_fname_ir = _os.path.split(irfname)[1]
        _, file_date = utils.parse_fname(short_fname)

        file_done = False
        attempts = 0

        while not file_done and attempts < 10:
            try:
                ans = {}
                temperature, exposure = headerextraction.get_temp_exposure(
                    fname)
                irtemperature, irexposure = headerextraction.get_temp_exposure(
                    irfname)

                ans = {'filename': short_fname, 'date': file_date, 'IR': False,
                       'monthdname': _os.path.split(dname)[1],
                       'temperature': temperature, 'exposure': exposure}

                ans_ir = {'filename': short_fname_ir, 'date': file_date, 'IR': True,
                          'monthdname': _os.path.split(dname)[1],
                          'temperature': irtemperature, 'exposure': irexposure}

                if rois:
                    ndvi = _get_corrected_ndvi(
                        fname, irfname, exposure, irexposure)
                    ndvi_ans = {}
                    for label, roi in rois.iteritems():
                        ndvi_ans[label] = _np.nanmean(ndvi[roi])

                    ans.update(ndvi_ans)
                    ans_ir.update(ndvi_ans)

                results[short_fname] = ans
                results[short_fname_ir] = ans_ir
                file_done = True
                print '.',
            except Exception, e:
                print 'x',
                attempts += 1
                if attempts == 10:
                    print str(e)
                    print ans
                    print fname, 'FAILED!!!!!!!!!!!!!!!!!!!!!!!!\n'

    return results
