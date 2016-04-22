import pickle as _pickle
from pkg_resources import resource_filename as _resource_filename

import numpy as _np

import skimage as _skimage
from skimage.color.adapt_rgb import adapt_rgb as _adapt_rgb
import skimage.io as _skimageio
from skimage.measure import label as _label
from skimage import filters as _filters

__all__ = ['get_temp_exposure']


def _get_lines(fname):
    """Returns the portions of the image cooresponding to the
    first three lines of text in the header.
    These are returned as 3D numpy arrays.
    """
    @_adapt_rgb(_skimage.color.adapt_rgb.hsv_value)
    def sobel_hsv(image):
        return _filters.sobel(image)

    data = _skimageio.imread(fname)

    l1_range = data[0:27, 0:850, :]
    l2_range = data[28:55, 0:500, :]
    l3_range = data[56:83, 0:350:]
    l4_range = data[84:111, 0:350:]

    l1_edges = _np.sum(
        _skimage.exposure.rescale_intensity(1 - sobel_hsv(l1_range)), axis=2) < 2
    l2_edges = _np.sum(
        _skimage.exposure.rescale_intensity(1 - sobel_hsv(l2_range)), axis=2) < 2
    l3_edges = _np.sum(
        _skimage.exposure.rescale_intensity(1 - sobel_hsv(l3_range)), axis=2) < 2
    l4_edges = _np.sum(
        _skimage.exposure.rescale_intensity(1 - sobel_hsv(l4_range)), axis=2) < 2

    try:
        l1_right_edge = 0 + \
            _np.where(_np.sum(l1_edges, axis=0) >= 25)[0].max()
    except ValueError:
        l1_right_edge = 850
    if l1_right_edge < 2:
        l1_right_edge = 850

    try:
        l2_right_edge = 0 + \
            _np.where(_np.sum(l2_edges, axis=0) >= 25)[0].max()
    except ValueError:
        l2_right_edge = 500
    if l2_right_edge < 2:
        l2_right_edge = 500

    try:
        l3_right_edge = 0 + \
            _np.where(_np.sum(l3_edges, axis=0) >= 25)[0].max()
    except ValueError:
        l3_right_edge = 350
    if l3_right_edge < 2:
        l3_right_edge = 350

    try:
        l4_right_edge = 0 + \
            _np.where(_np.sum(l4_edges, axis=0) >= 25)[0].max()
    except ValueError:
        l4_right_edge = 350
    if l4_right_edge < 2:
        l4_right_edge = 350

    line1 = data[0:27, :l1_right_edge, :]
    line2 = data[28:55, :l2_right_edge, :]
    line3 = data[56:83, :l3_right_edge, :]
    line4 = data[84:111, :l4_right_edge, :]
    return line1, line2, line3, line4


def get_header_contents(fname):
    """
    """
    line1, line2, line3, line4 = _get_lines(fname)

    output = {}
    for line in [('line1', line1),
                 ('line2', line2),
                 ('line3', line3),
                 ('line4', line4), ]:
        if len(_np.unique(line[1])) > 1:
            line_binary = _get_binary(line[1])
            output[line[0]] = _extract_digits(line_binary)
        else:
            output[line[0]] = ''

    return output


def get_exposure(fname):

    header_contents = get_header_contents(fname)

    # 4 line header, readingma style
    if "Exposure:" in header_contents['line4']:
        exposure_line = header_contents['line4']
        return exposure_line[exposure_line.index(':') + 1:]
    elif "Exposure:" in header_contents['line3']:  # quickbird style
        exposure_line = header_contents['line3']
        return exposure_line[exposure_line.index(':') + 1:]
    elif "Exposure:" in header_contents['line1']:  # harvard style
        exposure_line = header_contents['line1']
        return exposure_line[exposure_line.index('Exposure:') + 9:]
    else:
        return "didn't work"


def _get_binary(raw_data):
    """Given an image array returns the image converted to a
    "cleaned up" binary array.
    """
    #try:
    #    gray_data = _skimage.color.rgb2gray(raw_data)
    #except:
    #    gray_data = raw_data

    try:
        thresh = _skimage.filters.threshold_otsu(raw_data[:, :, 0])
        binary = raw_data[:, :, 0] > thresh
        #binary = binary[:, :, 0]
    except ValueError:
        print 'valueerror'
        binary = _np.ones(raw_data.shape).astype('bool')

    return binary


def _segment(binary):
    """Parses a chunk of text provided in a binary numpy array.
    Returns:
        a tuple with a list of unique tags for each each individual 'letter'
        a numpy array with these tags inserted where the 'letters' are.
    """
    label_image = _label(binary)

    digit_labels = []
    for column in label_image.T:
        col_vals = _np.unique(column)
        if len(col_vals) == 1:
            pass
        elif len(col_vals) == 2 and col_vals[1] not in digit_labels:
            digit_labels.append(col_vals[1])
        elif len(col_vals) == 3:
            label_image[label_image == col_vals[2]] = col_vals[1]
            if col_vals[1] not in digit_labels:
                digit_labels.append(col_vals[1])

    return digit_labels, label_image


def _get_digit(binary, label_image, digit):
    """Extracts a 'letter' from a labeled image corresponding to the specified tag.
    returns a binary numpy array with just that digit's pixels.
    """
    ymin, ymax = _np.where(label_image == digit)[0].min(), \
        _np.where(label_image == digit)[0].max() + 1
    xmin, xmax = _np.where(label_image == digit)[1].min(), \
        _np.where(label_image == digit)[1].max() + 1

    return binary[ymin:ymax, xmin:xmax]


def _return_num(num_image):
    """Given a binary array of a single digit
    returns the digit (string character) of the matching image in our DIGITDICT
    """
    for digit, digit_image in DIGITDICT.iteritems():
        if _np.array_equal(digit_image, num_image):
            return digit

    return -1


def _return_char(num_image):
    """Given a binary array of a single digit
    returns the digit (string character) of the matching image in our DIGITDICT
    """
    for digit, digit_image in DIGITDICT_FULL.iteritems():
        if _np.array_equal(digit_image, num_image):
            return digit

    return -1


def _extract_digits(binary):
    """Given a binary chunk (A single line) of text
    Returns the numeric digits it contains.

    Ignores letters and noise.
    """
    digit_labels, labeled_image = _segment(binary)

    ans = ""
    for digit_label in digit_labels:
        try:
            num = _return_char(_get_digit(binary, labeled_image, digit_label))
        except ValueError:
            num = -1
        if num != -1:
            ans += num
    return ans


def get_temp_exposure(fname):
    """Given a filename of a phenocam image
    Returns the temperature and exposure extracted from the image.
    """
    title, temperature_line, exposure_line = _get_lines(fname)
    binary_temperature = _get_binary(temperature_line)
    temp_str = _extract_digits(binary_temperature)
    temperature = float(temp_str)

    binary_exposure = _get_binary(exposure_line)
    exposure = int(_extract_digits(binary_exposure))

    return temperature, exposure

# load a dictionary with the binary pattern for each numeric digit
_digits_fname = _resource_filename(__name__, 'data/DIGITDICT.p')
DIGITDICT = _pickle.load(open(_digits_fname, "rb"))

# load a dictionary with the binary pattern for each character
_digits_full_fname = _resource_filename(__name__, 'data/DIGITDICT_FULL.p')
DIGITDICT_FULL = _pickle.load(open(_digits_full_fname, "rb"))
