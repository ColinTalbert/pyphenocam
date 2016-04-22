from pyphenocam import headerextraction

from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_raises
from nose.tools import raises


class Testheaderextraction(object):

    def test_init(self):
        fname = "tests/data/quickbird_2015_06_01_150006.jpg"
        temperature, exposure = headerextraction.get_temp_exposure(fname)

        assert_equal(temperature, 45.5)
        assert_equal(exposure, 76)
