import os as _os
import numpy as _np
from datetime import datetime as _datetime
import math as _math
import config

from bs4 import BeautifulSoup as _BS
import urllib
import urllib2

__all__ = ['parse_fname', 'process_files']


def fcl(df, dtObj):
    return df.iloc[_np.argmin(_np.abs(df.index.to_pydatetime() - dtObj))]


def parse_fname(fname):

    fname_noext = _os.path.splitext(fname)[0]
    parts = fname_noext.split("_")

    sitename = parts[0]
    ir_file = parts[1] == "IR"

    if ir_file:
        y, m, d, t = parts[2:]
    else:
        y, m, d, t = parts[1:]

    h, mm, s = t[:2], t[2:4], t[4:]

    return ir_file, sitename, _datetime(*[int(i) for i in [y, m, d, h, mm, s]])


def get_matched_fnames(dname):
    fnames = [_os.path.join(dname, f) for f in _os.listdir(dname)
              if f.endswith('.jpg') and not "_IR_" in f]
    ir_fnames = [_os.path.join(dname, f) for f in _os.listdir(dname)
                 if f.endswith('.jpg') and "_IR_" in f]

    matched_fnames = []
    for fname in fnames:
        try:
            matched = [
                f for f in ir_fnames if f.replace('_IR_', '_') == fname][0]
            matched_fnames.append((fname, matched))
        except IndexError:
            print "skipping", fname

    return matched_fnames


def get_closest_ir_fname(fname):

    just_fname = _os.path.split(fname)[-1]
    irfnameurl = config.get_url('IRFNAMEURL')
    _, sitename, dt = parse_fname(just_fname)
    url = irfnameurl.format(sitename, dt.year, dt.month, dt.day)

    html_page = urllib2.urlopen(url)
    soup = _BS(html_page, "lxml")
    ir_lookup = {}
    for link in soup.findAll('a'):
        href = link.get('href')
        if href and href.endswith('jpg'):
            ir_fname = href.split('/')[-1]
            ir_dt = parse_fname(ir_fname)[-1]
            ir_lookup[ir_dt] = ir_fname
   
    return ir_lookup[nearest_date(ir_lookup.keys(), dt)]


def convert_fname_to_ir(fname):
    """given a filename returns the closest IR equivelent file
    """
    dname, justfname = _os.path.split(fname)

    ir_fname = get_closest_ir_fname(fname)
    return _os.path.join(dname, ir_fname)


def dist(x1, y1, x2, y2):
    """Returns the distance between two points (X1, Y1) and (X2, Y2)
    """
    return _np.linalg.norm(_np.array((x1, y1)) - _np.array((x2, y2)))

def nearest_date(dates, pivot):
    return min(dates, key=lambda x: abs(x - pivot))