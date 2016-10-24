# pyphenocam

import os as _os
import config

import numpy as np

from bs4 import BeautifulSoup as _BS
import urllib
import urllib2
import re

import pandas as pd

import utils
import imageprocessing

__all__ = ['_get_lines', 'parse_fname', 'process_files']


def get_sites_df():
    """Returns a pandas data frame with a list of all the phenocam sites
    """
    url = "http://phenocam.sr.unh.edu/webcam/network/table"
    df = pd.read_html(url)[0]
    df.columns = ['site', 'lat', 'lon', 'elevation', 'description']
    return df
    #return pd.read_csv(config.get_url('SITEURL'))

SITES_DF = get_sites_df()


def get_sitenames():
    """returns a list of all the site names in the network
    """
    #allsites = get_sites_df()
    #return list(allsites.site.unique())

    return list(SITES_DF.site)
    #url = "http://phenocam.sr.unh.edu/webcam/gallery"
    #html_page = urllib2.urlopen(url)
    #soup = _BS(html_page, "lxml")
    #sites = []
    #for link in soup.findAll('a'):
    #    href = link.get('href')
    #    if href and href.startswith('/webcam/sites'):
    #        sites.append(href.split('/')[-2])
    #return sites


def get_site(sitename='harvard', cache_dname=None, load_all=False):
    """
    """
    if not sitename in get_sitenames():
        raise Exception, "Site {} not in network".format(sitename)

    if not cache_dname:
        site_dname = config.get_cache_dname()
    else:
        site_dname = cache_dname
        config.set_cache_dname(site_dname)

    return SiteData(sitename, site_dname, load_all=load_all)

ROI_TYPES = {'AG': 'Agriculture',
             'DB': 'Deciduous Broadleaf',
             'EB': 'Evergreen Broadleaf',
             'DN': 'Deciduous Needleaf',
             'EN': 'Evergreen Needleleaf',
             'GR': 'Grassland',
             'NV': 'Non-vegetated',
             'RF': 'Reference Panel',
             'SH': 'Shrub',
             'UN': 'Understory',
             'WL': 'Wetland',
             'XX': 'Mixed/Canopy/Other'}


class SiteData():

    """SiteData is the main class that encapsulates the data for a single site
        """

    def __init__(self, sitename, site_dname, load_all=False):
        """sitename -> is the name of the site to initialize,
                this must be an exact match for one of the strings in get_sitenames
            site_dname -> path to the local directory we'll be saving phenocam
                images and data into
            load_all -> boolean flag indicating whether to download all data 
                immediately.  By default data are downloaded as needed 
        """
        if not _os.path.exists(site_dname):
            _os.makedirs(site_dname)
        self.site_dname = site_dname

        self.sitename = sitename
        self.csvs = []
        self.pngs = []
        self.tifs = []
        self.others = []
        self.roi_url = "http://phenocam.sr.unh.edu/data/archive/{}/ROI".format(
            sitename)
        html_page = urllib2.urlopen(self.roi_url)
        soup = _BS(html_page, "lxml")
        for link in soup.findAll('a'):
            href = link.get('href')
            if href.endswith('.csv'):
                self.csvs.append(href)
            elif href.endswith('.png'):
                self.pngs.append(href)
            elif href.endswith('.tif'):
                self.tifs.append(href)
            else:
                self.others.append(href)

        self.rois = {}
        for tif in self.tifs:
            try:
                sitename_, roitype, roisequence, maskindex = tif.split('_')
                self.rois[roitype] = self.rois.get(roitype, {})
                self.rois[roitype]['mask'] = self.rois[roitype].get('mask', {})
                self.rois[roitype]['mask'][roisequence] = \
                    self.rois[roitype]['mask'].get(roisequence, {})
                self.rois[roitype]['mask'][roisequence][maskindex[:-4]] = \
                    self.rois[roitype]['mask'][roisequence].get(maskindex[:-4], {})
                self.rois[roitype]['mask'][roisequence][maskindex[:-4]] = tif
            except ValueError:
                pass #old style name schema, ignoring

        for csv in self.csvs:
            if csv.startswith(sitename):
                parts = csv.split('_')
                if len(parts) < 4:
                    pass
                elif parts[3] == 'timeseries':
                    sitename_, roitype, roisequence, which, v = parts
                elif parts[3] == 'gcc90':
                    sitename_, roitype, roisequence, which, length, v = parts
                elif len(parts) == 5:
                    sitename_, roitype, roisequence, length, v = parts
                    which = 'gcc90'

                if len(parts) == 4:
                    sitename_, roitype, roisequence, roi = parts
                    self.rois[roitype]['roicsv'] = self.rois[
                        parts[1]].get('roicsv', {})
                    self.rois[roitype]['roicsv'][roisequence] = csv
                else:
                    self.rois[roitype][which] = self.rois[
                        roitype].get(which, {})
                    self.rois[roitype][which][length] = self.rois[
                        roitype][which].get(length, {})
                    self.rois[roitype][which][length][
                        v[:-4]] = self.rois[roitype][which][length].get(v[:-4], {})
                    self.rois[roitype][which][length][v[:-4]] = csv
        try:
            one_day_fname = [c for c in self.csvs if "_1day_" in c][0]
            self.one_day = pd.read_csv(
                self.roi_url + '/' + one_day_fname, comment="#", parse_dates=['date'])
            self.one_day.index = self.one_day.date
            self.midday_fnames = self.one_day.midday_filename.tolist()
            self.midday_fnames = [
                value for value in self.midday_fnames if not str(value) == 'nan']
            timeseries_fname = [c for c in self.csvs if "_timeseries_" in c][-1]
            timeseries = pd.read_csv(
                self.roi_url + '/' + timeseries_fname, comment="#", parse_dates=['date'])
            timeseries.index = timeseries.date
            self.all_fnames = timeseries.filename
            self.all_fnames = [
                value for value in self.all_fnames if not str(value) == 'nan']
        except:
            pass

        url = "http://phenocam.sr.unh.edu/webcam/browse/{}/".format(self.sitename)
        html_page = urllib2.urlopen(url)
        soup = _BS(html_page, "lxml")

        years = {}
        for y in soup.find_all('div', {"class":"yearsummary"}):
            year = int(y.text.split()[1])
            years[year] = {}
            for table in y.find_all('table'):
                for a in table.find_all('a'):
                    month = int(a.get('href').split('/')[-2])
                    years[year][month] = {}
        self.data = years

        allsites = SITES_DF  # get_sites_df()
        self.y, self.x = allsites[allsites.site == sitename].values[0][1:3]

    def get_days(self, dt):
        fnameurl = config.get_url('FNAMEURL_BROWSE')
        url = fnameurl.format(self.sitename, dt.year, dt.month, dt.day)[:-3]
    
        html_page = urllib2.urlopen(url)
        soup = _BS(html_page, "lxml")
        ir_lookup = {}
        days = {}
        for link in soup.findAll("div", {"class":"calday"}):
            monthday = link.findAll("div", {"class":"monthday"})
            if monthday:
                day = int(monthday[0].findAll('strong')[0].text)
                images = int(link.findAll("div", {"class":"imagecount"})[0].text.split('=')[1])
                days[day] = images
        return days

    def get_closest_fname(self, dt):
        fnameurl = config.get_url('FNAMEURL_BROWSE')

        if not self.data.has_key(dt.year):
            raise Exception, "data for year {} not available".format(dt.year)

        if not self.data[dt.year].has_key(dt.month):
            raise Exception, "data for year/month {}/{} not available".format(dt.year, dt.month)

        if not self.data[dt.year][dt.month]:
            self.data[dt.year][dt.month] = self.get_days(dt)

        day_search_order = np.vstack((np.arange(30), np.arange(30)*-1)).reshape((-1,),order='F')[1:]
        for offset in day_search_order:
            d = dt.day + offset
            if self.data[dt.year][dt.month].has_key(d) and self.data[dt.year][dt.month][d] > 0:
                break

        url = fnameurl.format(self.sitename, dt.year, dt.month, dt.day+offset)

        html_page = urllib2.urlopen(url)
        soup = _BS(html_page, "lxml")
        ir_lookup = {}
        for link in soup.findAll('a'):
            href = link.get('href')
            if href and href.endswith('jpg'):
                ir_fname = href.split('/')[-1]
                ir_dt = utils.parse_fname(ir_fname)[-1]
                ir_lookup[ir_dt] = ir_fname


        return ir_lookup[utils.nearest_date(ir_lookup.keys(), dt)]

            


    def list_rois(self,):
        """Returns a list of the rois associated with this site"""
        return self.rois.keys()

    def get_roi_fname(self, roi_type=None, roi_sequence=None, roi_num=None):
        """Returns a local filename to the roi request
        Downloads a local copy if one doesn't exist"""

        if not roi_type:
            roi_type = self.rois.keys()[0]
        if not roi_sequence:
            roi_sequence = self.rois[roi_type]['mask'].keys()[0]
        if not roi_num:
            roi_num = self.rois[roi_type]['mask'][roi_sequence].keys()[0]

        roi_fname = self.rois[roi_type]['mask'][roi_sequence][roi_num]
        local_fname = _os.path.join(self.site_dname, self.sitename, roi_fname)
#         print local_fname

        if not _os.path.exists(local_fname):
            roi_tif_url = self.roi_url + "/" + roi_fname
            urllib.urlretrieve(roi_tif_url, local_fname)

        return local_fname

    def get_roi(self, roi_type=None, roi_sequence=None, roi_num=None, masked=False):
        """Returns a boolean numpy array for a specified ROI """
        fname = self.get_roi_fname(roi_type, roi_sequence, roi_num)
        roi = imageprocessing.get_boolean_photo_array(fname)
        if not roi[0, 0]:
            roi = np.logical_not(roi)
        if masked:
            roi = np.ma.masked_where(roi == 0, roi)
        return roi

    def convert_fname_to_url(self, fname):
        """returns the full url to a specified fname 
        """
        site, year, month, day, time = fname.split('_')
        url = "http://phenocam.sr.unh.edu/data/archive/{}/{}/{}/{}".format(
            site, year, month, fname)
        return url

    def convert_fname_to_cachefname(self, fname):
        site, year, month, day, time = fname.split('_')
        cache_fname = _os.path.join(self.site_dname, site, year, month, fname)
        return cache_fname

    def get_local_image_fname(self, fname, IR=False):
        """if it hasn't been previously downloaded a local copy of the file
        with a name of fname will be downloaded 

        if IR is True it will also download the cooresponding IR image
        """

        url = self.convert_fname_to_url(fname)
        local_fname = self.convert_fname_to_cachefname(fname)
        local_dname = _os.path.split(local_fname)[0]

        if not _os.path.exists(local_dname):
            _os.makedirs(local_dname)

        if not _os.path.exists(local_fname):
            urllib.urlretrieve(url, local_fname)

        if IR:
            ir_fname = utils.convert_fname_to_ir(fname)
            local_ir_fname = utils.convert_fname_to_ir(local_fname)
            if not _os.path.exists(local_ir_fname):
                ir_url = url.replace(fname, ir_fname)
                urllib.urlretrieve(ir_url, local_ir_fname)

        if IR:
            return local_ir_fname
        else:
            return local_fname

    def get_local_image(self, fname, IR=False):
        """if it hasn't been previously downloaded a local copy of the file
        with a name of fname will be downloaded 

        if IR is True it will also download the cooresponding IR image
        """
        if IR:
            local_ir_fname = self.get_local_image_fname(fname, IR)
            return imageprocessing.get_photo_array(local_ir_fname)
        else:
            local_fname = self.get_local_image_fname(fname, IR)
            return imageprocessing.get_photo_array(local_fname)

    def get_midday_image(self, which):

        if type(which) == int:
            midday_fname = self.midday_fnames[which]
        else:
            midday_fname = utils.fcl(self.one_day, which).midday_filename
        return self.get_local_image(midday_fname)

    def get_data(self, roi_type=None, which='gcc90', length='1day', version='v4'):
        if not roi_type:
            roi_type = self.rois.keys()[0]

        df = pd.read_csv(
            self.roi_url + '/' + self.rois[roi_type][which][length][version], comment="#", parse_dates=['date'])
        df.index = df.date

        return df

