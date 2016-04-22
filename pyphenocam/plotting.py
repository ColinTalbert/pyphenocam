# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as _mpl
import pandas as pd

try:
    import cartopy
    import cartopy.crs as ccrs
    from cartopy.io.img_tiles import MapQuestOpenAerial
    import cartopy.feature as cfeature
    import cartopy.io
    from cartopy.mpl.geoaxes import GeoAxes
except ImportError:
    pass

import matplotlib.pyplot as plt



import utils
import imageprocessing


ROI_TYPES = {'AG': {'name':'Agriculture',
                    'linecolor':'#ffb752'},
             'DB': {'name':'Deciduous Broadleaf',
                    'linecolor':'#00b050'},
             'EB': {'name':'Evergreen Broadleaf',
                    'linecolor':'#0f6001'},
             'DN': {'name':'Deciduous Needleaf',
                    'linecolor':'#FFB752'},
             'EN': {'name':'Evergreen Needleleaf',
                    'linecolor':'#003000'},
             'GR': {'name':'Grassland',
                    'linecolor':'#FFB752'},
             'NV': {'name':'Non-vegetated',
                    'linecolor':'#FFB752'},
             'RF': {'name':'Reference Panel',
                    'linecolor':'#FFB752'},
             'SH': {'name':'Shrub',
                    'linecolor':'#da7932'},
             'UN': {'name':'Understory',
                    'linecolor':'#FFB752'},
             'WL': {'name':'Wetland',
                    'linecolor':'#FFB752'},
             'XX': {'name':'Mixed/Canopy/Other',
                    'linecolor':'#FFB752'},}

def format_photo_axes(ax):
    """Given an matplotlib axes
    turns off the ticks and tick labels
    and sets the aspect to equal
    """
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def add_rois(ax, site, vistype='image', **kwargs):
    """Adds the rois for a given site to the passed axes.
    generally this ax will be displaying a phenocam image from the same site
    vistype == 'image' will display the roi as a translucent white mask
    vistype == 'line' will display the roi or rois as colored outlines
    **kwargs allows passing of ma
    """
    for roi in site.list_rois():
        roi_image = site.get_roi(roi, masked=True)
        if vistype=='image':
            ax.imshow(roi_image, cmap=_mpl.cm.Greys, alpha=0.4, vmax=1)
        else:
            if not kwargs.has_key('lw'):
                kwargs['lw'] = 2
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            contour_labels = []   
            for contour in imageprocessing.get_roi_contours(roi_image):
                if roi in contour_labels:
                    ax.plot(contour[:, 1], contour[:, 0], 
                    color=ROI_TYPES[roi]['linecolor'], **kwargs)
                else:
                    ax.plot(contour[:, 1], contour[:, 0], 
                    color=ROI_TYPES[roi]['linecolor'], 
                    label=ROI_TYPES[roi]['name'], **kwargs)
                contour_labels.append(roi)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
    

def photo_format(thing=None):
    """Given a matplotlib figure, axes, or list of axes
    formats each axes using the 
    format_photo_axes function
    if nothing is given the current figure will be formated
    """
    if not thing:
        thing = plt.gcf()
    try:
        for ax in thing:
            format_photo_axes(ax)
    except TypeError:
        for ax in thing.axes:
            format_photo_axes(ax)
    except:
        format_photo_axes(thing)


def plot_gcc_and_daymet(site, ax, rois=None, start_year=None, end_year=None, 
                        fontsize=10, lw=4):
    """given a site makes a chart of the GCC, and temperature and precipitation
    temp and precip only work for sites that have Daymet coverage (continental US only)
    """
    import daymetpy
    
    if not rois:
        rois = []
        for roi in site.rois.keys():
            rois.append((roi, ROI_TYPES[roi]['linecolor']))
        
    if not start_year:
        start_year = quickbird.get_data().date.min().year
    if not end_year:
        end_year = quickbird.get_data().date.max().year
    
    ax.set_ylabel('Canopy GCC', fontsize=fontsize)

    ax1 = ax.twinx()
    ax2 = ax.twinx()
    ax2.spines['right'].set_position(('axes', 1.1))

    tcolor, pcolor = 'indianred', 'steelblue'
    ax1.set_ylabel(u'Temp. (°C)', fontsize=fontsize, color=tcolor)
    ax1.tick_params(axis='y', colors=tcolor)
    ax2.set_ylabel(u'Precip. (mm)', fontsize=fontsize, color=pcolor)
    ax2.tick_params(axis='y', colors=pcolor)
    
    try:
        df_daymet = daymetpy.download_Daymet(lon=site.x, lat=site.y, start_yr=start_year, end_yr=end_year+1)
        rolling3day = df_daymet.rolling(window=30, center=False).mean()
        ax1.fill_between(rolling3day.index, rolling3day.tmin, rolling3day.tmax, alpha=0.3, lw=0, color=tcolor, label='Temperature')

        monthlysum = df_daymet.resample("M").sum()
        ax2.bar(monthlysum.index, monthlysum.prcp, width=20, alpha=0.3, color=pcolor, label='Precipitation')
    except Exception as e:
        print e
        pass
    
    for roi, color in rois:
        df = site.get_data(roi_type=roi, length='3day')
        gcc = df.ix['{}-01-01'.format(start_year):'{}-12-31'.format(end_year)].gcc_90
        ax.plot(gcc.index, gcc.values, alpha=0.7, lw=lw, color=color, zorder=100, label=ROI_TYPES[roi]['name'] )
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax1.tick_params(axis='both', which='major', labelsize=fontsize)
        ax2.tick_params(axis='both', which='major', labelsize=fontsize)

    ax.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2 
    ax.patch.set_visible(False) # hide the 'canvas'
    
def add_inner_title(ax, title, loc, size=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    if size is None:
        size = dict(size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=size,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at


class LocatorMap(GeoAxes):

    def __init__(self, networksites=[], *args, **kwargs):

        if not kwargs.has_key('fig'):
            kwargs['fig'] = plt.gcf()

        self.tiler = MapQuestOpenAerial()
        kwargs['map_projection'] = self.tiler.crs
        if not kwargs.has_key('rect'):
            kwargs['rect'] = [0, 0, 1, 1]
        super(LocatorMap, self).__init__(*args, **kwargs)
        self.add_image(self.tiler, 3)

        self.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())
        self.set_title('US Phenocam Network')
        self.coastlines()

        states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')

        self.add_feature(states_provinces, edgecolor='gray')
        self.add_feature(cartopy.feature.BORDERS, lw=2)

        self.networksites = networksites

        self.load_sites()

        self.cursite = None
        self.cur_annotation = None
        self.hov_annotation = None

    def load_sites(self):
        url = "http://phenocam.sr.unh.edu/webcam/roi/roilistinfo/?format=csv"

        self.sites = pd.read_csv(url)
        self.site_names = list(self.sites.site.unique())

        self.scatter(self.sites.lon.tolist(),
                     self.sites.lat.tolist(),
                     transform=ccrs.Geodetic(), color='r', edgecolor='black')

        for site in self.networksites:
            site_record = self.sites[self.sites.site == site]
            self.scatter(site_record.lon.tolist(),
                         site_record.lat.tolist(),
                         transform=ccrs.Geodetic(), marker="*",
                         s=200, color='green', edgecolor='black')

    def mouse_move(self, event):
        closest_site, nearestx, nearesty = self.get_closes_site(event)

        try:
            self.hov_annotation.remove()
        except:
            pass

        if not closest_site is None:
            self.hov_annotation = self.annotate(closest_site.site.tolist()[0],
                                                xy=(nearestx, nearesty),
                                                xytext=(
                                                    nearestx + 10000, nearesty - 300000),
                                                bbox=dict(
                                                    boxstyle="round", fc="0.8"),
                                                arrowprops=dict(arrowstyle="wedge,tail_width=0.7",
                                                                fc="0.6", ec="none",
                                                                connectionstyle="arc3,rad=-0.3"), alpha=0.5)

    def mouse_click(self, event):
        closest_site, nearestx, nearesty = self.get_closes_site(
            event, thresh=10)

        try:
            self.hov_annotation.remove()
        except:
            pass
        try:
            self.cur_annotation.remove()
        except:
            pass

        if not closest_site is None:
            self.cur_annotation = self.annotate(closest_site.site.tolist()[0],
                                                xy=(nearestx, nearesty),
                                                xytext=(
                                                    nearestx + 10000, nearesty - 300000),
                                                bbox=dict(
                                                    boxstyle="round", fc="0.8", ec="r"),
                                                arrowprops=dict(arrowstyle="wedge,tail_width=0.7",
                                                                fc="0.6", ec="r",
                                                                connectionstyle="arc3,rad=-0.3"))
            return closest_site.site.tolist()[0]

    def get_closes_site(self, event, thresh=0.35):

        clickx, clicky = ccrs.Geodetic().transform_point(event.xdata,
                                                         event.ydata, self.tiler.crs)

        self.sites['dist_to_click'] = self.sites.apply(
            lambda row: utils.dist(row['lon'], row['lat'], clickx, clicky), axis=1)

        closest_site = self.sites[
            self.sites.dist_to_click == self.sites.dist_to_click.min()]

        nearestx, nearesty = self.tiler.crs.transform_point(closest_site.lon.tolist()[0],
                                                            closest_site.lat.tolist()[0], ccrs.Geodetic())

        if float(closest_site.dist_to_click.tolist()[0]) > thresh:
            closest_site = None

        return closest_site, nearestx, nearesty
