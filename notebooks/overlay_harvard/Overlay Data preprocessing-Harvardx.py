
# coding: utf-8

# ## This notebook contains code to easily create the inputs required to run the process in one of the 'overlay' notebooks.

# #### In addition to the outputs produced by this output, the user will need to also obtain from other sources elevation (NED DEM) and optionally ortho imagery data (NAIP).  

# #### The final steps in the process is done in manually ArcScene outside of this notebook.  See instructions at the bottom of this notebook.

# In[1]:

import os

import matplotlib.pyplot as plt
import matplotlib as mpl

import rasterio


# In[2]:

import sys
sys.path.append(r"..")
import phenocamomatic

sys.path.append(r"J:\Projects\NCCSC\phenocam\Tools\DaymetPy\daymetpy")
import daymetpy


# # Create a point shapefile of our camera location

# In[3]:

output_dir = r"J:\Projects\NCCSC\phenocam\DerivedData"
site_name = "harvard"

output_dir = os.path.join(output_dir, site_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# In[4]:

site = phenocamomatic.dataaccess.get_site(site_name)


# In[5]:

site.x, site.y


# In[6]:

import fiona
from fiona.crs import from_epsg
from shapely.geometry import Point, mapping


# In[7]:

simpleschema = {'geometry': 'Point',
               'properties': {'name':'str'}}
shape_fname = os.path.join(output_dir, "ArcScene", "InputData", "wgs84","cameraloc.shp")
if not os.path.exists(os.path.split(shape_fname)[0]):
    os.makedirs(os.path.split(shape_fname)[0])


with fiona.open(shape_fname, 'w', crs=from_epsg(4326),driver='ESRI Shapefile', schema=simpleschema) as output:
    point = Point(site.x, site.y)
    output.write({'properties': {'name': site.sitename},'geometry': mapping(point)})


# # Download the landsat8 scene over this area

# In[8]:

from landsat.search import Search
from landsat.downloader import Downloader


# In[9]:

s = Search()
results = s.search(lat=site.y, lon=site.x, limit=100)
scene_id = results['results'][0]['sceneID']


# In[10]:

landsat_dname = os.path.join(output_dir, 'Landsat')
if not os.path.exists(landsat_dname):
    os.makedirs(landsat_dname)


# In[29]:

from landsat.downloader import Downloader
outdir = os.path.join(output_dir, "Landsat")
d = Downloader(download_dir=outdir)
d.download([str(scene_id)], bands=[5, 4])


# In[11]:

scene_dname = os.path.join(landsat_dname, scene_id)
scene_dname
b4_fname = os.path.join(scene_dname, [f for f in os.listdir(scene_dname) if f.endswith('_B4.TIF')][0])
b5_fname = os.path.join(scene_dname, [f for f in os.listdir(scene_dname) if f.endswith('_B5.TIF')][0])


# In[12]:

print scene_dname


# In[13]:

landsat = rasterio.open(b5_fname)
landsat_data = landsat.read(masked=True)

img_extents = [landsat.bounds.left, landsat.bounds.right, landsat.bounds.bottom, landsat.bounds.top]


# #### If the location isn't good (i.e. the red dot doesn't have landsat data around it) either get another landsat scene or mosaic a couple

# In[14]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

landsat_proj = ccrs.UTM(zone=int(landsat.crs_wkt.split("UTM zone ")[1].split('N')[0]), 
                        globe=ccrs.Globe(datum='WGS84', ellipse='WGS84'))
fig = plt.figure(figsize=(15, 15))
# ax_extent = [quickbird.x - 0.015, quickbird.x + 0.015, quickbird.y - 0.002, quickbird.y + 0.015]

img_extents = [landsat.bounds.left, landsat.bounds.right, landsat.bounds.bottom, landsat.bounds.top]

ax = plt.axes(projection=landsat_proj)
# ax.set_extent(ax_extent, ccrs.Geodetic())
ax.imshow(landsat_data[0, ::10, ::10], origin='upper', extent=img_extents, transform=landsat_proj, interpolation='none', 
          cmap=mpl.cm.jet)
ax.coastlines(resolution='10m')

geodetic = ccrs.Geodetic()
ax.scatter(site.x, site.y, color='r', s=200, alpha=0.5, transform=geodetic)


# ### We need to extract out a subset 254x254 centered horizontally on our camera loc, and extending north

# In[15]:

camx, camy = list(landsat_proj.transform_point(site.x, site.y, geodetic))
camx, camy


# In[16]:

a = landsat.affine


# In[17]:

import math
cam_col, cam_row = [int(math.floor(coord)) for coord in ~a * (camx, camy)]
ulcol, lrcol = cam_col-254/2, cam_col+254/2
ulrow, lrrow = cam_row-250, cam_row+4

print cam_col, cam_row
print ulcol, ulrow
print lrcol, lrrow


# In[18]:

ulx,uly = a * (ulcol, ulrow)
lrx,lry = a * (lrcol, lrrow)

print ulx, uly
print lrx, lry

print camx, camy


# In[19]:

get_ipython().magic(u'matplotlib inline')
fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection=landsat_proj)
ax.imshow(landsat_data[0, ::10, ::10], origin='upper', extent=img_extents, transform=landsat_proj, interpolation='none', 
          cmap=mpl.cm.jet)

ax.scatter(site.x, site.y, color='r', s=20, alpha=0.5, transform=geodetic)

ax.plot([ulx,lrx], [uly, uly], 'k-', lw=2, c='black', transform=landsat_proj)
ax.plot([ulx,lrx], [lry, lry], 'k-', lw=2, c='black', transform=landsat_proj)
ax.plot([ulx,ulx], [uly, lry], 'k-', lw=2, c='black', transform=landsat_proj)
ax.plot([lrx,lrx], [uly, lry], 'k-', lw=2, c='black', transform=landsat_proj)


# In[20]:

from shapely.geometry import mapping, Polygon
import fiona

# Here's an example Shapely geometry
poly = Polygon([(ulx, uly), (lrx, uly), (lrx, lry), (ulx, lry), (ulx, uly)])

# Define a polygon feature geometry with one attribute
schema = {
    'geometry': 'Polygon',
    'properties': {'id': 'int'},
}

# Write a new Shapefile
landsat_subset_dname = os.path.join(landsat_dname, 'Subset')
if not os.path.exists(landsat_subset_dname):
    os.makedirs(landsat_subset_dname)
    
boundary_fname = os.path.join(landsat_subset_dname, 'boundary.shp')
with fiona.open(boundary_fname, 'w', 'ESRI Shapefile', schema, crs=landsat.crs) as c:
    ## If there are multiple geometries, put the "for" loop here
    c.write({
        'geometry': mapping(poly),
        'properties': {'id': 123},
    })


# In[21]:

from shapely.geometry import mapping, Polygon
import fiona

# Here's an example Shapely geometry
poly = Polygon([(ulx, uly), (lrx, uly), (lrx, lry), (ulx, lry), (ulx, uly)])

# Define a polygon feature geometry with one attribute
schema = {
    'geometry': 'Polygon',
    'properties': {'row': 'int', 'col':'int'},
}

# Write a new Shapefile
landsat_subset_dname = os.path.join(landsat_dname, 'Subset')
if not os.path.exists(landsat_subset_dname):
    os.makedirs(landsat_subset_dname)
    
fishnet_fname = os.path.join(landsat_subset_dname, 'fishnet.shp')
with fiona.open(fishnet_fname, 'w', 'ESRI Shapefile', schema, crs=landsat.crs) as c:
    ## If there are multiple geometries, put the "for" loop here
    for row in range(254):
        cell_uly = uly+(row*a.e)
        for col in range(254):
            cell_ulx = ulx+(col*landsat.affine.a)
            poly = Polygon([(cell_ulx, cell_uly), 
                            (cell_ulx + a.a, cell_uly), 
                            (cell_ulx + a.a, cell_uly+a.e), 
                            (cell_ulx, cell_uly+a.e), 
                            (cell_ulx, cell_uly)])
    
    
            c.write({
                'geometry': mapping(poly),
                'properties': {'row': row,
                              'col':col},
            })


# # Create a subset of this for display purposes

# In[55]:

subset = landsat_data[0, ulrow:lrrow, ulcol:lrcol]

plt.imshow(subset)
plt.scatter(cam_col-ulcol, cam_row-ulrow, c='r')
phenocamomatic.plotting.format_photo_axes(plt.gca())


# In[56]:

subset_fname = os.path.join(landsat_subset_dname, "subset2.tif")

import copy
subset_meta = copy.copy(landsat.meta)
subset_meta['width'] = subset.shape[0]
subset_meta['height'] = subset.shape[1]

new_transform = list(landsat.meta['transform'])
new_transform[0] = ulx
new_transform[3] = uly
subset_meta['transform'] = tuple(new_transform)
import affine
subset_meta['affine'] = affine.Affine.from_gdal(*subset_meta['transform'])
with rasterio.open(subset_fname, 'w', **subset_meta) as dst:
    dst.write_band(1, subset.astype(rasterio.uint16))


# In[57]:

get_ipython().magic(u'matplotlib inline')
n=10
r = np.zeros(subset.shape).astype(np.dtype('i1'))
col_x3 = np.tile(np.repeat(np.arange(r.shape[1]), n), (r.shape[0])*n)
col_x3.shape = r.shape[0]*n, r.shape[1]*n

col_x3test = col_x3.copy()
col_x3test[col_x3test % 2 != 0] = 255 - col_x3test[col_x3test % 2 != 0]

plt.imshow(col_x3, interpolation='none')
plt.colorbar()


# In[58]:

row_x3 = np.tile(np.repeat(np.arange(r.shape[0]), n), (r.shape[1]*n))
row_x3.shape = r.shape[1]*n, r.shape[0]*n
row_x3 = row_x3.T

row_x3test = row_x3.copy()
row_x3test[row_x3test % 2 != 0] = 255 - row_x3test[row_x3test % 2 != 0]

plt.imshow(row_x3, interpolation='none')
plt.colorbar()


# In[59]:

b = np.zeros(row_x3.shape)


# In[60]:

get_ipython().magic(u'matplotlib inline')
plt.imshow(np.dstack([col_x3, row_x3, b])[:,:,:], interpolation='none')


# ##### One small detail to note, because of the way ArcScene renders a raster, we're increasing the resoultion 10x

# In[61]:

import scipy
subset_index_fname = os.path.join(landsat_subset_dname, "subset_index.tif")

import copy
subset_meta = copy.copy(landsat.meta)
subset_meta['width'] = subset.shape[0] * 10
subset_meta['height'] = subset.shape[1] * 10

new_transform = list(landsat.meta['transform'])
new_transform[0] = ulx
new_transform[3] = uly
new_transform[1] = new_transform[1] / 10.0
new_transform[-1] = new_transform[-1] / 10.0
subset_meta['transform'] = new_transform

import affine
subset_meta['affine'] = affine.Affine.from_gdal(*subset_meta['transform'])

subset_meta.update(
    dtype=rasterio.uint8,
    count=3,
    nodata=255)

with rasterio.open(subset_index_fname, 'w', **subset_meta) as dst:
    dst.write_band(1, col_x3.astype(rasterio.uint8))
    dst.write_band(2, row_x3.astype(rasterio.uint8))
    dst.write_band(3, b.astype(rasterio.uint8))


# ## Now download and process some NED 30m DEM data for our area

# In[1]:

from ulmo.usgs.ned import core
core.get_available_layers()


# In[3]:

import os
import ulmo
print(ulmo.util.get_ulmo_dir())

layer = '1/3 arc-second'
bbox = [-100, 35, -100.05, 35.05]
dname = r"C:\temp\test"
if not os.path.exists(dname):
    os.makedirs(dname)
    
result = core.get_raster(layer=layer, bbox=bbox, path=dname, mosaic=True)


# In[44]:

layer = '1/3 arc-second'
bbox = list(geodetic.transform_point(ulx, lry, landsat_proj))
bbox += list(geodetic.transform_point(lrx, uly, landsat_proj))
dname = os.path.join(output_dir, "ArcScene", "InputData", "DEM")
if not os.path.exists(dname):
    os.makedirs(dname)
    
result = core.get_raster(layer=layer, bbox=bbox, path=dname, mosaic=True)


# ### Reproject this to match our landsat subset grid

# In[1]:

dem = rasterio.open(result)
subset = rasterio.open(subset_fname)


# In[64]:

from rasterio.warp import calculate_default_transform, reproject, RESAMPLING

out_fname = os.path.join(dname, os.path.split(result[0])[1].replace('.tif', '_utm.tif'))

# affine, width, height = calculate_default_transform(
#     src.crs, dst_crs, src.width, src.height, *src.bounds)
kwargs = subset.meta.copy()
kwargs['dtype'] = dem.meta['dtype']
kwargs['nodata'] = dem.meta['nodata']
# kwargs.update({
#         'crs': subset.crs,
#         'transform': subset.affine,
#         'affine': subset.affine,
#         'width': subset.width,
#         'height': subset.height
#     })

with rasterio.open(out_fname, 'w', **kwargs) as dst:
    reproject(
        source=rasterio.band(dem, 1),
        destination=rasterio.band(dst, 1),
        src_transform=dem.affine,
        src_crs=dem.crs,
        dst_transform=subset.affine,
        dst_crs=subset.crs,
        resampling=RESAMPLING.cubic_spline)

    

    dem_data = dst.read_band(1, masked=True)   


# In[66]:

elev_subset_fname = out_fname
elev = rasterio.open(elev_subset_fname)
elev_data = elev.read()

nad83 = ccrs.Geodetic(globe=ccrs.Globe(datum='NAD83', ellipse='GRS80'))
# %matplotlib inline
# plt.imshow(landsat_data[0,:,:], interpolation='none')



elev_extents = [elev.bounds.left, elev.bounds.right, elev.bounds.bottom, elev.bounds.top]


fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection=landsat_proj)
ax.imshow(dem_data, origin='upper', extent=elev_extents, transform=landsat_proj, 
          cmap=mpl.cm.gist_earth, interpolation='none')

ax.scatter(site.x, site.y, color='r', s=20, alpha=0.5, transform=geodetic)

ax.plot([ulx,lrx], [uly, uly], 'k-', lw=2, c='black', transform=landsat_proj)
ax.plot([ulx,lrx], [lry, lry], 'k-', lw=2, c='black', transform=landsat_proj)
ax.plot([ulx,ulx], [uly, lry], 'k-', lw=2, c='black', transform=landsat_proj)
ax.plot([lrx,lrx], [uly, lry], 'k-', lw=2, c='black', transform=landsat_proj)


# Now open arcscene and bring in the subset_index.tif, fishnet.shp, and cameraloc.shp files
# Then create an elevation grid (30m dem) that matches the landsat grid, and maybe an ortho photo of the area as well (NAIP)
# 
# use the elevation grid to set the base heights of all
# extrude the camera loc point to create a visual pole
# 
# Navigate to the camera location and orient the scene manually to recreate the phenocam photo view.  This can be a bit fiddly.
# Change the display of the fishnet.shp to have white squares and red outlines. turn off shadows.
# Export this scene to 2d. (..\ArcScene\red_gridlines.bmp)
# Change the display of the subset_index.tif to have a raster resolution of 3 in the base heights screen and stretch = None and turn off the apply gamma stretch check box in symbology.
# Export this scene to 2d. (..\ArcScene\IndexGrid_final.bmp)
# 

# ### Get a sample of modis data to cover our area

# In[68]:

import pymodis


# In[35]:

pymodis.downmodis


# In[69]:

pymodis.


# In[70]:

import pandas as pd


# In[85]:

pd.read_table("http://modis-land.gsfc.nasa.gov/pdf/sn_bound_10deg.txt", skiprows=17, skipfooter=13)


# In[89]:


url="http://landweb.nascom.nasa.gov/developers/sn_tiles/sn_gring_10deg.txt"
s=requests.get(url).content
c=pd.read_csv('\n'.join(io.StringIO(s.decode('utf-8')).read().splitlines()))


# In[3]:

import pandas as pd
import io
import requests

import shapely

import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
from pandas import DataFrame

url="http://landweb.nascom.nasa.gov/developers/sn_tiles/sn_gring_10deg.txt"
s=requests.get(url).content
contents = '\n'.join(io.StringIO(s.decode('utf-8')).read().splitlines()[7:-2])
hv_lookup = pd.read_csv(StringIO(contents), sep=r"\s+", parse_dates=False, 
                        na_values=[-99.0, -999.0], 
                        names = ['v', 'h', 'll_lon', 'll_lat', 'ul_lon', 'ul_lat', 'ur_lon', 'ur_lat', 'lr_lon', 'lr_lat'])


# In[4]:

hv_lookup


# In[6]:

import cartopy
modis = cartopy.crs.Sinusoidal.MODIS
geodetic = cartopy.crs.Geodetic()

lon = -109.0
lat = 35.0

def get_modis(lon, lat):
    return list(modis.transform_point(lon, lat, geodetic))


# In[40]:

get_ipython().magic(u'matplotlib qt4')
from matplotlib import pyplot
from descartes.patch import PolygonPatch

fig = pyplot.figure(1, figsize=(12,12), dpi=90)
ax = fig.add_subplot(111)

ax_ll_lon, _ = get_modis(-180, 0)
ax_ur_lon, _ = get_modis(180, 0)
_, ax_ll_lat = get_modis(0, -90)
_, ax_ur_lat = get_modis(0, 90)

tile_width = (ax_ur_lon - ax_ll_lon)/36
tile_height = (ax_ur_lat - ax_ll_lat)/18

ax.set_xlim((ax_ll_lon, ax_ur_lon))
ax.set_ylim((ax_ll_lat, ax_ur_lat))

for h in range(36):
    left_lon = ax_ll_lon + (tile_width*h)
    right_lon = left_lon + tile_width
    for v in range(18):
        top_lat = ax_ur_lat - (tile_height*v)
        bottom_lat = top_lat - tile_height
        
        poly = shapely.geometry.Polygon([(left_lon, top_lat),
                             (right_lon, top_lat),
                             (right_lon, bottom_lat),
                             (left_lon, bottom_lat),
                             (left_lon, top_lat),])
        patch = PolygonPatch(poly, facecolor='red', edgecolor='w', alpha=0.5, zorder=2)
        ax.add_patch(patch)
        
        ax.text((right_lon+left_lon)/2, (top_lat+bottom_lat)/2, '{} {}'.format(h, v))


# for row_i in range(hv_lookup.shape[0]-100):
#     try:
#         row = hv_lookup.loc[row_i]
#         if pd.notnull(hv_lookup.loc[row_i].ll_lon):

#             ll_lon, ll_lat = get_modis(row['ll_lon'], row['ll_lat'])
#             lr_lon, lr_lat = get_modis(row['lr_lon'], row['lr_lat'])
#             ul_lon, ul_lat = get_modis(row['ul_lon'], row['ul_lat'])
#             ur_lon, ur_lat = get_modis(row['ur_lon'], row['ur_lat'])


#     #         print ll_lon, ll_lat, ur_lat, ur_lon
#     #         if ll_lon > -170 and lr_lon < 170 and \
#     #             ul_lon > -170 and ur_lon < 170:


#             poly = shapely.geometry.Polygon([(ll_lon, ll_lat),
#                              (lr_lon, lr_lat),
#                              (ur_lon, ur_lat),
#                              (ul_lon, ul_lat),
#                              (ll_lon, ll_lat),])
#             patch = PolygonPatch(poly, facecolor='red', edgecolor='w', alpha=0.1, zorder=2)
#             ax.add_patch(patch)
            
        
# #         print "g"
#     except ValueError:
#         print ".",
#         pass


# In[36]:

(ax_ur_lon - ax_ll_lon)/36


# In[11]:

row['ll_lon']


# In[198]:

get_ipython().magic(u'matplotlib inline')
row = hv_lookup.loc[626]
poly = shapely.geometry.Polygon([(row['ll_lon'], row['ll_lat']),
                         (row['lr_lon'], row['lr_lat']),
                         (row['ur_lon'], row['ur_lat']),
                         (row['ul_lon'], row['ul_lat']),
                         (row['ll_lon'], row['ll_lat']),])

fig = pyplot.figure(1, figsize=(12,12), dpi=90)
ax = fig.add_subplot(111)

patch = PolygonPatch(poly, facecolor='red', edgecolor='w', alpha=0.5, zorder=2)
ax.add_patch(patch)
ax.set_xlim((-180.0, 180.0))
ax.set_ylim((-90.0, 90.0))


# In[174]:

hv_lookup


# In[ ]:

def get_hv(lat, lon):
    """returns the horizontal and vertical index of a modis tile
    cooresponding to a passed latitude and longitude
    """
    


# In[129]:

lon = -109.0
lat = 35.0


# In[134]:

lon_rows = hv_lookup[(hv_lookup.lon_min<=lon) & (hv_lookup.lon_max>=lon) & (hv_lookup.lat_min<=lat) & (hv_lookup.lat_max>=lat)]
lon_rows


# In[21]:

pd.notnull(hv_lookup.loc[14].ll_lon)


# In[31]:

get_modis(-180, 0)


# In[42]:

import pygaarst


# In[43]:

from pygaarst import modapsclient as m
a = m.ModapsClient()


# In[45]:

a.getDataLayers('MOD13Q1')


# In[1]:

import pymodis


# In[6]:

# destination foldert
dest = r"C:\temp_colin\downloads\tilemap3_r4_0"
# tiles to download
tiles = "h18v04,h19v04"
# starting day
day = "2014-08-14"
# number of day to download
delta = 16


# In[7]:

modisDown = pymodis.downmodis.downModis(destinationFolder=dest, tiles=tiles, today=day, delta=delta)
modisDown.connect()


# In[8]:

print dest


# In[10]:

from osgeo import gdal


# In[11]:

gdal.GetDriverByName('HDF5')


# In[12]:

gdal.GetDriverByName('HDF4')


# In[ ]:



