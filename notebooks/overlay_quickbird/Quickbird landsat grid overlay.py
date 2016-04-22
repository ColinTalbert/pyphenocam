
# coding: utf-8

# In[1]:

import os

import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import data, img_as_float

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
import skimage.transform as trans

import shutil
import itertools
from datetime import datetime

import rasterio


# In[2]:

import sys
sys.path.append(r"..\..")

import pyphenocam


# ### Outside of this notebook we extracted a chunk of landsat to use as a managable reference grid

# In[3]:

landsat_subset_fname = r"J:\Projects\NCCSC\phenocam\DerivedData\UTM\landsat_subset.tif"
landsat = rasterio.open(landsat_subset_fname)
landsat_data = landsat.read()


# In[4]:

get_ipython().magic(u'matplotlib inline')
plt.imshow(landsat_data[0,:,:], interpolation='none')
pyphenocam.plotting.format_photo_axes(plt.gca())


# ## This was overlaid on our ArcScene photo recreation and converted to a couple of grid representations

# In[5]:

from IPython.display import Image
Image(filename=r"J:\Projects\NCCSC\phenocam\Tools\3D_visualization\quickbird_landsat.png")


# ### Which are then exported as flat images for use in the rest of this notebook

# In[6]:

ls_red_grid_fname = r"J:\Projects\NCCSC\phenocam\DerivedData\UTM\FinalLandsatGrids\red_gridlines.bmp"
ls_grid_index_fname = r"J:\Projects\NCCSC\phenocam\DerivedData\UTM\FinalLandsatGrids\IndexGrid.bmp"

quickbird = pyphenocam.dataaccess.get_site('quickbird')
which_img = -197


# In[7]:

get_ipython().magic(u'matplotlib inline')
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
local_fname = quickbird.get_local_image_fname(quickbird.midday_fnames[which_img])
# print local_fname
sample_image = quickbird.get_local_image(quickbird.midday_fnames[which_img])
ax.imshow(sample_image)

# pyphenocam.plotting.add_rois(ax, quickbird, vistype='line', lw=7, alpha=0.9)
# 
pyphenocam.plotting.format_photo_axes(ax)
# plt.legend(loc=1)

exposure = pyphenocam.headerextraction.get_exposure(local_fname)
print "Extracted exposure: ", exposure


# ### Our landsat grid image create in ArcScene

# In[8]:

data_grid = skimage.io.imread(ls_red_grid_fname)


# In[9]:

plt.figure(figsize=(10, 7))
plt.imshow(data_grid)
pyphenocam.plotting.format_photo_axes(plt.gca())


# ### But how do these align?

# In[10]:

# While approximatly equal in aspect we need to reshape our grid overlay
#to match the dimensions of our photos
data_trans = trans.resize(data_grid, 
            (sample_image.shape[0], sample_image.shape[1], 3), preserve_range=False)

plt.imshow(data_trans[:,:,1]<1)
pyphenocam.plotting.format_photo_axes(plt.gca())


# In[11]:

# And we need to 'burn' the red lines above into our sample image for display
superimposed = sample_image.copy()
superimposed[data_trans[:,:,1]<1, 0] = 1.0
superimposed[data_trans[:,:,1]<1, 1] = 0.0
superimposed[data_trans[:,:,1]<1, 2] = 0.0



# In[12]:

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111)
local_fname = quickbird.get_local_image_fname(quickbird.midday_fnames[which_img])
# print local_fname
ax.imshow(superimposed)
pyphenocam.plotting.format_photo_axes(ax)



# ## Now to crosswalk this to specific Landsat pixels

# In[13]:

index_grid = skimage.io.imread(ls_grid_index_fname)
index_grid = trans.resize(index_grid, (sample_image.shape[0], sample_image.shape[1], 3), preserve_range=True, order=0)

index_grid = np.ma.masked_where(index_grid > 200, index_grid)


# ##### This grid was construncted such that the red channel contains an index to the landsat pixel row and the green channel contains an indexto the landsat pixel column

# In[14]:

fig, axes = plt.subplots(1, 3, figsize=(15,4))

for index, cmap in enumerate([mpl.cm.Reds, mpl.cm.Greens, mpl.cm.Blues]):
    axes[index].imshow(index_grid[:,:,index], cmap=cmap)
    pyphenocam.plotting.format_photo_axes(axes[index])
    pyphenocam.plotting.add_inner_title(axes[index], cmap.name[:-1], 9)

    
#Note that the blue channel is not used


# In[15]:

single_pixel = np.logical_and(index_grid[:,:,0]==60, index_grid[:,:,1]==99)

single_pixel = np.ma.asarray(trans.resize(single_pixel, 
            (sample_image.shape[0], sample_image.shape[1], 3), preserve_range=False))[:,:,1]

single_pixel.mask = single_pixel==False

fig, ax = plt.subplots(1, figsize=(20,10))
ax.imshow(superimposed)
ax.imshow(single_pixel, alpha = 1.0, cmap=mpl.cm.Reds_r, interpolation='none')
pyphenocam.plotting.format_photo_axes(ax)


# # But how do we get the actual coordinates of the pixel centroids instead of this rather arbitrary index?

# In[16]:

x = 1
y = 1

landsat.affine *  (x, y)


# In[17]:

plt.imshow(landsat_data[0,:,:], interpolation='none')
pyphenocam.plotting.format_photo_axes(plt.gca())
plt.scatter(*(x, y))


# In[18]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ax_proj = ccrs.LambertConformal()
landsat_proj = ccrs.UTM(zone=12, globe=ccrs.Globe(datum='WGS84',
                                              ellipse='WGS84'))
geodetic = ccrs.Geodetic()

fig = plt.figure(figsize=(15, 15))
ax_extent = [quickbird.x - 0.015, quickbird.x + 0.015, quickbird.y - 0.002, quickbird.y + 0.015]

img_extents = [landsat.bounds.left, landsat.bounds.right, landsat.bounds.bottom, landsat.bounds.top]

ax = plt.axes(projection=ax_proj)
ax.set_extent(ax_extent, ccrs.Geodetic())
ax.imshow(landsat_data[0, :, :], origin='upper', extent=img_extents, transform=landsat_proj, interpolation='none', 
          cmap=mpl.cm.jet_r, vmin=9267, vmax=12233)

# # ax.set_xmargin(0.05)
# # ax.set_ymargin(0.10)

# mark a known place to help us geo-locate ourselves
locx, locy = list(ax_proj.transform_point(quickbird.x, quickbird.y, geodetic))
ax.plot(locx, locy, 'bo', markersize=15, color='black', alpha=0.5)
ax.text(locx+75, locy-15, 'quickbird camera location', bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})

ax.coastlines()
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.BORDERS)
ax.gridlines()
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray')


# In[19]:

get_ipython().magic(u'matplotlib qt4')
from ipywidgets import interactive

col_index, row_index = 0,0

def plot_one(col_index=30, row_index=50):
    single_pixel = np.logical_and(index_grid[:,:,0]==col_index*3, index_grid[:,:,1]==row_index*3)
    single_pixel = np.ma.asarray(trans.resize(single_pixel, 
                (sample_image.shape[0], sample_image.shape[1], 3), preserve_range=False))[:,:,1]
    single_pixel.mask = single_pixel==False

    fig = plt.figure(figsize=(25, 15))
    ax = plt.subplot(121)
    ax.imshow(superimposed)
    ax.imshow(single_pixel, alpha = 1.0, cmap=mpl.cm.Reds_r, interpolation='none')
    pyphenocam.plotting.format_photo_axes(ax)

    ax_proj = landsat_proj
    ax2 = plt.subplot(122, projection=ax_proj)
    ax2.set_extent(ax_extent, ccrs.Geodetic())
    ax2.imshow(landsat_data[0, :, :], origin='upper', extent=img_extents, transform=landsat_proj, interpolation='none', 
          cmap=mpl.cm.jet_r, vmin=9267, vmax=12233)
    colx, coly = landsat.affine * (col_index, row_index)
    colx += landsat.transform[1]/2.
    coly += landsat.transform[5]/2.
    
    colxgeo, colygeo = list(ax_proj.transform_point(colx, coly, landsat_proj))
    ax2.plot(colxgeo, colygeo, 'bo', markersize=10, color='black', alpha=0.5)
    ax2.text(colxgeo+75, colygeo-15, 'highlighted pixel', bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})

    # mark a known place to help us geo-locate ourselves
    locx, locy = list(ax_proj.transform_point(quickbird.x, quickbird.y, geodetic))
    ax2.plot(locx, locy, 'bo', markersize=15, color='black', alpha=0.5)
    ax2.text(locx+75, locy-10, 'quickbird camera location', bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})

    plt.tight_layout()
    
interactive(plot_one, col_index=(0, landsat.shape[0], 1), row_index=(0, landsat.shape[1], 1))

