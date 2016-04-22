
# coding: utf-8

# In[7]:

import os

import rasterio


# In[8]:

import sys
sys.path.append(r"..\..")

import pyphenocam


# In[9]:

base_dname = r"J:\Projects\NCCSC\phenocam\DerivedData\nationalelkrefuge"

elevation_subset_fname = os.path.join(base_dname, "Landsat", "Subset", "landsat_subset.tif")
landsat = rasterio.open(landsat_subset_fname)
landsat_data = landsat.read()


# In[ ]:



