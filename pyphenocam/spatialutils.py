import os
import sys

import datetime as dt
import math

import fiona
from fiona.crs import from_epsg
from shapely.geometry import Point, mapping

import rasterio

__all__ = []

def get_raster_extents(raster):
    return [raster.bounds.left, raster.bounds.right, 
            raster.bounds.bottom, raster.bounds.top]

def xy2colrow(x, y, raster):
    a = raster.affine
    col, row = [int(math.floor(coord)) for coord in ~a * (x, y)]
    return col, row

def colrow2xy(col, row, raster):
    a = raster.affine
    return a * (col, row)

