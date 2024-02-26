import os
import glob
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from pyproj import Proj, transform
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterstats import zonal_stats
import geopandas as gpd
from osgeo import gdal, ogr, osr
import numpy as np
from pyproj import CRS
import time

start_time = time.time()
#%% Create chm mosaic geotiff
input_folder = '/data/gedi/_neon/LENO_2021/NEON_struct-ecosystem/NEON.D08.LENO.DP3.30015.001.2021-05.basic.20240212T100230Z.RELEASE-2024'
mosaic_path = os.path.dirname(os.path.dirname(input_folder))+'/'+os.path.basename(input_folder)[9:13]+'_'+os.path.basename(input_folder)[28:32]+'.tif'
tif_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

output_srs = osr.SpatialReference()
output_srs.ImportFromEPSG(4326)

gdal.Warp(mosaic_path, [os.path.join(input_folder, tif) for tif in tif_files], dstSRS=output_srs)

print(f'Mosaic created and saved to: {mosaic_path}')

#%%
# Load shapefile
shapefile_path = '/data/gedi/_neon/NEON_plots_subplots/final_plots_not.shp'
gdf = gpd.read_file(shapefile_path)
gdf = gdf.to_crs(CRS.from_epsg(32616))
gdf['area'] = gdf.geometry.area.round()
gdf = gdf.to_crs(CRS.from_epsg(4326))



# Load raster
mosaic_path = '/data/gedi/_neon/LENO/LENO/NISARP_03112_CHM_2019.tif'
output_path = os.path.dirname(mosaic_path)+'/'+os.path.basename(mosaic_path)[:-4]+'_4326.tif'
target_epsg = 'EPSG:4326'
gdal.Warp(output_path, mosaic_path, dstSRS=target_epsg)


def calculate_mean_top_10_percent(pixels):
    pixels = np.array(pixels)
    threshold = np.percentile(pixels, 95)  
    top_10_percent = pixels[pixels >= threshold]
    return np.mean(top_10_percent) if len(top_10_percent) > 0 else np.nan

#stats = zonal_stats(gdf['geometry'], mosaic_path, stats='count mean', nodata=-9999, add_stats={'mean_top_5_percent': calculate_mean_top_10_percent})
stats = zonal_stats(gdf['geometry'], output_path, stats='percentile_95 mean count', nodata=-9999)


result_df = gdf.copy()
result_df['chm_mean'] = [stat['mean'] if stat else np.nan for stat in stats]
result_df['chm_RH95'] = [stat['percentile_95'] if stat else np.nan for stat in stats]
result_df['chm_count'] = [stat['count'] if stat else np.nan for stat in stats]
#result_df['year'] = os.path.basename(input_folder)[28:32]
result_df['year'] = 2019
print(result_df[['geometry', 'chm_mean','chm_RH95']])

'''
result_df[result_df['subplotID'].str.startswith(os.path.basename(input_folder)[9:13])]
output_shapefile_path = '/data/gedi/_neon/LENO_2021/chm_height_'+os.path.basename(input_folder)[9:13]+'_'+os.path.basename(input_folder)[28:32]+'.shp'
result_df.to_file(output_shapefile_path)
print(f'Results saved to: {output_shapefile_path}')
result_df.to_csv('/data/gedi/_neon/LENO_2021/chm_height_'+os.path.basename(input_folder)[9:13]+'_'+os.path.basename(input_folder)[28:32]+'.csv')
'''

output_shapefile_path = '/data/gedi/_neon/LENO_2021/chm_height_LENO_2019.shp'
result_df.to_file(output_shapefile_path)
result_df.to_csv('/data/gedi/_neon/LENO_2021/chm_height_LENO_2019.csv')

