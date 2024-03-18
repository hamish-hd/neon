'''
This code will read the NEON_struct-ecosystem and subfolders within it. For each subfolder, representing a year, a mosaic is created and saved. 
Then it will read the shapefile of the plots and subplots and then calculate the mean, RH95 and count of the pixels within each plot.
'''

import os
import geopandas as gpd
from osgeo import gdal, osr
from rasterstats import zonal_stats
import numpy as np

from pyproj import CRS
# Set input directory
input_directory = '/data/gedi/_neon/_new/ORNL/NEON_struct-ecosystem'

# Iterate over subdirectories
for subdirectory in os.listdir(input_directory):
    subdirectory_path = os.path.join(input_directory, subdirectory)

    if os.path.isdir(subdirectory_path):
        mosaic_path = os.path.join(os.path.dirname(os.path.dirname(subdirectory_path)),
                                   f'chm_{subdirectory[9:13]}_{subdirectory[28:32]}.tif')
        tif_files = [f for f in os.listdir(subdirectory_path) if f.endswith('.tif')]

        output_srs = osr.SpatialReference()
        output_srs.ImportFromEPSG(4326)
        
        
        gdal.Warp(mosaic_path, [os.path.join(subdirectory_path, tif) for tif in tif_files], dstSRS=output_srs)

        print(f'Mosaic saved to: {mosaic_path}')

        shapefile_path = '/data/gedi/_neon/NEON_plots_subplots/final_plots_not.shp'
        gdf = gpd.read_file(shapefile_path)
        gdf = gdf.to_crs(CRS.from_epsg(32616))
        gdf['area'] = gdf.geometry.area.round()
        gdf = gdf.to_crs(CRS.from_epsg(4326))
        #gdf = gdf[gdf['subplotID'].str.contains('_400')]

        stats = zonal_stats(gdf['geometry'], mosaic_path, stats='percentile_95 mean count', nodata=-9999)

        result_df = gdf.copy()
        result_df['chm_mean'] = [stat['mean'] if stat else np.nan for stat in stats]
        result_df['chm_RH95'] = [stat['percentile_95'] if stat else np.nan for stat in stats]
        result_df['chm_count'] = [stat['count'] if stat else np.nan for stat in stats]
        result_df['year'] = subdirectory[28:32]

        output_shapefile_path = os.path.join(os.path.dirname(os.path.dirname(subdirectory_path)),
                                   f'chm_height_{subdirectory[9:13]}_{subdirectory[28:32]}.shp')
        result_df.to_file(output_shapefile_path)
        print(f'Shapefile saved to: {output_shapefile_path}')
        
        output_csv_path = os.path.join(os.path.dirname(os.path.dirname(subdirectory_path)),
                                   f'chm_height_{subdirectory[9:13]}_{subdirectory[28:32]}.csv')
        result_df.to_csv(output_csv_path)
        print(f'CSV saved to: {output_csv_path}')