import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats
import numpy as np
from rasterio.warp import reproject, Resampling
from pyproj import Proj, transform
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from osgeo import gdal, gdalconst, osr
from rasterstats import zonal_stats

#%%
# Replace these paths with your actual file paths
shapefile_path = 'file:///data/gedi/_neon/output_shapefile.shp'

# Read the shapefile
gdf = gpd.read_file(shapefile_path)
gdf = gdf.to_crs(epsg=32616)
gdf['area'] = gdf['geometry'].area  # Calculate area in square meters
gdf = gdf.to_crs(epsg=4326)
print(gdf.head())
#%% HH zonal stats
HH_vrt_path = '/data/gedi/hd/Desktop/NISARP_03112_CX_129_HH.vrt'
with rasterio.open(HH_vrt_path) as src:
    num_bands = src.count

columns = [f'HH_{i}_mean' for i in range(1, num_bands + 1)] + \
          [f'HH_{i}_stdev' for i in range(1, num_bands + 1)]
HH_df = pd.DataFrame(columns=columns, index=gdf.index)

for band_number in range(1, num_bands + 1):
    with rasterio.open(HH_vrt_path) as src:
        for index, row in gdf.iterrows():
            geometry = row['geometry']
            out_image, out_transform = mask(src, [geometry], crop=True, indexes=band_number)
            values = out_image.flatten()
            values = values[~np.isnan(values)]

            mean_value = np.mean(values)
            std_dev_value = np.std(values)

            HH_df.at[index, f'HH_{band_number}_mean'] = mean_value
            HH_df.at[index, f'HH_{band_number}_stdev'] = std_dev_value

print(HH_df)

# Save the DataFrame to a CSV file
#HH_df.to_csv('/path/to/zonal_stats_output.csv', index=False)

#%% HV zonal stats
HV_vrt_path = '/data/gedi/hd/Desktop/NISARP_03112_CX_129_HH.vrt'
with rasterio.open(HV_vrt_path) as src:
    num_bands = src.count

columns = [f'HV_{i}_mean' for i in range(1, num_bands + 1)] + \
          [f'HV_{i}_stdev' for i in range(1, num_bands + 1)]
HV_df = pd.DataFrame(columns=columns, index=gdf.index)

for band_number in range(1, num_bands + 1):
    with rasterio.open(HV_vrt_path) as src:
        for index, row in gdf.iterrows():
            geometry = row['geometry']

            out_image, out_transform = mask(src, [geometry], crop=True, indexes=band_number)

            values = out_image.flatten()
            values = values[~np.isnan(values)]

            mean_value = np.mean(values)
            std_dev_value = np.std(values)

            HV_df.at[index, f'HV_{band_number}_mean'] = mean_value
            HV_df.at[index, f'HV_{band_number}_stdev'] = std_dev_value

print(HV_df)
#%% merge df
df = pd.concat([HH_df, HV_df], axis=1)
df.to_csv('/data/gedi/_neon/_new/LENO/nisar_data_m2_leno.csv', index=False)
df['area'] = gdf['area']


#%%
# Input files
agb_file = '/data/gedi/_neon/_new/LENO/agb_m2_LENO_2019.tif'
#shapefile_path ='file:///data/gedi/_neon/output_shapefile.shp'

gdf = gpd.read_file(shapefile_path)
reprojected_tiff_path = agb_file[:-4]+'_4326.tif'
input_ds = gdal.Open(agb_file, gdalconst.GA_ReadOnly)
input_proj = input_ds.GetProjection()
input_geo = input_ds.GetGeoTransform()
target_srs = osr.SpatialReference()
target_srs.ImportFromEPSG(4326)
# Create an output raster dataset
output_ds = gdal.Warp(reprojected_tiff_path, input_ds, dstSRS=target_srs.ExportToWkt())
# Close datasets
input_ds = None
output_ds = None
#%%
stats = zonal_stats(gdf['geometry'], reprojected_tiff_path, stats='max', nodata=-9999)

# Add statistics to DataFrame
df['AGB'] = [stat['max'] if stat else np.nan for stat in stats]
df['AGB_ha'] = df['AGB']/df['area']*10















#%% Plot 
plt.figure(figsize=(10, 10))
plt.scatter(df['AGB_ha'],10*np.log10(df['HV_4_mean'].astype('float64')),color='black')
plt.xlabel('AGB_ha')
plt.ylabel('HV_1')
#plt.xlim(0,150)
plt.ylim(-15,0)
#plt.xlim(0,100)
plt.title('HV_1_mean vs AGB')
plt.grid(True)
plt.show()
