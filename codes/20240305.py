import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from osgeo import ogr, osr,gdal
from rasterio.warp import calculate_default_transform, reproject, Resampling
import gc
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import matplotlib.pyplot as plt

#%%
agb_file = '/data/gedi/_neon/_new/LENO/agb_LENO_2019.tif'


HH_vrt = '/data/gedi/hd/Desktop/NISARP_03112_CX_129_HH.vrt'
HV_vrt = '/data/gedi/hd/Desktop/NISARP_03112_CX_129_HV.vrt'

plots = '/data/gedi/plots.shp'
gdf = gpd.read_file(plots).to_crs(epsg=4326)

def reproject_to_4326(input_dataset, output_dataset):
    src_ds = gdal.Open(input_dataset)
    
    dst_crs = osr.SpatialReference()
    dst_crs.ImportFromEPSG(4326)
    
    warp_options = gdal.WarpOptions(
        format='GTiff',
        dstSRS=dst_crs,
        resampleAlg=gdal.GRA_NearestNeighbour
    )

    gdal.Warp(output_dataset, src_ds, options=warp_options)

    src_ds = None

def reproject_vrt(vrt_file, output_vrt_file):
    dataset_paths = gdal.Open(vrt_file).GetFileList()

    reprojected_datasets = []
    for idx, input_dataset in enumerate(dataset_paths):
        output_dataset = f'/data/gedi/hd/Desktop/reprojected_dataset_{idx}.tif'  # Adjust the output path as needed
        reproject_to_4326(input_dataset, output_dataset)
        reprojected_datasets.append(output_dataset)

    vrt_options = gdal.BuildVRTOptions(resampleAlg=gdal.GRA_NearestNeighbour)
    gdal.BuildVRT(output_vrt_file, reprojected_datasets, options=vrt_options)



reproject_vrt(HH_vrt, HH_vrt[:-4]+'_4326.vrt')

#%%
HH_path = '/data/gedi/hd/Desktop/NISARP_03112_CX_129_HH.vrt'
dataset = gdal.Open(HH_path)
if dataset is None:
    print("Failed to open the VRT file.")
    exit(1)
HH = []
for band_index in range(dataset.RasterCount):
    band = dataset.GetRasterBand(band_index + 1)
    band_data = band.ReadAsArray()
    HH.append(band_data)
del dataset
gc.collect()

    
HV_path = '/data/gedi/hd/Desktop/NISARP_03112_CX_129_HV.vrt'
dataset = gdal.Open(HV_path)
if dataset is None:
    print("Failed to open the VRT file.")
    exit(1)
HV = []
for band_index in range(dataset.RasterCount):
    band = dataset.GetRasterBand(band_index + 1)
    band_data = band.ReadAsArray()
    HV.append(band_data)
del dataset
gc.collect()


plt.imshow(HV[6], cmap='gist_earth_r',vmin=0,vmax=1)
plt.colorbar()
plt.show()