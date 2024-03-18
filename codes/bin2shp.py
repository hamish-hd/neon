import numpy as np
import pandas as pd

# Function to read the ENVI .hdr file and extract metadata
def read_hdr_file(hdr_file):
    metadata = {}
    with open(hdr_file, 'r') as f:
        for line in f:
            parts = line.strip().split('=')
            if len(parts) == 2:
                key, value = parts
                metadata[key.strip()] = value.strip()
    return metadata

# Function to read the ENVI .bin file and extract data
def read_envi_bin_file(bin_file):
    data = np.fromfile(bin_file, dtype=np.float64)
    return data

# Read .hdr file to get metadata
hdr_file = '/data/gedi/_neon/NISARP_03112_polygon_additional.hdr'
hdr_metadata = read_hdr_file(hdr_file)

# Assuming the number of lines and samples per line are specified in the .hdr file
num_lines = int(hdr_metadata['lines'])

# Read .bin file
bin_file = '/data/gedi/_neon/NISARP_03112_polygon_additional.bin'
bin_data = read_envi_bin_file(bin_file)

# Reshape the data into a 2D array with dimensions (311, 4) for band 1
band1_data = bin_data[:num_lines * 4].reshape(num_lines, 4)

# Create DataFrame to store latitude information
df = pd.DataFrame(band1_data, columns=['Latitude_1', 'Latitude_2', 'Latitude_3', 'Latitude_4'])

# Print the DataFrame
print(df)

#%%
import numpy as np
import pandas as pd

# Function to read the ENVI .hdr file and extract metadata
def read_hdr_file(hdr_file):
    metadata = {}
    with open(hdr_file, 'r') as f:
        for line in f:
            parts = line.strip().split('=')
            if len(parts) == 2:
                key, value = parts
                metadata[key.strip()] = value.strip()
    return metadata

# Function to read the ENVI .bin file and extract data
def read_envi_bin_file(bin_file):
    data = np.fromfile(bin_file, dtype=np.float64)
    return data

# Read .hdr file to get metadata
hdr_file = '/data/gedi/_neon/NISARP_03112_polygon_additional.hdr'
hdr_metadata = read_hdr_file(hdr_file)

# Assuming the number of lines and samples per line are specified in the .hdr file
num_lines = int(hdr_metadata['lines'])

# Read .bin file
bin_file = '/data/gedi/_neon/NISARP_03112_polygon_additional.bin'
bin_data = read_envi_bin_file(bin_file)

# Reshape band 1 data into a 2D array with dimensions (311, 4)
band1_data = bin_data[:num_lines * 4].reshape(num_lines, 4)

# Reshape band 2 data into a 2D array with dimensions (311, 4)
band2_data = bin_data[num_lines * 4:].reshape(num_lines, 4)

# Create DataFrame to store latitude information from band 1
latitude_columns = ['Latitude_1', 'Latitude_2', 'Latitude_3', 'Latitude_4']
latitude_df = pd.DataFrame(band1_data, columns=latitude_columns)

# Create DataFrame to store longitude information from band 2
longitude_columns = ['Longitude_1', 'Longitude_2', 'Longitude_3', 'Longitude_4']
longitude_df = pd.DataFrame(band2_data, columns=longitude_columns)

# Combine latitude and longitude DataFrames
df = pd.concat([latitude_df, longitude_df], axis=1)

# Print the DataFrame
print(df)
#%%
import geopandas as gpd
from shapely.geometry import Polygon

# Assuming you have a list of polygons' coordinates
polygons = [Polygon([(lon1, lat1), (lon2, lat2), (lon3, lat3), (lon4, lat4)]) 
            for (lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4) in 
            zip(df['Longitude_1'], df['Latitude_1'], df['Longitude_2'], df['Latitude_2'], 
                df['Longitude_3'], df['Latitude_3'], df['Longitude_4'], df['Latitude_4'])]

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=polygons)

# Set the coordinate reference system (CRS) to EPSG 4326 (WGS 84)
gdf.crs = "EPSG:4326"

# Save as shapefile
output_shapefile = bin_file[:-4]+'.shp'
gdf.to_file(output_shapefile)
