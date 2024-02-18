#%% Tree locations
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import math
import os
from shapely.wkt import loads
import datetime

def calculate_tree_location(origin_point, azimuth, distance):
    azimuth_rad = math.radians(azimuth)
    delta_lat = distance * math.cos(azimuth_rad)
    delta_lon = distance * math.sin(azimuth_rad)
    new_lat = origin_point.y + (delta_lat / 111111)  
    new_lon = origin_point.x + (delta_lon / (111111 * math.cos(math.radians(origin_point.y))))
    return Point(new_lon, new_lat)

neon_plotpoints = '/data/gedi/_neon/NEON_plots_subplots/neon_not_points.shp'
#Give the shapefile that has plot and points of all NEON sites

plot_points = gpd.read_file(neon_plotpoints)
plot_points = plot_points.to_crs(epsg = 4326)
############
struct_plant_dir = '/data/gedi/_neon/LENO_2021/NEON_struct-plant/'
#Directory of NEON plant structure

maptag = pd.DataFrame()

for root, dirs, files in os.walk(struct_plant_dir):
    for file in files:
        if file.endswith('.csv') and 'mappingandtagging' in file:
            file_path = os.path.join(root, file)
            print(file_path)
            df = pd.read_csv(file_path)
            df = df.dropna(subset=['pointID'])
            df['pointID'] = df['pointID'].astype(int)
            df['plot_point'] = df['plotID'].astype(str) + '_' + df['pointID'].astype(str)
            maptag = pd.concat([maptag, df], ignore_index=True)
            print('MapTag shape:',maptag.shape)

maptag = maptag
maptag.drop_duplicates(inplace=True)
maptag.to_csv('/data/gedi/_neon/LENO_2021/NEON_struct-plant/maptag.csv', index=False)

print(maptag)

#############
merged_data = pd.merge(plot_points, maptag, on='plot_point')

merged_data['geometry'] = merged_data.apply(lambda row: calculate_tree_location(row['geometry'], row['stemAzimuth'], row['stemDistance']), axis=1)
tree_locations = merged_data[['plot_point', 'geometry','individualID','taxonID', 'scientificName', 'taxonRank']]
#########################
tree_measurements = pd.DataFrame()

for root, dirs, files in os.walk(struct_plant_dir):
    for file in files:
        if file.endswith('.csv') and 'apparentindividual' in file:
            file_path = os.path.join(root, file)
            
            df = pd.read_csv(file_path)
            print('rows :',df.shape)
            print(file_path)
            #df = df.dropna(subset=['tagStatus'])
            tree_measurements = pd.concat([tree_measurements, df], ignore_index=True)
            print('merged df shape:',tree_measurements.shape)
            #print('shape:',tree_measurements.shape)

tree_measurements.drop_duplicates(inplace=True)


tree_measurements.to_csv('/data/gedi/_neon/LENO_2021/NEON_struct-plant/app_indi.csv', index=False)

print(tree_measurements)
########################

n = gpd.GeoDataFrame(pd.merge(tree_locations, tree_measurements, on='individualID', how='inner'))
#############
current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

n.crs = 'EPSG:4326'

shapefile_path = os.path.join(os.path.dirname(os.path.dirname(struct_plant_dir)), f'tree_locations_{current_datetime}.shp')
csv_path = os.path.join(os.path.dirname(os.path.dirname(struct_plant_dir)), f'tree_locations_{current_datetime}.csv')

n.to_file(shapefile_path, driver='ESRI Shapefile')
print(f"Tree locations exported to Shapefile: {shapefile_path}")

n.to_csv(csv_path, index=False)

print(f"Tree locations exported to CSV: {csv_path}")

##########################################################################