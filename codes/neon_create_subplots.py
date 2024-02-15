import geopandas as gpd
from shapely.geometry import Polygon, Point
import pandas as pd

#%%
original_shapefile = gpd.read_file("file:///data/gedi/_neon/All_NEON_TOS_Plots_V10 (copy)/All_NEON_TOS_Plot_Polygons_V10.shp")
original_shapefile = original_shapefile.to_crs(epsg=32616)
new_polygons = gpd.GeoDataFrame(columns=['geometry', 'subplotID'])

def subdivide_4_parts(polygon, identifier_suffix):
    xmin, ymin, xmax, ymax = polygon.bounds
    x_mid = (xmin + xmax) / 2
    y_mid = (ymin + ymax) / 2

    bottom_left = Polygon([(xmin, ymin), (x_mid, ymin), (x_mid, y_mid), (xmin, y_mid)])
    bottom_right = Polygon([(x_mid, ymin), (xmax, ymin), (xmax, y_mid), (x_mid, y_mid)])
    top_right = Polygon([(x_mid, y_mid), (xmax, y_mid), (xmax, ymax), (x_mid, ymax)])
    top_left = Polygon([(xmin, y_mid), (x_mid, y_mid), (x_mid, ymax), (xmin, ymax)])

    return [
        (bottom_left, f"{identifier_suffix}_21_400"),
        (bottom_right, f"{identifier_suffix}_23_400"),
        (top_right, f"{identifier_suffix}_41_400"),
        (top_left, f"{identifier_suffix}_39_400")
    ]

def subdivide_16_parts(polygon, identifier_suffix):
    subdivisions_16_parts = []
    names_16_parts = ['21', '22', '23', '24', '30', '31', '32', '33', '39', '40', '41', '42', '48', '49', '50', '51']

    for i, name in enumerate(names_16_parts):
        col = i % 4
        row = i // 4

        x_min_sub = polygon.bounds[0] + col * (polygon.bounds[2] - polygon.bounds[0]) / 4
        y_min_sub = polygon.bounds[1] + row * (polygon.bounds[3] - polygon.bounds[1]) / 4
        x_max_sub = x_min_sub + (polygon.bounds[2] - polygon.bounds[0]) / 4
        y_max_sub = y_min_sub + (polygon.bounds[3] - polygon.bounds[1]) / 4

        subdivision = Polygon([(x_min_sub, y_min_sub), (x_max_sub, y_min_sub), (x_max_sub, y_max_sub), (x_min_sub, y_max_sub)])
        new_plot_id = f"{identifier_suffix}_{name}_100"
        subdivisions_16_parts.append((subdivision, new_plot_id))

    return subdivisions_16_parts

def subdivide_4_subplots(polygon, identifier_suffix):
    subdivisions_4_subplots = []
    names_4_subplots = ['1', '2', '3', '4']

    for i, name in enumerate(names_4_subplots):
        col = i % 2
        row = i // 2

        x_min_sub = polygon.bounds[0] + col * (polygon.bounds[2] - polygon.bounds[0]) / 2
        y_min_sub = polygon.bounds[1] + row * (polygon.bounds[3] - polygon.bounds[1]) / 2
        x_max_sub = x_min_sub + (polygon.bounds[2] - polygon.bounds[0]) / 2
        y_max_sub = y_min_sub + (polygon.bounds[3] - polygon.bounds[1]) / 2

        subdivision = Polygon([(x_min_sub, y_min_sub), (x_max_sub, y_min_sub), (x_max_sub, y_max_sub), (x_min_sub, y_max_sub)])
        new_plot_id = f"{identifier_suffix[:-4]}_25_{name}"
        subdivisions_4_subplots.append((subdivision, new_plot_id))

    return subdivisions_4_subplots

for index, original_polygon in original_shapefile.iterrows():
    plot_dim = original_polygon['plotDim']

    if plot_dim == '40m x 40m':
        subdivisions_4_parts = subdivide_4_parts(original_polygon['geometry'], original_polygon['plotID'])
        
        for subdivision, new_plot_id in subdivisions_4_parts:
            new_polygons = pd.concat([new_polygons, gpd.GeoDataFrame({'geometry': [subdivision], 'plotID_identifier': [new_plot_id]}, crs=original_shapefile.crs)], ignore_index=True)

        subdivisions_16_parts = subdivide_16_parts(original_polygon['geometry'], original_polygon['plotID'])

        for subdivision, new_plot_id in subdivisions_16_parts:
            new_polygons = pd.concat([new_polygons, gpd.GeoDataFrame({'geometry': [subdivision], 'plotID_identifier': [new_plot_id]}, crs=original_shapefile.crs)], ignore_index=True)

            subdivisions_4_subplots = subdivide_4_subplots(subdivision, new_plot_id)

            for subplot, new_subplot_id in subdivisions_4_subplots:
                new_polygons = pd.concat([new_polygons, gpd.GeoDataFrame({'geometry': [subplot], 'plotID_identifier': [new_subplot_id]}, crs=original_shapefile.crs)], ignore_index=True)


new_polygons.drop(columns=['subplotID'], inplace=True)  
new_polygons.rename(columns={'plotID_identifier': 'subplotID'}, inplace=True)  

new_polygons.to_file("/data/gedi/_neon/NEON_plots_subplots/squares_not.shp")

#%%
original_shapefile = "/data/gedi/_neon/All_NEON_TOS_Plots_V10 (copy)/All_NEON_TOS_Plot_Polygons_V10.shp"

gdf = gpd.read_file(original_shapefile)

gdf = gdf.to_crs(epsg=32616)

def divide_and_label(row):
    x, y = row['geometry'].exterior.xy
    origin_x, origin_y = x[0], y[0]

    vertices = []

    for i in range(5):
        for j in range(5):
            vertices.append(Point(origin_x + j * 10, origin_y - i * 10))

    vertices_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(vertices), crs=gdf.crs)

    # Add a 'PlotID' column by repeating the 'plotID' value
    vertices_gdf['PlotID'] = row['plotID']

    # Add a 'Vertex_Number' column to label each vertex
    vertices_gdf['Vertex_Number'] = [
        f"{num}" for num in [
            57, 58, 59, 60, 61,
            48, 49, 50, 51, 52,
            39, 40, 41, 42, 43,
            30, 31, 32, 33, 34,
            21, 22, 23, 24, 25
        ]
    ]

    return vertices_gdf

# Apply the function to each row in the original GeoDataFrame and concatenate the resulting GeoDataFrames
result_gdf_list = []

for _, row in gdf.iterrows():
    result_gdf_list.append(divide_and_label(row))

# Concatenate the resulting GeoDataFrames and reset the index
result_gdf = pd.concat(result_gdf_list, ignore_index=True)
result_gdf['plot_pointID'] = result_gdf['PlotID'].astype(str) + '_' + result_gdf['Vertex_Number'].astype(str)
# Display the resulting GeoDataFrame
print(result_gdf)

output_shapefile_path = '/data/gedi/_neon/NEON_plots_subplots/neon_not_points.shp'

# Save the GeoDataFrame to a shapefile
result_gdf.to_file(output_shapefile_path)

#%%
point_gdf = result_gdf
def generate_squares_around_point(row, gdf_crs):
    # Extract the 'PlotID' and 'Vertex_Num' from the input row
    plot_id = row['PlotID']
    vertex_num = row['Vertex_Number']
    
    point = row['geometry']
    x, y = point.xy
    d = 3.16  # Distance from the point to the vertices

    # Calculate the coordinates of the squares' vertices
    square_1 = Polygon([(x[0], y[0]),
                        (x[0] + d, y[0]),
                        (x[0] + d, y[0] + d),
                        (x[0], y[0] + d),
                        (x[0], y[0])])

    square_2 = Polygon([(x[0], y[0]),
                        (x[0], y[0] + d),
                        (x[0] - d, y[0] + d),
                        (x[0] - d, y[0]),
                        (x[0], y[0])])

    square_3 = Polygon([(x[0], y[0]),
                        (x[0] - d, y[0]),
                        (x[0] - d, y[0] - d),
                        (x[0], y[0] - d),
                        (x[0], y[0])])

    square_4 = Polygon([(x[0], y[0]),
                        (x[0], y[0] - d),
                        (x[0] + d, y[0] - d),
                        (x[0] + d, y[0]),
                        (x[0], y[0])])

    # Extract 'Vertex_number' field to assign names to squares
    names = [f'{vertex_num}_10_1', f'{vertex_num}_10_2',
             f'{vertex_num}_10_3', f'{vertex_num}_10_4']

    return gpd.GeoDataFrame(geometry=[square_1, square_2, square_3, square_4], 
                            index=names, crs=gdf_crs, data={'PlotID': plot_id})

# Create an empty GeoDataFrame to store the squares
squares_series = gpd.GeoDataFrame()

# Apply the function to each row in the GeoDataFrame
for idx, row in point_gdf.iterrows():
    squares_series = pd.concat([squares_series, generate_squares_around_point(row, point_gdf.crs)])

# Display the resulting GeoDataFrame with squares and names
print(squares_series)

# Save the GeoDataFrame to a shapefile
squares_series.to_file('/data/gedi/_neon/NEON_plots_subplots/17_sq2.shp')


features_to_remove = [
    '21_10_2', '21_10_3', '21_10_4', '22_10_3', '22_10_4',
    '21_10_3', '23_10_4', '24_10_3', '24_10_4', '25_10_3',
    '25_10_4', '34_10_1', '34_10_4', '43_10_1', '43_10_4',
    '52_10_1', '52_10_4', '61_10_1', '61_10_2', '61_10_4',
    '60_10_1', '60_10_2', '59_10_1', '59_10_2', '58_10_1',
    '58_10_2', '57_10_1', '57_10_2', '57_10_3', '48_10_2',
    '48_10_3', '39_10_2', '39_10_3', '30_10_2', '30_10_3',
    '25_10_1', '23_10_3'
]

# Remove specified features
filtered_squares_series = squares_series[~squares_series.index.isin(features_to_remove)]

rename_mapping = {
    '22_10_2': '21_10_2',
    '30_10_4': '21_10_3',
    '31_10_3': '21_10_4',
    '23_10_2': '22_10_2',
    '31_10_4': '22_10_3',
    '32_10_3': '22_10_4',
    '24_10_2': '22_10_5',
    '24_10_2': '23_10_2',
    '32_10_4': '23_10_3',
    '33_10_3': '23_10_4',
    '25_10_2': '24_10_2',
    '33_10_4': '24_10_3',
    '34_10_3': '24_10_4',
    
    '31_10_2': '30_10_2',
    '39_10_4': '30_10_3',
    '40_10_3': '30_10_4',
    
    '32_10_2': '31_10_2',
    '40_10_4': '31_10_3',
    '41_10_3': '31_10_4',
    
    '33_10_2': '32_10_2',
    '41_10_4': '32_10_3',
    '42_10_3': '32_10_4',
    
    '34_10_2': '33_10_2',
    '42_10_4': '33_10_3',
    '43_10_3': '33_10_4',
    
    '40_10_2': '39_10_2',
    '48_10_4': '39_10_3',
    '49_10_3': '39_10_4',
    
    '41_10_2': '40_10_2',
    '49_10_4': '40_10_3',
    '50_10_3': '40_10_4',
    
    '42_10_2': '41_10_2',
    '50_10_4': '41_10_3',
    '51_10_3': '41_10_4',
    
    '43_10_2': '42_10_2',
    '51_10_4': '42_10_3',
    '52_10_3': '42_10_4',
    
    '49_10_2': '48_10_2',
    '57_10_4': '48_10_3',
    '58_10_3': '48_10_4',
    
    '50_10_2': '49_10_2',
    '58_10_4': '49_10_3',
    '59_10_3': '49_10_4',
    
    '51_10_2': '50_10_2',
    '59_10_4': '50_10_3',
    '60_10_3': '50_10_4',
    
    '52_10_2': '51_10_2',
    '60_10_4': '51_10_3',
    '61_10_3': '51_10_4',
    
    
}

# Rename specified features
filtered_squares_series.rename(index=rename_mapping, inplace=True)
filtered_squares_series['subplotID'] = filtered_squares_series['PlotID'] + '_' + filtered_squares_series.index
filtered_squares_series.drop(columns=['PlotID'], inplace=True) 

# Display the resulting GeoDataFrame
print(filtered_squares_series)

# Save the GeoDataFrame to a new shapefile
filtered_squares_series.to_file('/data/gedi/_neon/NEON_plots_subplots/squares_10_not.shp')

#%%

import geopandas as gpd
from shapely.geometry import Point

# Replace 'path_to_shapefile_1' and 'path_to_shapefile_2' with your actual file paths
shapefile_path_1 = '/data/gedi/_neon/NEON_plots_subplots/squares_not.shp'
shapefile_path_2 = '/data/gedi/_neon/NEON_plots_subplots/squares_10_not.shp'

# Read the shapefiles
gdf_1 = gpd.read_file(shapefile_path_1)
gdf_2 = gpd.read_file(shapefile_path_2)

# Convert to EPSG 4326 (WGS84)
gdf_1 = gdf_1.to_crs(epsg=4326)
gdf_2 = gdf_2.to_crs(epsg=4326)

# Combine the GeoDataFrames
combined_gdf = gpd.GeoDataFrame(pd.concat([gdf_1, gdf_2], axis=0, sort=False))
combined_gdf = combined_gdf.drop(columns=['index'])
# Display the resulting GeoDataFrame
print(combined_gdf)

# Save the combined GeoDataFrame to a new shapefile
combined_gdf.to_file('/data/gedi/_neon/NEON_plots_subplots/final_plots_not.shp')
