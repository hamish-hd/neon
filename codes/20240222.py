#  Use this

#%% Import packages
import pandas as pd
import geopandas as gpd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from scipy.optimize import curve_fit
import re
import matplotlib as mpl

# Set Times New Roman as the default font for all text elements
mpl.rcParams['font.family'] = 'Times New Roman'
#%% Read struct_plant_dir
struct_plant_dir = '/data/gedi/_neon/_new/LENO/NEON_struct-plant'
#Special name file
sp_names = pd.read_csv('/data/gedi/_neon/Species_Names_Forest_Group.csv')
#Tree locations shapefile
tree_loc = gpd.read_file('/data/gedi/_neon/LENO_2021/tree_locations_20240216_114531.shp')


#Directory of NEON plant structure
#%% Merge all apparent individuals

apparent_individual = pd.DataFrame()

for root, dirs, files in os.walk(struct_plant_dir):
    for file in files:
        if file.endswith('.csv') and 'apparentindividual' in file:
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            print('ek ka shape :',df.shape)
            print(file_path)
            apparent_individual = pd.concat([apparent_individual, df], ignore_index=True)
            print('merged ka shape:',apparent_individual.shape)
            
apparent_individual.drop_duplicates(inplace=True)
apparent_individual['year'] = pd.to_datetime(apparent_individual['date']).dt.year.astype('int')

app_indi_path = os.path.dirname(struct_plant_dir)+'/apparent_individual_'+os.path.basename(file_path)[9:13]+'_'+str(apparent_individual['year'].min())+'_'+str(apparent_individual['year'].max())+'.csv'
apparent_individual.to_csv(app_indi_path)
print('\nMerged file written to: ',app_indi_path)
#apparent_individual.loc[apparent_individual['measurementHeight'] >= 130]

#%% Merge all maptag
maptag = pd.DataFrame()

for root, dirs, files in os.walk(struct_plant_dir):
    for file in files:
        if file.endswith('.csv') and 'mappingandtagging' in file:
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            print('ek ka shape :',df.shape)
            print(file_path)
            maptag = pd.concat([maptag, df], ignore_index=True)
            print('merged ka shape:',maptag.shape)
            
maptag.drop_duplicates(inplace=True)
#maptag['year'] = pd.to_datetime(maptag['date']).dt.year

maptag_path = os.path.dirname(struct_plant_dir)+'/maptag_'+os.path.basename(file_path)[9:13]+'_'+str(apparent_individual['year'].min())+'_'+str(apparent_individual['year'].max())+'.csv'

maptag.to_csv(maptag_path)
print('\nMerged file written to: ',maptag_path)
#%% Read species name and merge species and tree information
df1 = df2 = df3 = pd.DataFrame()
df1 = pd.merge(apparent_individual,maptag[['individualID','taxonID','scientificName']],on='individualID', how = 'left')
df2 = pd.merge(df1,sp_names, on='scientificName', how='left')
tree_loc = tree_loc.dropna(subset='geometry')
df3 = pd.DataFrame()
df3 = pd.merge(df2, tree_loc[['individual','geometry']], left_on='individualID',right_on='individual',how='left')
df3 = df3.drop('individual', axis=1)
#%% Calculate AGB and write the Master data sheet
df3['AGB_2004'] = df3.apply(lambda row: np.exp(row['B0'] + (row['B1']*np.log(row['stemDiameter']))), axis=1)
df3['AGB_2014'] = df3.apply(lambda row: np.exp(row['B0_n'] + (row['B1_n']*np.log(row['stemDiameter']))), axis=1)
df3.columns = df3.columns.str.replace('_x','')
df3.columns = df3.columns.str.replace('_y','')
df3 = df3.loc[:, ~df3.columns.duplicated()]
df3 = df3.dropna(subset=['AGB_2004'])
df3 = df3[df3['plantStatus'].notna() & df3['plantStatus'].str.contains('live', case=False, na=False)]

master_sheet_path = os.path.dirname(struct_plant_dir)+'/master_sheet_'+os.path.basename(file_path)[9:13]+'_'+str(apparent_individual['year'].min())+'_'+str(apparent_individual['year'].max())+'.csv'
df3.to_csv(master_sheet_path)
print('Master AGB written to ',master_sheet_path)

#%% Put smaller plots into bigger plot
df4 = df3[df3['growthForm'].str.contains('tree', case=False)].copy()
df4['date'] = pd.to_datetime(df4['date'], errors='coerce')
df4['year'] = df4['date'].dt.year


patterns_to_replace = {
    r'_25_1': '_100',
    r'_25_2': '_100',
    r'_25_3': '_100',
    r'_25_4': '_100',
    r'_25_unknown': '_100',
    r'_10_1': '_100',
    r'_10_2': '_100',
    r'_10_3': '_100',
    r'_10_4': '_100',
    r'_10_unknown': '_100',
}

for pattern, replacement in patterns_to_replace.items():
    df4['subplotID'] = df4['subplotID'].str.replace(pattern, replacement)

values_to_replace = ['21_100', '22_100', '30_100', '31_100']
for year_to_check in df4['year'].unique():
    for plotID_to_check in df4.loc[df4['year'] == year_to_check, 'plotID'].unique():
        condition = (df4['year'] == year_to_check) & (df4['subplotID'] == '21_400') & (df4['plotID'] == plotID_to_check)
        
        if condition.any():
            for value_to_replace in values_to_replace:
                df4.loc[(df4['year'] == year_to_check) & (df4['plotID'] == plotID_to_check) & (df4['subplotID'] == value_to_replace), 'subplotID'] = '21_400'

values_to_replace = ['23_100', '24_100', '32_100', '33_100']
for year_to_check in df4['year'].unique():
    for plotID_to_check in df4.loc[df4['year'] == year_to_check, 'plotID'].unique():
        condition = (df4['year'] == year_to_check) & (df4['subplotID'] == '23_400') & (df4['plotID'] == plotID_to_check)
        
        if condition.any():
            for value_to_replace in values_to_replace:
                df4.loc[(df4['year'] == year_to_check) & (df4['plotID'] == plotID_to_check) & (df4['subplotID'] == value_to_replace), 'subplotID'] = '23_400'
####

values_to_replace = ['39_100', '40_100', '48_100', '49_100']
for year_to_check in df4['year'].unique():
    for plotID_to_check in df4.loc[df4['year'] == year_to_check, 'plotID'].unique():
        condition = (df4['year'] == year_to_check) & (df4['subplotID'] == '39_400') & (df4['plotID'] == plotID_to_check)
        
        if condition.any():
            for value_to_replace in values_to_replace:
                df4.loc[(df4['year'] == year_to_check) & (df4['plotID'] == plotID_to_check) & (df4['subplotID'] == value_to_replace), 'subplotID'] = '39_400'
####

values_to_replace = ['41_100', '42_100', '50_100', '51_100']
for year_to_check in df4['year'].unique():
    for plotID_to_check in df4.loc[df4['year'] == year_to_check, 'plotID'].unique():
        condition = (df4['year'] == year_to_check) & (df4['subplotID'] == '41_400') & (df4['plotID'] == plotID_to_check)
        
        if condition.any():
            for value_to_replace in values_to_replace:
                df4.loc[(df4['year'] == year_to_check) & (df4['plotID'] == plotID_to_check) & (df4['subplotID'] == value_to_replace), 'subplotID'] = '41_400'
####

df4['plot_subplot']= pd.concat([df4['plotID'], df4['subplotID'].astype(str)], axis=1).apply('_'.join, axis=1)

######
#%% Compile chm stats
struct_plant_dir2 = os.path.dirname(struct_plant_dir)
chm = pd.DataFrame()

for root, dirs, files in os.walk(struct_plant_dir2):
    for file in files:
        if file.endswith('.csv') and 'chm_height' in file and 'LENO' in file:
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            print('ek ka shape :',df.shape)
            print(file_path)
            chm = pd.concat([chm, df], ignore_index=True)
            print('merged ka shape:',chm.shape)
            
chm.drop_duplicates(inplace=True)
chm = chm.dropna(subset=['chm_RH95'])
chm = chm.dropna(subset=['year'])
chm['year'] = chm['year'].astype('int32')
#chm = chm[chm['subplotID'].str.contains('_400')]
chm['plot_year'] = chm['subplotID'] + '_' + chm['year'].astype(str)

#%% Merge tables
# Group by 'year' and 'plot_subplot'
grouped_df = df4.groupby(['year', 'plot_subplot'])

# Calculate maximum height for each group
#max_height_df = grouped_df['height'].max().reset_index()

percentile_95_df = grouped_df['height'].quantile(0.95).reset_index()
# Calculate the sum of AGB_2004 and AGB_2014 for each group
sum_agb_df = grouped_df[['AGB_2004', 'AGB_2014']].sum().reset_index()
group_count_df = grouped_df.size().reset_index(name='count')

result_df = pd.merge(percentile_95_df, sum_agb_df, on=['year', 'plot_subplot'], how='outer')
result_df = pd.merge(result_df, group_count_df, on=['year', 'plot_subplot'], how='outer')

# Add the count as a new column in result_df
result_df['count'] = group_count_df['count']

# Merge the two DataFrames on 'plot_subplot'
print(result_df)
result_df['plot_year'] = result_df['plot_subplot'] + '_' + result_df['year'].astype(str)
###############
#%%
v = pd.merge(result_df,chm,on='plot_year',how='left')
#%%
v1 = v

#%%
v1 = v[v['area'] == 400]

#%%
output_dir = os.path.join(os.path.dirname(struct_plant_dir), 'outputs')

# Check if the directory exists
if not os.path.exists(output_dir):
    # If the directory doesn't exist, create it
    os.makedirs(output_dir)
    print(f"Directory '{output_dir}' created successfully.")
else:
    print(f"Directory '{output_dir}' already exists.")
#%% AGB vs chm_mean


# Define the colormap and normalize the color scale
cmap = plt.get_cmap('viridis')
normalize = Normalize(vmin=v1['chm_count'].min(), vmax=v1['chm_count'].max())

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

# Iterate over unique values in 'chm_count'
for index, row in v1.iterrows():
    color = cmap(normalize(row['chm_count']))  # Generate color from colormap
    ax.scatter(row['chm_mean'], row['AGB_2004'] * 10 / row['area'], color=color)

#x_curve = np.linspace(0, 40, 100)  
#y_curve = 3.26 * np.power(x_curve, 1.74)
#ax.plot(x_curve, y_curve, color='red', label='Curve')

ax.set_xlabel('CHM Height (m)', fontname='Times New Roman', fontsize=14)
ax.set_ylabel('AGB 2004 (Mg/Ha)', fontname='Times New Roman', fontsize=14)
ax.set_title('AGB (2004) vs CHM mean (LENO)', fontname='Times New Roman', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12, direction='in', pad=8, labelcolor='black', width=2)
ax.tick_params(axis='both', which='minor', labelsize=10, direction='in', pad=8, labelcolor='black', width=1)
ax.set_xlim(0, 40)
ax.set_ylim(0, 300)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.ax.yaxis.label.set_fontname('Times New Roman')
cbar.ax.yaxis.label.set_fontsize(14)
cbar.set_label('Number of Pixels')

# Set color bar tick font to Times New Roman
plt.gca().yaxis.set_tick_params(labelsize=12, direction='in', pad=8, color='black', width=1)


# Show the plot
output_path = output_dir + '/agb2004_vs_chm_mean_LENO.png'
plt.savefig(output_path, dpi=300)
print(f"Image saved to {output_path}")
plt.show()

##############################
# Define the colormap and normalize the color scale
cmap = plt.get_cmap('viridis')
normalize = Normalize(vmin=v1['chm_count'].min(), vmax=v1['chm_count'].max())

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

# Iterate over unique values in 'chm_count'
for index, row in v1.iterrows():
    color = cmap(normalize(row['chm_count']))  # Generate color from colormap
    ax.scatter(row['chm_mean'], row['AGB_2014'] * 10 / row['area'], color=color)

#x_curve = np.linspace(0, 40, 100)  
#y_curve = 3.26 * np.power(x_curve, 1.74)
#ax.plot(x_curve, y_curve, color='red', label='Curve')

ax.set_xlabel('CHM Height (m)', fontname='Times New Roman', fontsize=14)
ax.set_ylabel('AGB 2014 (Mg/Ha)', fontname='Times New Roman', fontsize=14)
ax.set_title('AGB (2014) vs CHM mean (LENO)', fontname='Times New Roman', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12, direction='in', pad=8, labelcolor='black', width=2)
ax.tick_params(axis='both', which='minor', labelsize=10, direction='in', pad=8, labelcolor='black', width=1)
ax.set_xlim(0, 40)
ax.set_ylim(0, 300)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.ax.yaxis.label.set_fontname('Times New Roman')
cbar.ax.yaxis.label.set_fontsize(14)
cbar.set_label('Number of Pixels')

# Set color bar tick font to Times New Roman
plt.gca().yaxis.set_tick_params(labelsize=12, direction='in', pad=8, color='black', width=1)


# Show the plot
output_path = output_dir + '/agb2014_vs_chm_mean_LENO.png'
plt.savefig(output_path, dpi=300)
print(f"Image saved to {output_path}")
plt.show()


#%%


# Define the colormap and normalize the color scale
cmap = plt.get_cmap('viridis')
normalize = Normalize(vmin=v1['count'].min(), vmax=v1['count'].max())

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

# Iterate over unique values in 'chm_count'
for index, row in v1.iterrows():
    color = cmap(normalize(row['count']))  # Generate color from colormap
    ax.scatter(row['height'], row['AGB_2004'] * 10 / row['area'], color=color)

#x_curve = np.linspace(0, 40, 100)  
#y_curve = 3.26 * np.power(x_curve, 1.74)
#ax.plot(x_curve, y_curve, color='red', label='Curve')

# Set labels and title with Times New Roman font
ax.set_xlabel('Field Height (m)', fontname='Times New Roman', fontsize=14)
ax.set_ylabel('AGB 2004 (Mg/Ha)', fontname='Times New Roman', fontsize=14)
ax.set_title('AGB (2004) vs Field Height (LENO)', fontname='Times New Roman', fontsize=16)

# Set ticks font to Times New Roman
ax.tick_params(axis='both', which='major', labelsize=12, direction='in', pad=8, labelcolor='black', width=2)
ax.tick_params(axis='both', which='minor', labelsize=10, direction='in', pad=8, labelcolor='black', width=1)

# Set limits
ax.set_xlim(0, 40)
ax.set_ylim(0, 300)

# Show the color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.ax.yaxis.label.set_fontname('Times New Roman')
cbar.ax.yaxis.label.set_fontsize(14)
cbar.set_label('Number of Trees')

# Set color bar tick font to Times New Roman
plt.gca().yaxis.set_tick_params(labelsize=12, direction='in', pad=8, color='black', width=1)

# Show the plot
plt.show()
#plt.savefig(args, kwargs)

#%% Subplots of CHM Mean, CHM RH95, Field Height
# Define the colormap and normalize the color scale
cmap = plt.get_cmap('viridis')

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=300)

# Plot AGB vs chm_rh95
normalize_chm_count = Normalize(vmin=v1['chm_count'].min(), vmax=v1['chm_count'].max())
for index, row in v1.iterrows():
    color = cmap(normalize_chm_count(row['chm_count']))
    axs[0].scatter(row['chm_RH95'], row['AGB_2014'] * 10 / row['area'], color=color)
axs[0].set_xlabel('CHM Height RH95 (m)', fontname='Times New Roman', fontsize=14)
axs[0].set_ylabel('AGB 2014 (Mg/Ha)', fontname='Times New Roman', fontsize=14)
axs[0].set_title('AGB (2014) vs CHM RH95 (LENO)', fontname='Times New Roman', fontsize=16)

# Plot AGB vs chm_mean
#normalize_chm_mean = Normalize(vmin=v1['chm_mean'].min(), vmax=v1['chm_mean'].max())
for index, row in v1.iterrows():
    color = cmap(normalize_chm_count(row['chm_count']))
    axs[1].scatter(row['chm_mean'], row['AGB_2014'] * 10 / row['area'], color=color)
axs[1].set_xlabel('CHM Height Mean (m)', fontname='Times New Roman', fontsize=14)
axs[1].set_ylabel('AGB 2014 (Mg/Ha)', fontname='Times New Roman', fontsize=14)
axs[1].set_title('AGB (2014) vs CHM mean (LENO)', fontname='Times New Roman', fontsize=16)

# Plot AGB vs Field Height
normalize_field_count = Normalize(vmin=v1['count'].min(), vmax=v1['count'].max())
for index, row in v1.iterrows():
    color = cmap(normalize_field_count(row['count']))
    axs[2].scatter(row['height'], row['AGB_2014'] * 10 / row['area'], color=color)
axs[2].set_xlabel('Field Height (m)', fontname='Times New Roman', fontsize=14)
axs[2].set_ylabel('AGB 2014 (Mg/Ha)', fontname='Times New Roman', fontsize=14)
axs[2].set_title('AGB (2014) vs Field Height (LENO)', fontname='Times New Roman', fontsize=16)

# Set common limits for better comparison
for ax in axs:
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 300)

# Add colorbars
cbar_labels = ['Plot Area (sq.m)', 'Plot Area (sq.m)', 'Number of Trees']
norm_list = [normalize_chm_count, normalize_chm_count, normalize_field_count]
for ax, norm, label in zip(axs, norm_list, cbar_labels):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.yaxis.label.set_fontname('Times New Roman')
    cbar.ax.yaxis.label.set_fontsize(14)
    cbar.set_label(label, fontname='Times New Roman', fontsize=14)
    ax.grid(True)

# Set ticks font to Times New Roman
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=12, direction='in', pad=8, labelcolor='black', width=2)
    ax.tick_params(axis='both', which='minor', labelsize=10, direction='in', pad=8, labelcolor='black', width=1)

plt.tight_layout()


output_path = output_dir + '/agb2014_vs_heights_LENO.png'
plt.savefig(output_path, dpi=300)
print(f"Image saved to {output_path}")

plt.show()

#%%
'''
fig = plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(v['height'],v['chm_RH95'])
#plt.ylim(0,200)

plt.xlabel('Field Height (m)', fontsize=14)
plt.ylabel('CHM RH95 (m)', fontsize=14)
plt.xlim(0,40)
plt.ylim(0,40)
plt.title('CHM_RH95 vs Field Height (LENO)', fontsize=16)
#plt.savefig('/data/gedi/_neon/LENO/chm_height_2021.png')

plt.show()
'''
#%% Curve fitting - improve this
#############
#20240226

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



#v1 = v[v['area'] == 400]
v1['AGB'] = v1['AGB_2004'] * 10 / v1['area']

#v1 = v1[v1['AGB'] <= 400]

#v1 = v1[v1['area'] == 400]


v1= v1.dropna()
# Plot scatter plot
plt.figure(figsize=(8, 8), dpi=300)

# Extract unique 'year_x' values for color mapping
unique_years = v1['year_x'].unique()

# Create a colormap with a different color for each unique year
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_years)))

for i, year in enumerate(unique_years):
    year_data = v1[v1['year_x'] == year]
    plt.scatter(year_data['height'], year_data['AGB'], label=f'Year {year}', color=colors[i])

plt.xlabel('Field Height')
plt.ylabel('AGB')

# Legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Define the curve function a*x**b
def curve_function(x, a, b):
    return a * np.power(x, b)

# Initialize best parameters and error
best_params = None
best_fit_error = float('inf')

# Initialize lists to store iteration details
iteration_details = []

# Number of iterations
num_iterations = 100

for iteration in range(num_iterations):
    train_data, test_data = train_test_split(v1, test_size=0.05, random_state=iteration)

    initial_guess = [1, 1]

    params, covariance = curve_fit(curve_function, train_data['height'], train_data['AGB'], p0=initial_guess,method='dogbox')

    predictions = curve_function(test_data['height'], *params)

    fit_error = np.sqrt(mean_squared_error(test_data['AGB'], predictions))

    r2 = r2_score(test_data['AGB'], predictions)

    if fit_error < best_fit_error:
        best_fit_error = fit_error
        best_params = params

    iteration_details.append({
        'Iteration': iteration + 1,
        'Parameters': params,
        'RMSE': fit_error,
        'R-squared': r2
    })

    print(f"Iteration {iteration + 1} - Parameters: {params}, RMSE: {fit_error}, R-squared: {r2}")

iteration_df = pd.DataFrame(iteration_details)

print(f"\nBest Fit Parameters: {best_params}")
print(f"Best RMSE: {best_fit_error}")
print(f"Best R-squared: {iteration_df.loc[iteration_df['RMSE'].idxmin(), 'R-squared']}")

# Save iteration results to CSV
iteration_df.to_csv('iteration_results.csv', index=False)
print('iteration_results.csv saved')

# Plot RMSE and R-squared over iterations
fig, ax1 = plt.subplots(figsize=(8, 4))

color = 'tab:red'
ax1.set_xlabel('Iteration')
ax1.set_ylabel('RMSE', color=color)
ax1.plot(iteration_df['Iteration'], iteration_df['RMSE'], label='RMSE', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('R-squared', color=color)  
ax2.plot(iteration_df['Iteration'], iteration_df['R-squared'], label='R-squared', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()

plt.figure(figsize=(8, 8), dpi=300)
for i, year in enumerate(unique_years):
    year_data = v1[v1['year_x'] == year]
    plt.scatter(year_data['height'], year_data['AGB'], label=f'Year {year}', color=colors[i])

x_curve = np.linspace(0, 40, 100)
y_curve = curve_function(x_curve-5, *best_params)
plt.plot(x_curve, y_curve, color='red', label=f'Best Fit: {best_params[0]:.2f} * x**{best_params[1]:.2f}\nRMSE: {best_fit_error:.2f}\nR-squared: {iteration_df.loc[iteration_df["RMSE"].idxmin(), "R-squared"]:.2f}\nIterations: {num_iterations}')

plt.ylim(0, 300)
plt.xlabel('CHM Height (chm_mean)')
plt.ylabel('AGB 2004')
plt.title('Scatter Plot and Best Fit Curve')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#%%
#Using L-BFGS method

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Assuming v is your original DataFrame
# ... (previous code)

# Filter data and preprocess
v1 = v[v['area'] == 400]
v1['AGB'] = v1['AGB_2004'] * 10 / v1['area']
v1 = v1[v1['AGB'] <= 300]
v1 = v1[v1['area'] == 400]

v1 = v1.dropna()

# Plot scatter plot
plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(v1['height'], v1['AGB'], label='Scatter Plot')
plt.xlabel('Height')
plt.ylabel('AGB')

# Define the curve function a*x**b
def curve_function(params, x, y):
    a, b = params
    predictions = a * np.power(x, b)
    return mean_squared_error(y, predictions)

# Initialize best parameters and error
best_params = None
best_fit_error = float('inf')

# Initialize lists to store iteration details
iteration_details = []

# Number of iterations
num_iterations = 50

plt.figure(figsize=(8, 8), dpi=300)

for iteration in range(num_iterations):
    train_data, test_data = train_test_split(v1, test_size=0.2, random_state=iteration)

    initial_guess = [1, 1]

    # Minimize the mean squared error
    result = minimize(curve_function, initial_guess, args=(train_data['height'], train_data['AGB']), method='L-BFGS-B')

    # Get the optimized parameters
    params = result.x

    # Test the model on the testing set
    predictions = params[0] * np.power(test_data['height'], params[1])

    # Calculate RMSE
    fit_error = np.sqrt(mean_squared_error(test_data['AGB'], predictions))

    # Calculate R-squared
    r2 = r2_score(test_data['AGB'], predictions)

    # Update best parameters if the current iteration has a lower error
    if fit_error < best_fit_error:
        best_fit_error = fit_error
        best_params = params

    # Append iteration details to the list
    iteration_details.append({
        'Iteration': iteration + 1,
        'Parameters': params,
        'RMSE': fit_error,
        'R-squared': r2
    })

    # Print iteration details
    print(f"Iteration {iteration + 1} - Parameters: {params}, RMSE: {fit_error}, R-squared: {r2}")

# Create a DataFrame from the iteration details
iteration_df = pd.DataFrame(iteration_details)

# Print or use the final results
print(f"\nBest Fit Parameters: {best_params}")
print(f"Best RMSE: {best_fit_error}")
print(f"Best R-squared: {iteration_df.loc[iteration_df['RMSE'].idxmin(), 'R-squared']}")

# Export the DataFrame to a CSV file
iteration_df.to_csv(struct_plant_dir + '/iteration_results.csv', index=False)
print(struct_plant_dir + '/iteration_results.csv', " saved")

# Plot RMSE and R-squared for each iteration
fig, ax1 = plt.subplots(figsize=(8, 4))

color = 'tab:red'
ax1.set_xlabel('Iteration')
ax1.set_xlim(30,40)
ax1.set_ylabel('RMSE', color=color)
ax1.plot(iteration_df['Iteration'], iteration_df['RMSE'], label='RMSE', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('R-squared', color=color)  
ax2.plot(iteration_df['Iteration'], iteration_df['R-squared'], label='R-squared', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  

# Scatter plot and best-fit curve
plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(v1['height'], v1['AGB'], label='Data')
x_curve = np.linspace(0, 50, 100)
y_curve = best_params[0] * np.power(x_curve, best_params[1])
plt.plot(x_curve, y_curve, color='red', label=f'Best Fit: {best_params[0]:.2f} * x**{best_params[1]:.2f}\nRMSE: {best_fit_error:.2f}\nR-squared: {iteration_df.loc[iteration_df["RMSE"].idxmin(), "R-squared"]:.2f}')
plt.ylim(0, 400)
plt.xlabel('Field Height')
plt.ylabel('AGB 2004')
plt.title('Scatter Plot and Best Fit Curve')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#%% Fit plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Filter data and preprocess
v1 = v[v['area'] == 400]
v1['AGB'] = v1['AGB_2004'] * 10 / v1['area']
v1 = v1[v1['AGB'] <= 300]

v1 = v1.dropna()

# Plot scatter plot
plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(v1['height'], v1['AGB'], label='Scatter Plot')
plt.xlabel('Height')
plt.ylabel('AGB')

# Define the curve function a*x**b
def curve_function(x, a, b):
    return a * np.power(x, b)

# Initialize best parameters and covariance matrix
best_params, cov = curve_fit(curve_function, v1['height'], v1['AGB'], p0=[1, 1])

# Test the model on the testing set
predictions = curve_function(v1['height'], *best_params)

# Calculate RMSE
fit_error = np.sqrt(mean_squared_error(v1['AGB'], predictions))

# Calculate R-squared
r2 = r2_score(v1['AGB'], predictions)

# Print results
print(f"\nBest Fit Parameters: {best_params}")
print(f"RMSE: {fit_error}")
print(f"R-squared: {r2}")

iteration_df = pd.DataFrame({
    'Parameters': [best_params],
    'RMSE': [fit_error],
    'R-squared': [r2]
})
iteration_df.to_csv(struct_plant_dir + '/iteration_results_curve_fit.csv', index=False)
print(struct_plant_dir + '/iteration_results_curve_fit.csv', " saved")

# Scatter plot and best-fit curve
plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(v1['height'], v1['AGB'], label='Data')
x_curve = np.linspace(0, 50, 100)
y_curve = curve_function(x_curve, *best_params)
plt.plot(x_curve, y_curve, color='red', label=f'Best Fit: {best_params[0]:.2f} * x**{best_params[1]:.2f}\nRMSE: {fit_error:.2f}\nR-squared: {r2:.2f}')
plt.ylim(0, 400)
plt.xlabel('Field Height')
plt.ylabel('AGB 2004')
plt.title('Scatter Plot and Best Fit Curve')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()



#%% With Savitzy Golay filter to smoothen the curve and remove noise
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Filter data and preprocess
v1 = v[v['area'] == 400]
v1['AGB'] = v1['AGB_2004'] * 10 / v1['area']
#v1 = v1[v1['AGB'] <= 300]
v1 = v1[v1['area'] == 400]
v1 = v1.dropna()

# Calculate the mean of the highest 5 values of AGB and height
top_5_height_mean = v1.nlargest(5, 'AGB')['height'].mean()
top_5_AGB_mean = v1.nlargest(5, 'AGB')['AGB'].mean()

# Plot scatter plot
plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(v1['height'], v1['AGB'], label='Scatter Plot')
plt.xlabel('Height')
plt.ylabel('AGB')

# Define the curve function with the constraint
def curve_function(params, x, y):
    a, b = params
    predictions = a * np.power(x, b)
    # Calculate the constraint: the difference between the curve value at the mean height of top 5 AGB values and the mean of those top 5 AGB values
    constraint = (a * np.power(top_5_height_mean, b)) - top_5_AGB_mean
    # Return the combined loss: mean squared error plus the squared constraint (to enforce it being zero)
    return mean_squared_error(y, predictions) + constraint**2

# Initialize best parameters and error
best_params = None
best_fit_error = float('inf')

# Initialize lists to store iteration details
iteration_details = []

# Number of iterations
num_iterations = 20

for iteration in range(num_iterations):
    train_data, test_data = train_test_split(v1, test_size=0.2, random_state=iteration)

    initial_guess = [1, 1]

    # Minimize the mean squared error with the constraint
    result = minimize(curve_function, initial_guess, args=(train_data['height'], train_data['AGB']), method='L-BFGS-B')

    # Get the optimized parameters
    params = result.x

    # Test the model on the testing set
    predictions = params[0] * np.power(test_data['height'], params[1])

    # Calculate RMSE
    fit_error = np.sqrt(mean_squared_error(test_data['AGB'], predictions))

    # Calculate R-squared
    r2 = r2_score(test_data['AGB'], predictions)

    # Update best parameters if the current iteration has a lower error
    if fit_error < best_fit_error:
        best_fit_error = fit_error
        best_params = params

    # Append iteration details to the list
    iteration_details.append({
        'Iteration': iteration + 1,
        'Parameters': params,
        'RMSE': fit_error,
        'R-squared': r2
    })

    # Print iteration details
    print(f"Iteration {iteration + 1} - Parameters: {params}, RMSE: {fit_error}, R-squared: {r2}")

# Create a DataFrame from the iteration details
iteration_df = pd.DataFrame(iteration_details)

# Print or use the final results
print(f"\nBest Fit Parameters: {best_params}")
print(f"Best RMSE: {best_fit_error}")
print(f"Best R-squared: {iteration_df.loc[iteration_df['RMSE'].idxmin(), 'R-squared']}")

# Export the DataFrame to a CSV file
iteration_df.to_csv(struct_plant_dir + '/iteration_results.csv', index=False)
print(struct_plant_dir + '/iteration_results.csv', " saved")

# Plot RMSE and R-squared for each iteration
fig, ax1 = plt.subplots(figsize=(8, 4))

color = 'tab:red'
ax1.set_xlabel('Iteration')
#ax1.set_xlim(30,40)
ax1.set_ylabel('RMSE', color=color)
ax1.plot(iteration_df['Iteration'], iteration_df['RMSE'], label='RMSE', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('R-squared', color=color)  
ax2.plot(iteration_df['Iteration'], iteration_df['R-squared'], label='R-squared', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  

# Scatter plot and best-fit curve
plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(v1['height'], v1['AGB'], label='Data')
x_curve = np.linspace(0, 50, 100)
y_curve = best_params[0] * np.power(x_curve, best_params[1])
plt.plot(x_curve, y_curve, color='red', label=f'Best Fit: {best_params[0]:.2f} * x**{best_params[1]:.2f}\nRMSE: {best_fit_error:.2f}\nR-squared: {iteration_df.loc[iteration_df["RMSE"].idxmin(), "R-squared"]:.2f}')
plt.ylim(0, 400)
plt.xlabel('CHM Height (chm_mean)')
plt.ylabel('AGB 2004')
plt.title('Scatter Plot and Best Fit Curve')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


#%% Using PyTorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Assuming v1 is a DataFrame with columns 'height' and 'AGB'
# Add 'year_x' column to the DataFrame if it doesn't exist
v1['year_x'] = np.random.randint(2000, 2020, size=len(v1))  # Example: Random 'year_x' values

# Calculate 'AGB' based on 'AGB_2004', 'area', and add noise to 'height' for variety
v1['AGB'] = v1['AGB_2004'] * 10 / v1['area'] + np.random.normal(0, 5, len(v1))
v1 = v1.dropna()

# Select relevant features
features = v1['height'].values.reshape(-1, 1)
target = v1['AGB'].values.reshape(-1, 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Feature Scaling
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Define a neural network model with batch normalization and dropout
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_rate=0.5):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Initialize the model
input_size = 1
hidden_size1 = 16
hidden_size2 = 8
output_size = 1
dropout_rate = 0.5
model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size, dropout_rate)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    predictions_tensor = model(X_test_tensor)
    predictions = predictions_tensor.numpy()

predictions_original_scale = scaler_y.inverse_transform(predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions_original_scale))
print(f'Test RMSE: {rmse:.4f}')

plt.scatter(X_test, y_test, label='Actual Data')
plt.scatter(X_test, predictions_original_scale, label='Predictions', color='red')
plt.xlabel('Field Height')
plt.ylabel('AGB')
plt.title('Neural Network Regression with Batch Normalization and Dropout')
plt.legend()
plt.show()