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

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


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
        condition = (df4['year'] == year_to_check) & (df4['subplotID'] == '41_400') & (df4['plotID'] == plotID_to_check)
        
        if condition.any():
            for value_to_replace in values_to_replace:
                df4.loc[(df4['year'] == year_to_check) & (df4['plotID'] == plotID_to_check) & (df4['subplotID'] == value_to_replace), 'subplotID'] = '21_400'

values_to_replace = ['23_100', '24_100', '32_100', '33_100']
for year_to_check in df4['year'].unique():
    for plotID_to_check in df4.loc[df4['year'] == year_to_check, 'plotID'].unique():
        condition = (df4['year'] == year_to_check) & (df4['subplotID'] == '41_400') & (df4['plotID'] == plotID_to_check)
        
        if condition.any():
            for value_to_replace in values_to_replace:
                df4.loc[(df4['year'] == year_to_check) & (df4['plotID'] == plotID_to_check) & (df4['subplotID'] == value_to_replace), 'subplotID'] = '23_400'
####

values_to_replace = ['39_100', '40_100', '48_100', '49_100']
for year_to_check in df4['year'].unique():
    for plotID_to_check in df4.loc[df4['year'] == year_to_check, 'plotID'].unique():
        condition = (df4['year'] == year_to_check) & (df4['subplotID'] == '41_400') & (df4['plotID'] == plotID_to_check)
        
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
grouped_df = df4.groupby(['year', 'plot_subplot'])

#max_height_df = grouped_df['height'].max().reset_index()

percentile_95_df = grouped_df['height'].quantile(0.95).reset_index()
sum_agb_df = grouped_df[['AGB_2004', 'AGB_2014']].sum().reset_index()
group_count_df = grouped_df.size().reset_index(name='count')

result_df = pd.merge(percentile_95_df, sum_agb_df, on=['year', 'plot_subplot'], how='outer')
result_df = pd.merge(result_df, group_count_df, on=['year', 'plot_subplot'], how='outer')

result_df['count'] = group_count_df['count']

print(result_df)
result_df['plot_year'] = result_df['plot_subplot'] + '_' + result_df['year'].astype(str)

#%%
v = pd.merge(result_df,chm,on='plot_year',how='left')
#%%
v1 = v

#%% Filter only plots with area 400 sq m
#v1 = v[v['area'] == 400]

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
cmap = plt.get_cmap('viridis')
normalize = Normalize(vmin=v1['chm_count'].min(), vmax=v1['chm_count'].max())

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

for index, row in v1.iterrows():
    color = cmap(normalize(row['chm_count']))  
    ax.scatter(row['chm_mean'], row['AGB_2004'] * 10 / row['area'], color=color)

x_curve = np.linspace(0, 40, 100)  
y_curve = 3.26 * np.power(x_curve, 1.34)
ax.plot(x_curve, y_curve, color='red', label='Curve')

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

x_curve = np.linspace(0, 40, 100)  
y_curve = 3.26 * np.power(x_curve, 1.34)
ax.plot(x_curve, y_curve, color='red', label='Curve')

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


x_curve = np.linspace(0, 40, 100)  
#y_curve = 3.26 * np.power(x_curve, 1.34)

y_curve = 3.26 * np.exp(x_curve/6)
ax.plot(x_curve, y_curve, color='red', label='Curve')


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
cbar_labels = ['CHM Count', 'CHM Count', 'Count']
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
#%%
#############



#v1 = v[v['area'] == 400]
v1['AGB'] = v1['AGB_2004'] * 10 / v1['area']

#v1 = v1[v1['AGB'] <= 200]

#v1 = v1[v1['area'] == 400]


v1= v1.dropna()
# Plot scatter plot
plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(v1['height'], v1['AGB'], label='Scatter Plot')
plt.xlabel('CHM_mean')
plt.ylabel('AGB')


# Define the curve function a*x**b
def curve_function(x, a, b):
    return a * np.power(x, b)

# Initialize best parameters and error
best_params = None
best_fit_error = float('inf')

# Initialize lists to store iteration details
iteration_details = []

# Number of iterations
num_iterations = 15

plt.figure(figsize=(8, 8), dpi=300)

for iteration in range(num_iterations):
    train_data, test_data = train_test_split(v1, test_size=0.3, random_state=iteration)

    initial_guess = [3, 1]

    params, covariance = curve_fit(curve_function, train_data['height'], train_data['AGB'], p0=initial_guess)

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

iteration_df.to_csv(struct_plant_dir + '/iteration_results.csv', index=False)
print(struct_plant_dir + '/iteration_results.csv', " saved")

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

# Scatter plot and best-fit curve
plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(v1['height'], v1['AGB'], label='Data')
x_curve = np.linspace(0, 50, 100)
y_curve = curve_function(x_curve, *best_params)
plt.plot(x_curve, y_curve, color='red', label=f'Best Fit: {best_params[0]:.2f} * x**{best_params[1]:.2f}\nRMSE: {best_fit_error:.2f}\nR-squared: {iteration_df.loc[iteration_df["RMSE"].idxmin(), "R-squared"]:.2f}')
plt.ylim(0, 500)
plt.xlabel('CHM Height (chm_mean)')
plt.ylabel('AGB 2004')
plt.title('Scatter Plot and Best Fit Curve')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#%% Ye woh trial wala code hai

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit

# Assuming v1 is your DataFrame containing the data

# Drop NaN values before splitting
v1 = v1.dropna()

# Plot scatter plot
plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(v1['height'], v1['AGB'], label='Scatter Plot')
plt.xlabel('CHM_mean')
plt.ylabel('AGB')

# Define the curve function a*x**b
def curve_function(x, a, b, c):
    return ((x-a)**b)*np.exp(x/c)

# Initialize best parameters and error
best_params = None
best_fit_error = float('inf')

# Initialize lists to store iteration details
iteration_details = []

# Number of iterations
num_iterations = 15

plt.figure(figsize=(8, 8), dpi=300)

for iteration in range(num_iterations):
    # Drop NaN values before splitting
    v1 = v1.dropna()
    train_data, test_data = train_test_split(v1, test_size=0.3, random_state=iteration)

    initial_guess = [3, 1, 5]

    params, covariance = curve_fit(curve_function, train_data['height'], train_data['AGB'], p0=initial_guess)

    predictions = curve_function(test_data['height'], *params)

    # Filter out NaN values from test_data['AGB'] and predictions arrays
    non_nan_indices = ~np.isnan(test_data['AGB']) & ~np.isnan(predictions)
    test_agb = test_data['AGB'][non_nan_indices]
    pred_agb = predictions[non_nan_indices]

    fit_error = np.sqrt(mean_squared_error(test_agb, pred_agb))

    r2 = r2_score(test_agb, pred_agb)

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

iteration_df.to_csv(struct_plant_dir + '/iteration_results.csv', index=False)
print(struct_plant_dir + '/iteration_results.csv', " saved")

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

# Scatter plot and best-fit curve
plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(v1['height'], v1['AGB'], label='Data')
x_curve = np.linspace(0, 50, 100)
y_curve = curve_function(x_curve-best_params[2], *best_params)
plt.plot(x_curve, y_curve, color='red', label=f'Best Fit: {best_params[0]:.2f} * x**{best_params[1]:.2f}\nRMSE: {best_fit_error:.2f}\nR-squared: {iteration_df.loc[iteration_df["RMSE"].idxmin(), "R-squared"]:.2f}')
plt.ylim(0, 500)
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


#v1 = v[v['area'] == 400]
v1['AGB'] = v1['AGB_2004'] * 10 / v1['area']
#v1 = v1[v1['AGB'] <= 300]
#v1 = v1[v1['area'] == 400]

v1 = v1.dropna()

plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(v1['height'], v1['AGB'], label='Scatter Plot')
plt.xlabel('Height')
plt.ylabel('AGB')

def curve_function(params, x, y):
    a, b = params
    predictions = a * np.power(x, b)
    return mean_squared_error(y, predictions)

best_params = None
best_fit_error = float('inf')

iteration_details = []

num_iterations = 50

plt.figure(figsize=(8, 8), dpi=300)

for iteration in range(num_iterations):
    train_data, test_data = train_test_split(v1, test_size=0.2, random_state=iteration)

    initial_guess = [1, 1]

    result = minimize(curve_function, initial_guess, args=(train_data['height'], train_data['AGB']), method='L-BFGS-B')

    params = result.x

    predictions = params[0] * np.power(test_data['height'], params[1])

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

iteration_df.to_csv(struct_plant_dir + '/iteration_results.csv', index=False)
print(struct_plant_dir + '/iteration_results.csv', " saved")

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

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Assuming v is your original DataFrame
# ... (previous code)

v1 = v[v['area'] == 400]
v1['AGB'] = v1['AGB_2004'] * 10 / v1['area']
v1 = v1[v1['AGB'] <= 300]

v1 = v1.dropna()

plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(v1['height'], v1['AGB'], label='Scatter Plot')
plt.xlabel('Height')
plt.ylabel('AGB')

def curve_function(x, a, b):
    return a * np.power(x, b)

best_params, cov = curve_fit(curve_function, v1['height'], v1['AGB'], p0=[1, 1])

predictions = curve_function(v1['height'], *best_params)

fit_error = np.sqrt(mean_squared_error(v1['AGB'], predictions))

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
'''
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
'''

#%% 