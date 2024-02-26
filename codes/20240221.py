#%%
import pandas as pd
import geopandas as gpd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import curve_fit
#%%
struct_plant_dir = '/data/gedi/_neon/_new/LENO/NEON_struct-plant'
#Directory of NEON plant structure
#%%
#Merge all apparent individuals

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

#%%
#Merge all maptag
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
#%%
sp_names = pd.read_csv('/data/gedi/_neon/Species_Names_Forest_Group.csv')

df1 = df2 = df3 = pd.DataFrame()
df1 = pd.merge(apparent_individual,maptag[['individualID','taxonID','scientificName']],on='individualID', how = 'left')
df2 = pd.merge(df1,sp_names, on='scientificName', how='left')
tree_loc = gpd.read_file('file:///data/gedi/_neon/LENO_2021/tree_locations_20240216_114531.shp')
tree_loc = tree_loc.dropna(subset='geometry')
df3 = pd.DataFrame()
df3 = pd.merge(df2, tree_loc[['individual','geometry']], left_on='individualID',right_on='individual',how='left')
df3 = df3.drop('individual', axis=1)
#%%
df3['AGB_2004'] = df3.apply(lambda row: np.exp(row['B0'] + (row['B1']*np.log(row['stemDiameter']))), axis=1)
df3['AGB_2014'] = df3.apply(lambda row: np.exp(row['B0_n'] + (row['B1_n']*np.log(row['stemDiameter']))), axis=1)
df3.columns = df3.columns.str.replace('_x','')
df3.columns = df3.columns.str.replace('_y','')
df3 = df3.loc[:, ~df3.columns.duplicated()]
df3 = df3.dropna(subset=['AGB_2004'])
#df3 = df3[df3['plantStatus'].str.contains('live',case=False)]
df3 = df3[df3['plantStatus'].notna() & df3['plantStatus'].str.contains('live', case=False, na=False)]

master_sheet_path = os.path.dirname(struct_plant_dir)+'/master_sheet_'+os.path.basename(file_path)[9:13]+'_'+str(apparent_individual['year'].min())+'_'+str(apparent_individual['year'].max())+'.csv'
df3.to_csv(master_sheet_path)
print('Master AGB written to ',master_sheet_path)

#%%
df4 = df3[df3['growthForm'].str.contains('tree', case=False)]
df4['date'] = pd.to_datetime(df4['date'],errors='coerce')
df4['year'] = df4['date'].dt.year


df4.loc[:, 'subplotID'] = df4['subplotID'].str.replace(r'_25_1', '_100')
df4.loc[:, 'subplotID'] = df4['subplotID'].str.replace(r'_25_2', '_100')
df4.loc[:, 'subplotID'] = df4['subplotID'].str.replace(r'_25_3', '_100')
df4.loc[:, 'subplotID'] = df4['subplotID'].str.replace(r'_25_4', '_100')


df4.loc[:, 'subplotID'] = df4['subplotID'].str.replace(r'_10_1', '_100')
df4.loc[:, 'subplotID'] = df4['subplotID'].str.replace(r'_10_2', '_100')
df4.loc[:, 'subplotID'] = df4['subplotID'].str.replace(r'_10_3', '_100')
df4.loc[:, 'subplotID'] = df4['subplotID'].str.replace(r'_10_4', '_100')

#####

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
#%%
struct_plant_dir2 = '/data/gedi/_neon/LENO_2021'
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

#%%
###################
# Group by 'year' and 'plot_subplot'
grouped_df = df4.groupby(['year', 'plot_subplot'])

# Calculate maximum height for each group
max_height_df = grouped_df['height'].max().reset_index()

# Calculate the sum of AGB_2004 and AGB_2014 for each group
sum_agb_df = grouped_df[['AGB_2004', 'AGB_2014']].sum().reset_index()
group_count_df = grouped_df.size().reset_index(name='count')

result_df = pd.merge(max_height_df, sum_agb_df, on=['year', 'plot_subplot'], how='outer')
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


v1 = v[v['area'] == 400]
fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

# Iterate ov1er unique v1alues in 'area'
for area_v1alue in v1['chm_count'].unique():
    subset = v1[v1['chm_count'] == area_v1alue]
    ax.scatter(subset['chm_mean'], subset['AGB_2014'] * 10 / subset['area'], label=f'Count {area_v1alue}')

x_curve = np.linspace(0, 40, 100)  
y_curve = 3.26*np.power(x_curve,1.74)
ax.plot(x_curve, y_curve, color='red', label='Curve')




# Set labels and title
ax.set_xlabel('CHM Height')
ax.set_ylabel('AGB 2004')
ax.set_title('AGB (2004) v1s CHM 95 (LENO 2021)')

# Add legend outside the box
ax.legend(title='Area', bbox_to_anchor=(1.05, 1), loc='upper left')

# Set limits
ax.set_xlim(0, 40)
ax.set_ylim(0, 300)

# Show or sav1e the plot
plt.show()

#%%
fig = plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(v['height'],v['AGB_2004']*10/400)


x_curve = np.linspace(0, 40, 100)  
y_curve = 0.24*np.power(x_curve-10,1.99)
plt.plot(x_curve, y_curve, color='red', label='0.24*height**1.99')

x_curve = np.linspace(0, 40, 100)  
y_curve = 3.26*np.power(x_curve-10,1.74)
plt.plot(x_curve, y_curve, color='purple', label='3.26*height**1.74')


plt.ylim(0,200)
plt.xlabel('Field Height')
plt.ylabel('AGB 2004')
plt.title('AGB (2004) vs Field Height (LENO 2021)')
plt.savefig('/data/gedi/_neon/LENO/height_agb_2021.png')
plt.legend(['Field', '0.24*height**1.99', '3.26*height**1.74'], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#%%
fig = plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(v['height'],v['chm_RH95'])
#plt.ylim(0,200)
plt.xlabel('Field Height')
plt.ylabel('CHM')
plt.xlim(0,40)
plt.ylim(0,40)
plt.title('CHM_RH95 vs Field Height (LENO 2021)')
plt.savefig('/data/gedi/_neon/LENO/chm_height_2021.png')

plt.show()
#%%
fig = plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(v['chm_mean'],v['AGB_2004']*10/v['area'])
#plt.ylim(0,200)

#for i, txt in enumerate(v['subplotID']):
#    plt.annotate(txt, (v['chm_RH95'].iloc[i], v['AGB_2004'].iloc[i]*10/v['area'].iloc[i]), fontsize=8)

plt.xlabel('CHM Mean')
plt.ylabel('AGB 2004')
plt.title('AGB (2004) vs CHM Height (LENO 2021)')
#plt.ylim(0,200)
plt.savefig('/data/gedi/_neon/LENO/chm_agb_2021.png')
plt.show()

#%%

v1 = v[v['area'] == 400]
v1['AGB'] = v1['AGB_2004'] * 10 / v['area']

# Plot scatter plot
plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(v1['chm_RH95'], v1['AGB'], label='Scatter Plot')

# Define the curve function a*x**b
def curve_function(x, a, b):
    return a * np.power(x, b)

# Number of iterations
num_iterations = 2

# Initialize variables to store the best parameters and corresponding fit error
best_params = None
best_fit_error = float('inf')

# Perform iterative curve fitting
for _ in range(num_iterations):
    # Random initial guess for curve fitting parameters
    initial_guess = [3, 1]

    params, covariance = curve_fit(curve_function, v1['chm_RH95'], v1['AGB'], p0=initial_guess)

    fit_error = np.sum((v1['AGB'] - curve_function(v1['chm_RH95'], *params))**2)

    if fit_error < best_fit_error:
        best_fit_error = fit_error
        best_params = params

# Generate y values for the best-fitted curve
x_curve = np.linspace(0, 50, 100)
y_curve = curve_function(x_curve, *best_params)

plt.plot(x_curve, y_curve, color='red', label=f'Best Fit: {best_params[0]:.2f} * x**{best_params[1]:.2f}')

# Use plt instead of ax for the second plot
x_curve2 = np.linspace(0, 50, 100)  
y_curve2 = 3.26 * np.power(x_curve2, 1.34)
plt.plot(x_curve2, y_curve2, color='blue', label='3.26*x**1.74')

plt.ylim(0, 500)
plt.xlabel('CHM Height (chm_RH95)')
plt.ylabel('AGB 2004')
plt.title('Scatter Plot and Best Fit Curve')

# Add legend
plt.legend()

# Show or save the plot
plt.show()
