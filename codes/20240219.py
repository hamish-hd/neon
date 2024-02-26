import geopandas as gpd
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
#%%
struct_plant_dir = '/data/gedi/_neon/LENO/NEON_struct-plant'
#Directory of NEON plant structure
#%%
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

#apparent_individual.to_csv(struct_plant_dir+'apparent_individual.csv')

apparent_individual.loc[apparent_individual['measurementHeight'] >= 130]
#%%
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

maptag.to_csv(struct_plant_dir+'maptag.csv')
#%%
sp_names = pd.read_csv('/data/gedi/_neon/Species_Names_Forest_Group.csv')


df1 = pd.merge(apparent_individual,maptag,on='individualID', how = 'outer')
df2 = pd.merge(df1,sp_names, on='scientificName', how='outer')
tree_loc = gpd.read_file('file:///data/gedi/_neon/LENO_2021/tree_locations_20240216_114531.shp')
df3 = pd.merge(df2, tree_loc, left_on='individualID',right_on='individual',how='outer')

#%%

df3['AGB_2004'] = df3.apply(lambda row: np.exp(row['B0'] + (row['B1']*np.log(row['stemDiameter']))), axis=1)
df3['AGB_2014'] = df3.apply(lambda row: np.exp(row['B0_n'] + (row['B1_n']*np.log(row['stemDiameter']))), axis=1)
df3.columns = df3.columns.str.replace('_x','')
df3.columns = df3.columns.str.replace('_y','')
df3 = df3.loc[:, ~df3.columns.duplicated()]
df3 = df3.dropna(subset=['AGB_2004'])
#df3 = df3[df3['plantStatus'].str.contains('live',case=False)]
df3 = df3[df3['plantStatus'].notna() & df3['plantStatus'].str.contains('live', case=False, na=False)]
df3.to_csv(struct_plant_dir+'/master_agb.csv')

#%%
df4 = df3[df3['growthForm'].str.contains('tree', case=False)]
df4['subplotID'] = df4['subplotID'].str.replace(r'_25_.*', '_100')
df4['subplotID'] = df4['subplotID'].str.replace(r'_10_.*', '_100')

df4['plot_subplot']= pd.concat([df4['plotID'], df4['subplotID'].astype(str)], axis=1).apply('_'.join, axis=1)

patterns_to_replace = ['_25_1', '_25_2', '_25_3', '_25_4','_25_unknown','_10_1', '_10_2', '_10_3', '_10_4']
for pattern in patterns_to_replace:
    df4.loc[:, 'plot_subplot'] = df4['plot_subplot'].str.replace(pattern, '_100')

#%%
#chm = pd.read_csv('/data/gedi/_neon/LENO_2021/chm_height_LENO_2021.csv')
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
chm = chm[chm['subplotID'].str.contains('_400')]
chm['plot_year'] = chm['subplotID'] + '_' + chm['year'].astype(str)


#%%
'''
#df4 = df4[df4['year'] == 2022]
df4 = df4[df4['subplotID'].str.contains('_400')]


grouped_df = df4.groupby('plot_subplot')

max_height_df = grouped_df['height'].max().reset_index()

sum_agb_df = grouped_df[['AGB_2004', 'AGB_2014']].sum().reset_index()

result_df = pd.merge(max_height_df, sum_agb_df, on='plot_subplot', how='outer')
print(result_df)
'''
################
#%%
df4 = df4[df4['subplotID'].str.contains('_400')]

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
v = pd.merge(result_df,chm,on='plot_year',how='inner')
#%%
fig = plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(v['chm_RH95'],v['AGB_2014']*10/v['area'])
#plt.ylim(0,200)

#for i, txt in enumerate(v['subplotID']):
#    plt.annotate(txt, (v['chm_RH95'].iloc[i], v['AGB_2004'].iloc[i]*10/v['area'].iloc[i]), fontsize=8)

plt.xlabel('CHM Height')
plt.ylabel('AGB 2004')
plt.title('AGB (2004) vs CHM 95 (LENO 2021)')
plt.xlim(0,40)
plt.ylim(0,300)
plt.savefig('/data/gedi/_neon/LENO/chm_agb_2021.png')
plt.show()

#%%
fig = plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(v['height'],v['AGB_2004']*10/400)
plt.ylim(0,200)
plt.xlabel('Field Height')
plt.ylabel('AGB 2004')
plt.title('AGB (2004) vs Field Height (LENO 2021)')
plt.savefig('/data/gedi/_neon/LENO/height_agb_2021.png')

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
min_count = v['count'].min()
max_count = v['count'].max()

#v = v[v['count'] > 15]
# Create a ScalarMappable for the color mapping with adjusted normalization
norm = Normalize(v['count'].min(), v['count'].max())
sm = ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])

# Map count values to Viridis colors
v['color'] = v['count'].map(sm.to_rgba)

# Create the scatter plot with Viridis colors for each count
fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
sc = ax.scatter(v['chm_RH95'], v['AGB_2004']*10/v['area'], c=v['color'], s=50, edgecolors='k', linewidths=0.5, alpha=0.8)

ax.set_xlabel('CHM Height (chm_RH95)')
ax.set_ylabel('AGB 2004')
ax.set_title('AGB (2004) vs CHM 95 (LENO 2021)')

# Create a colorbar to indicate count values
cbar = plt.colorbar(sm, label='Count')
plt.xlim(0,60)
# Save the plot
plt.savefig('/data/gedi/_neon/LENO/chm_agb_2021.png', bbox_inches='tight')

# Display the plot
plt.show()
