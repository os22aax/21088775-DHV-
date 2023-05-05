#!/usr/bin/env python
# coding: utf-8

# In[194]:


pip install matplotlib


# In[1]:


import numpy as np
import pandas as pd
#%matplotlib inline
import matplotlib.pyplot as plt
#import matplotlib.pypolt as plt
#import matplotlib as plt
#from matplotlib import pyplot as pllt
import seaborn as sns


# In[2]:


#load .csv file with details with Agricultural land 
col_names=['Country Name','Country Code','Indicator Name','1960','1961','1962','1963','1964','1965','1966','1967','1968','1969',
           '1970','1971','1972','1973','1974','1975','1976','1977','1978','1979',
          '1980','1981','1982','1983','1984','1985','1986','1987','1988','1989',
          '1990','1991','1992','1993','1994','1995','1996','1997','1998','1999',
          '2000','2001','2002','2003','2004','2005','2006','2007','2008','2009',
          '2010','2011','2012','2013','2014','2015','2016','2017','2018','2019',
          '2020','2021']
AgriDF = pd.read_csv("agri.csv", names = col_names, skiprows=5)
AgriDF.head()


# In[3]:


#select specific colums from dataset
AgriDF = AgriDF[['Country Name','Indicator Name','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999',
          '2000','2001','2002','2003','2004','2005','2006','2007','2008','2009',
          '2010','2011','2012','2013','2014','2015','2016','2017','2018','2019',
          '2020','2021']]
AgriDF.head()


# In[4]:


#select specific countries from data set and set index correctly with new data set
countries = ['China', 'United States', 'India', 'Russian Federation', 'Japan', 'Germany', 'Canada', 'Iran, Islamic Rep.', 'Korea, Rep.', 'Indonesia']
AgriDF_1 = AgriDF[AgriDF['Country Name'].isin(countries)].reset_index(drop=True)
AgriDF_1.head()


# In[5]:


#set index as country column
AgriDF_1.set_index('Country Name').head()


# In[6]:


#transpose the Dataframe
AgriDF_1 = AgriDF_1.transpose()
AgriDF_1.head()


# In[8]:


#set the countries as new column names 
AgriDF_T = AgriDF_1.rename(columns=AgriDF_1.iloc[0])
AgriDF_T.head()


# In[9]:


#drop selected column names 
AgriDF_T_L = AgriDF_T.drop(['Country Name', 'Indicator Name'])
AgriDF_T_L.head()


# In[10]:


#set index name as Year
AgriDF_T_L.index.names = ['Year']
AgriDF_T_L.head()


# In[12]:


#AgriDF_T['Agri_mean'] = AgriDF_T.max()
#AgriDF_T
AgriDF_T_L.dtypes


# In[13]:


#change data type to float
AgriDF_T_L = AgriDF_T_L.astype(float)


# In[14]:


AgriDF_T_L.dtypes


# In[15]:


#add new column as Agricultural Mean
AgriDF_T_L['Agricultural Mean'] = AgriDF_T_L.mean(axis=1)
AgriDF_T_L.head()
#AgriDF_T.max(axis=1)


# In[16]:


#select two specific columns from data frame
Agri_mean = AgriDF_T_L[['Agricultural Mean']]
Agri_mean.head()


# In[17]:


#import Net migration data set as Data frame
colM_names=['Country Name','Country Code','Indicator Name','Indicator Code','1960','1961','1962','1963','1964','1965','1966','1967','1968','1969',
           '1970','1971','1972','1973','1974','1975','1976','1977','1978','1979',
          '1980','1981','1982','1983','1984','1985','1986','1987','1988','1989',
          '1990','1991','1992','1993','1994','1995','1996','1997','1998','1999',
          '2000','2001','2002','2003','2004','2005','2006','2007','2008','2009',
          '2010','2011','2012','2013','2014','2015','2016','2017','2018','2019',
          '2020','2021']
MigrateDF = pd.read_csv('mig.csv', names = colM_names, skiprows=5)
MigrateDF.head()


# In[18]:


MigrateDF = MigrateDF[['Country Name','Indicator Name','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999',
          '2000','2001','2002','2003','2004','2005','2006','2007','2008','2009',
          '2010','2011','2012','2013','2014','2015','2016','2017','2018','2019',
          '2020','2021']]
MigrateDF.head()


# In[19]:


countries = ['China', 'United States', 'India', 'Russian Federation', 'Japan', 'Germany', 'Canada', 'Iran, Islamic Rep.', 'Korea, Rep.', 'Indonesia']
MigrateDF_1 = MigrateDF[MigrateDF['Country Name'].isin(countries)].reset_index(drop=True)
MigrateDF_1.head()


# In[20]:


MigrateDF_1.set_index('Country Name').head()


# In[21]:


MigrateDF_1 = MigrateDF_1.transpose()
MigrateDF_1.head()


# In[22]:


MigrateDF_T = MigrateDF_1.rename(columns=MigrateDF_1.iloc[0])
MigrateDF_T.head()


# In[23]:


MigrateDF_T = MigrateDF_T.drop(['Country Name', 'Indicator Name'])
MigrateDF_T.head()


# In[24]:


MigrateDF_T.index.names = ['Year']
MigrateDF_T.head()


# In[26]:


MigrateDF_T = MigrateDF_T.astype(float)


# In[27]:


MigrateDF_T.dtypes


# In[28]:


MigrateDF_T['Migrate_mean'] = MigrateDF_T.mean(axis=1)
MigrateDF_T.head()


# In[29]:


Migrate_mean1 = MigrateDF_T[['Migrate_mean']]
Migrate_mean1.head()


# In[30]:


#import Urban population dataset as Dataframe
colC_names=['Country Name','Country Code','Indicator Name','Indicator Code','1960','1961','1962','1963','1964','1965','1966','1967','1968','1969',
           '1970','1971','1972','1973','1974','1975','1976','1977','1978','1979',
          '1980','1981','1982','1983','1984','1985','1986','1987','1988','1989',
          '1990','1991','1992','1993','1994','1995','1996','1997','1998','1999',
          '2000','2001','2002','2003','2004','2005','2006','2007','2008','2009',
          '2010','2011','2012','2013','2014','2015','2016','2017','2018','2019',
          '2020','2021']
worldDF = pd.read_csv('co2.csv', names= colC_names, skiprows=5)
worldDF.head()


# In[31]:


Co2DF = worldDF[worldDF["Indicator Name"] == 'CO2 emissions (kt)'].reset_index(drop = True)
Co2DF.head()


# In[32]:


Co2DF = Co2DF[['Country Name','Indicator Name','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999',
          '2000','2001','2002','2003','2004','2005','2006','2007','2008','2009',
          '2010','2011','2012','2013','2014','2015','2016','2017','2018','2019',
          '2020','2021']]
Co2DF.head()


# In[33]:


countries = ['China', 'United States', 'India', 'Russian Federation', 'Japan', 'Germany', 'Canada', 'Iran, Islamic Rep.', 'Korea, Rep.', 'Indonesia']
Co2DF_1 = Co2DF[Co2DF['Country Name'].isin(countries)].reset_index(drop=True)
Co2DF_1.head()


# In[34]:


#fill NAN values with values indicates in next columns
Co2DF_1 = Co2DF_1.fillna(method="ffill", axis="columns")
Co2DF_1.head()


# In[35]:


Co2DF_1.set_index('Country Name').head()


# In[36]:


Co2DF_1 = Co2DF_1.transpose()
Co2DF_1.head()


# In[37]:


Co2DF_T = Co2DF_1.rename(columns=Co2DF_1.iloc[0])
Co2DF_T.head()


# In[38]:


Co2DF_T = Co2DF_T.drop(['Country Name', 'Indicator Name'])
Co2DF_T.head()


# In[39]:


Co2DF_T.index.names = ['Year']
Co2DF_T.head()


# In[40]:


import matplotlib.gridspec as gridspec


# In[41]:


Co2DF_T.dtypes


# In[42]:


Co2DF_T = Co2DF_T.astype(float)


# In[43]:


Co2DF_T.dtypes


# In[44]:


Co2DF_T.head()


# In[46]:


Co2DF_T['Co2_mean'] = Co2DF_T.mean(axis=1)
Co2DF_T.head()


# In[47]:


Co2_mean1 = Co2DF_T[['Co2_mean']]
Co2_mean1.head()


# In[119]:


Co2DF_T_Country = Co2DF_T[['China', 'United States', 'India', 'Russian Federation', 'Japan', 'Germany', 'Canada', 
                     'Iran, Islamic Rep.', 'Korea, Rep.', 'Indonesia']]
Co2DF_T_Country.head()


# In[48]:


PopulationDF = worldDF[worldDF["Indicator Name"] == 'Population, total'].reset_index(drop = True)
PopulationDF.head()


# In[49]:


PopulationDF = PopulationDF[['Country Name','Indicator Name','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999',
          '2000','2001','2002','2003','2004','2005','2006','2007','2008','2009',
          '2010','2011','2012','2013','2014','2015','2016','2017','2018','2019',
          '2020','2021']]
PopulationDF.head()


# In[50]:


countries = ['China', 'United States', 'India', 'Russian Federation', 'Japan', 'Germany', 'Canada', 'Iran, Islamic Rep.', 'Korea, Rep.', 'Indonesia']
PopulationDF_1 = PopulationDF[PopulationDF['Country Name'].isin(countries)].reset_index(drop=True)
PopulationDF_1.head()


# In[51]:


PopulationDF_1 = PopulationDF_1.fillna(method="ffill", axis="columns")
PopulationDF_1.head()


# In[52]:


PopulationDF_1.set_index('Country Name').head()


# In[53]:


PopulationDF_1 = PopulationDF_1.transpose()
PopulationDF_1.head()


# In[54]:


PopulationDF_T = PopulationDF_1.rename(columns=PopulationDF_1.iloc[0])
PopulationDF_T.head()


# In[55]:


PopulationDF_T = PopulationDF_T.drop(['Country Name', 'Indicator Name'])
PopulationDF_T.head()


# In[56]:


PopulationDF_T.index.names = ['Year']
PopulationDF_T.head()


# In[58]:


PopulationDF_T.dtypes


# In[59]:


PopulationDF_T = PopulationDF_T.astype(float)


# In[60]:


PopulationDF_T.dtypes


# In[61]:


PopulationDF_T['Population_mean'] = PopulationDF_T.mean(axis=1)
PopulationDF_T.head()


# In[62]:


Population_mean1 = PopulationDF_T[['Population_mean']]
Population_mean1.head()


# In[63]:


#merge mean columns from four diffternt dataframes 
from functools import reduce
data_merge = reduce(lambda left, right:
                   pd.merge(left, right, left_index=True, right_index=True, how='outer' ),
                    [Agri_mean, Population_mean1, Co2_mean1, Migrate_mean1])
#pd.merge(Agri_mean, Temp_mean1, Migrate_mean1, Co2_mean1, left_index=True, right_index=True, how='outer')
data_merge.head()


# In[64]:


data_merge.dtypes


# In[65]:


#add index column for data_merge dataframe
dd = data_merge.reset_index()
dd.head()


# In[66]:


dd = dd.astype({'Year':'string'})
dd.head()


# In[67]:


dd.dtypes


# In[70]:


from mpl_toolkits import mplot3d


# In[75]:


#find the correlation among the mean values
df_corr = data_merge[['Agricultural Mean', 'Population_mean', 'Migrate_mean', 'Co2_mean']].corr()


# In[132]:


#plot heatmap to identify the correlation
#sns.heatmap(df_corr, annot=True)


# In[134]:


#set grids for place dashboard using gridspec
fig = plt.figure(figsize=(22,22), constrained_layout = False , dpi=300)
gs = fig.add_gridspec(nrows=7, ncols=6, hspace=0.8, wspace=0.5)

ax1 = fig.add_subplot(gs[0,:6]) 
ax1.set_title('HOW CO2 EFFECTS WORLD\nby Omesha Prashanthika Samarakoon (ID:21088775)',dict(size=25, color='blue', 
                                                                                            weight='bold'))
# define function to split the text to add discription to the dashboard
def split_text(text, max_length):
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + len(current_line) > max_length:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word)
    lines.append(' '.join(current_line))
    return '\n'.join(lines)

ax1.set_axis_off()
long_text =''' World co2 emission level is increasing daily with the development of the world In (figure 01) we can
         identify comparison of thr co2 emission between 10 specific countries in last 3 decades. From selected 
         countries, China shows the highest level of co2 emission to the environment it gradually incresced
         till year 2000 and after that shows rapid rise up. As well as USA and India is also showing 
         increasing trend over last 30 years and lowest contribution done by Indonesia from selected countries.
         '''
wrapped_text = split_text(long_text, max_length=160)
ax1.text(0, 0, wrapped_text, dict(size=16))

#place Co2 emission dataframe plot as figure one plot
ax2 = fig.add_subplot(gs[1:3,:3]) 
ax2.set_title('Co2 Emission of Last 3 Decades (Figure 01)', dict(size=17, color='grey', weight='bold'))

for x in Co2DF_T_Country.columns:
    co2plot = ax2.plot(Co2DF_T_Country, label = x)
ax2.legend(Co2DF_T_Country.columns)
ax2.set_xlabel('Year', fontsize=14)
ax2.set_ylabel('Co2 Emission', fontsize=14 )
ax2.tick_params(axis = 'x', labelrotation = 90)
ax2.grid()

#plot mean value comparission of 4 indicators in one plot with difftent y-axis
ax3 = fig.add_subplot(gs[1:3,3:6]) 
ax3.set_title('Comparison Between Mean Values (Figure 02)', dict(size=17, color='grey', weight='bold'))
                                                                 
years = ['1990','1991','1992','1993','1994','1995','1996','1997','1998','1999',
          '2000','2001','2002','2003','2004','2005','2006','2007','2008','2009',
          '2010','2011','2012','2013','2014','2015','2016','2017','2018','2019',
          '2020','2021']
first = dd['Co2_mean']
second = dd['Population_mean']
third = dd['Agricultural Mean']
fourth = dd['Migrate_mean']
plt.grid()
ax3.plot(years, first, color='blue')

ax4 = ax3.twinx()
ax4.plot(years, second, color='green')

ax5 = ax3.twinx()
ax5.plot(years, third, color='red')
ax5.spines['right'].set_position(('outward',40))

ax6 = ax3.twinx()
ax6.plot(years, fourth, color='purple')
ax6.spines['right'].set_position(('outward',90))
ax3.set_xlabel('Year',fontsize=14)

ax3.set_ylabel('Co2 Emmission', color='blue', fontsize=12)
ax4.set_ylabel('Total Population', color='green', fontsize=12)
ax5.set_ylabel('Agricultural Land', color='red', fontsize=12)
ax6.set_ylabel('Number of Migrates', color='purple', fontsize=12)
ax3.tick_params(axis='y', colors='blue')
ax4.tick_params(axis='y', colors='green')
ax5.tick_params(axis='y', colors='red')
ax6.tick_params(axis='y', colors='purple')

ax6.spines['left'].set_color('blue')
ax4.spines['right'].set_color('green')
ax5.spines['right'].set_color('red')
ax6.spines['right'].set_color('purple')

ax3.tick_params(axis='x', labelrotation = 90)

#select countries with highest, middle and lowest co2 emission and plot scatter plots to identify the relationships among them
#plot for china
ax4 = fig.add_subplot(gs[3:5,:2], projection='3d') 
ax4.set_title('[0,4]')
fg = ax4.scatter3D(MigrateDF_T['China'], PopulationDF_T['China'], AgriDF_T_L['China'],
                   s= Co2DF_T['China']/10000, c=Co2DF_T['China'])

ax4.set_xlabel('Net Migrates', fontsize=14)
ax4.set_ylabel('Total Population', fontsize=14)
ax4.set_zlabel('Agricultural Lands', fontsize=14)
ax4.set_title('China (Figure 03)', dict(size=17, color='grey', weight='bold'))
plt.colorbar(fg,  pad=0.2)


#plot for Japan
ax5 = fig.add_subplot(gs[3:5,2:4], projection='3d') 
ax5.set_title('[0,5]')
fg2 = ax5.scatter3D(MigrateDF_T['Japan'], PopulationDF_T['Japan'], AgriDF_T_L['Japan'],
                   s= Co2DF_T['Japan']/10000, c=Co2DF_T['Japan'])

ax5.set_xlabel('Net Migrates', fontsize=14)
ax5.set_ylabel('Total Population', fontsize=14)
ax5.set_zlabel('Agricultural Lands', fontsize=14)
ax5.set_title('Japan (Figure 04)', dict(size=17, color='grey', weight='bold'))
plt.colorbar(fg2,  pad=0.2)

#plot for Indonesia
ax6 = fig.add_subplot(gs[3:5,4:6], projection='3d') 
ax6.set_title('[0,6]')
fg1 = ax6.scatter3D(MigrateDF_T['Indonesia'], PopulationDF_T['Indonesia'], AgriDF_T_L['Indonesia'],
                   s= Co2DF_T['Indonesia']/10000, c=Co2DF_T['Indonesia'])

ax6.set_xlabel('Net Migrates', fontsize=14)
ax6.set_ylabel('Total Population', fontsize=14)
ax6.set_zlabel('Agricultural Lands', fontsize=14)
ax6.set_title('Indonesia (Figure 05)', dict(size=17, color='grey', weight='bold'))
plt.colorbar(fg1,  pad=0.2)

#plot heatmap to identify the correlation among indicators 
ax7 = fig.add_subplot(gs[5:7,:3]) 
ax7.set_title('Correlation Between Indicators (Figure 06)', dict(size=17, color='grey', weight='bold'))
sns.heatmap(df_corr, annot=True)
ax7.set_xlabel('Indicator Name', fontsize=14)
ax7.set_ylabel('Indicator Name', fontsize=14)


ax8 = fig.add_subplot(gs[5:7,3:6]) 
ax8.set_axis_off()

long_text2 = '''In figure 2 we can identify the relationship among Carbon Dioxide Emission, 
Total Population, Agricultural Lands and Net Migrants in considerable countries over last 30 years
when plotting average values of these indicators, it is clearly visible carbon dioxide emission 
and total population shows direct proportional relationship and agricultural lands and Co2 emission
shows inverse proportional relationship between them. And net migrates don’t show any special relationship with 
respect to the co2 emission.In Figure 3,4,5 shows the relationship among those factors in 
China (highest Co2 Emission), Japan (average Co2 Emission) and Indonesia (lowest Co2 emission) 
consequently. In figure 6, we can identify the correlation among the chosen indicators,
Co2 emission vs Total population shows high positive correlation, 
and agricultural lands vs Co2 emission shows high negative correlation. As this trend we can imagine,
in future with the increse of populaton, Co2 emission will be increse and people have to face the lack 
of food if they unbale to find a way to produce food, instead of output agricultural lands.'''

wrapped_text2 = split_text(long_text2, max_length=80)
ax8.text(0, 0, wrapped_text2, dict(size=16))

plt.show()


# In[135]:


fig.savefig('21088775.png', dpi=300)


# In[ ]:




