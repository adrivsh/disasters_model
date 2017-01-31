import sys
from pandas import Series,DataFrame,read_csv,concat
import os
from country_run import *
from lib_for_data_reading import *
from lib_for_disasters import *
from indicators import *
import re

data2day           = 30
year               = 2030
ini_year           = 2012
extrpoor_threshold = 1.9

nameofthisround = 'sept2016_allrp_newdf_3'

model             = os.getcwd()
data              = model+'/data/'
scenar_folder     = model+'/scenar_def/'
finalhhdataframes = model+'/finalhhdataframes/'
finalhhdataframes = 'C:\\Users\\julierozenberg\\Box Sync\\household_surveys\\new_create_dataframe\\finalhhdataframes_clean\\'
data_gidd_csv     = model+'/data_gidd_csv_v4/'

#load codes
codes = read_csv('wbccodes2014.csv')

#load disasters data
food_share_data        = read_csv(data+"food_shares_wbreg.csv")
fa_guessed             = read_csv(data+"disasters/hazard_ratios_sept2016_plus_drought.csv",index_col='iso3')
vulnerabilities        = read_csv(data+"disasters/cat_info_sept2016.csv",index_col='iso3')

#uncertain range for some parameters
disasters_param_bounds = read_csv(data+"disasters_param_bounds.csv",index_col="var")
scenar_param_matrix    = read_csv(data+"scenar_param_matrix.csv")

#storage of results
without_disasters = "{}/without_disasters_{}/".format(model,nameofthisround)

if not os.path.exists(without_disasters):
	os.makedirs(without_disasters)
	
#getting the list of available countries
list_csv=os.listdir(finalhhdataframes)
all_surveys=dict()
for myfile in list_csv:
	cc = re.search('(.*)_finalhhframe.csv', myfile).group(1)
	all_surveys[cc]=read_csv(finalhhdataframes+myfile)


for countrycode in list(all_surveys.keys()):

	country_results = DataFrame()
	out = filter_country(countrycode,all_surveys,codes)
	if out is None:
		continue
	else:
		finalhhframe,new_countrycode = out
		finalhhframe = finalhhframe.rename(columns={'Y2012':'Y'})
		finalhhframe['totY'] = finalhhframe.Y*finalhhframe.nbpeople
		wbreg = codes.loc[codes['country']==reverse_correct_countrycode(countrycode),'wbregion'].values[0]
		indicators_bau  = calc_indic(finalhhframe,data2day,extrpoor_threshold)
		indicators_bau.columns = indicators_bau.columns+"_bau"
		
		if countrycode not in fa_guessed.index:
			continue
		# if os.path.isfile(without_disasters+"{}_results_{}.csv".format(countrycode,nameofthisround)):
			# continue
			# country_results = read_csv(without_disasters+"{}_results_{}.csv".format(countrycode,nameofthisround))
			# country_results = country_results.ix[country_results.hazard_name!='floodglofris',:]
			
		hazards = fa_guessed.loc[countrycode,:]
		
		if countrycode not in vulnerabilities.index:
			vp = None
			vr = None
		else:
			vp  = vulnerabilities.loc[countrycode,'vp']
			vr  = vulnerabilities.loc[countrycode,'vr']
				
	for disasters_cursor in [0,0.5,1]:
		
		for (hazard_name,hazard_data) in hazards.groupby('hazard'):
									
			for rp in hazard_data.rp.unique():
				
				fa_poor    = hazard_data.ix[(hazard_data.rp==rp),'fa'].values[0]
				fa_nonpoor = hazard_data.ix[(hazard_data.rp==rp),'fa'].values[0]
				
				if (fa_poor==0)&(fa_nonpoor==0):
					
					outputs = concat([DataFrame([[new_countrycode,disasters_cursor,hazard_name,rp]],columns=['countrycode','disasters_cursor','hazard_name','rp']),calc_indic(finalhhframe,data2day,extrpoor_threshold),indicators_bau],axis=1)
					
				else:
					if hazard_name!='drought':
						new_hhframe  = shock_country(finalhhframe,disasters_cursor,fa_poor,fa_nonpoor,vp,vr,disasters_param_bounds,countrycode,hazard_name)
					else:
						new_hhframe  = drought_country_impact(finalhhframe,countrycode,fa_poor,disasters_cursor,wbreg,food_share_data,data2day,disasters_param_bounds)
					
					# new_hhframe  = new_hhframe.drop(new_hhframe.weight.isnull())
					indicators   = calc_indic(new_hhframe,data2day,extrpoor_threshold)
					
					outputs         = concat([DataFrame([[new_countrycode,disasters_cursor,hazard_name,rp]],columns=['countrycode','disasters_cursor','hazard_name','rp']),indicators,indicators_bau],axis=1)
			
				country_results = country_results.append(outputs,ignore_index=True)

	country_results.to_csv(without_disasters+"{}_results_{}.csv".format(countrycode,nameofthisround),index=False)