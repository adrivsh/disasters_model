import numpy as np
from pandas import read_csv, Series
from lib_for_growth_model import *

def get_shares_outside(futurehhframe):
	shares_outside = Series(index=['ag','manu','serv'])
	shares_outside['ag']   = 0.8
	shares_outside['manu'] = 0.5
	shares_outside['serv'] = 0.3
	if np.average(futurehhframe['Y'],weights=futurehhframe['weight']*futurehhframe['nbpeople'])>10000:
		shares_outside['manu'] = 0.1
		shares_outside['serv'] = 0.05
	return shares_outside
	

def food_price_impact(futurehhframe,price_change,wbreg,food_share_data,day2data):
	list_segments = [(0,2.97),(2.97,8.44),(8.44,23.03),(23.03,float("inf"))]
	for (i,j) in list_segments:
		select=(futurehhframe['Y']>=i*day2data)&(futurehhframe['Y']<j*day2data)
		if j<24:
			food_share = food_share_data.ix[(food_share_data.wbregion==wbreg)&(food_share_data.consSeg=="{}-{}".format(i,j)),"food_share"]
		else:
			food_share = food_share_data.ix[(food_share_data.wbregion==wbreg)&(food_share_data.consSeg==">{}".format(i)),"food_share"]
		new_price_index = float(food_share)*(1-price_change)+(1-float(food_share))
		futurehhframe.ix[select,'Y'] = futurehhframe.ix[select,'Y']/new_price_index
	return futurehhframe
	
def shock(new_hhframe,hazard_share_p,hazard_share_r,losses_poor,losses_rich):
	'''
	This models a negative shock (what would people's revenue be without the impacts of the shocks?) So income increase
	'''
	if hazard_share_p<=0:
		return new_hhframe
	else:
		hazard_share_p = min(0.999,hazard_share_p)
		hazard_share_r = min(0.999,hazard_share_r)
		
		th = perc_with_spline(new_hhframe['Y'],new_hhframe['nbpeople']*new_hhframe['weight'],0.2)
		isbelowline    = new_hhframe.Y<float(th)
		finalhhframe_r = new_hhframe.ix[~isbelowline,:].copy()
		finalhhframe_p = new_hhframe.ix[isbelowline,:].copy()
		
		finalhhframe_r_affected           = finalhhframe_r.copy()
		finalhhframe_r_affected['weight'] = finalhhframe_r_affected['weight']*(hazard_share_r)
		finalhhframe_r_affected['Y']      = finalhhframe_r_affected['Y']*(1+losses_rich)
		finalhhframe_r_affected.index     = finalhhframe_r_affected.index.astype(object).astype(str)+"ra"

		finalhhframe_r['weight']          = finalhhframe_r['weight']*(1-hazard_share_r)
		
		finalhhframe_p_affected           = finalhhframe_p.copy()
		finalhhframe_p_affected['weight'] = finalhhframe_p_affected['weight']*(hazard_share_p)
		finalhhframe_p_affected['Y']      = finalhhframe_p_affected['Y']*(1+losses_poor)
		finalhhframe_p_affected.index     = finalhhframe_p_affected.index.astype(object).astype(str)+"pa"

		finalhhframe_p['weight']          = finalhhframe_p['weight']*(1-hazard_share_p)

	return finalhhframe_r.append(finalhhframe_r_affected).append(finalhhframe_p).append(finalhhframe_p_affected)
	
def shock_drought(finalhhframe,sh_people_affected,losses_poor,food_prices_increase,wbreg,food_share_data,data2day):
	'''
	This models a negative shock (what would people's revenue be without the impacts of the shocks?) So income increase
	'''
	if sh_people_affected<=0:
		return finalhhframe
	else:
		sh_people_affected         = min(0.999,sh_people_affected)
		finalhhframetemp           = finalhhframe.copy()
		finalhhframe21             = finalhhframe.copy()
		finalhhframetemp['weight'] = finalhhframetemp['weight']*(1-sh_people_affected)
		finalhhframe21['weight']   = finalhhframe21['weight']*(sh_people_affected)
		select_farmers             = (finalhhframe21.cat3workers>0)|(finalhhframe21.cat4workers>0)
		finalhhframe21.ix[select_farmers,'Y'] = finalhhframe21.ix[select_farmers,'Y']*(1+losses_poor)
		finalhhframe21.ix[select_farmers,'totY'] = finalhhframe21.ix[select_farmers,'totY']*(1+losses_poor)
		finalhhframe21             = food_price_impact(finalhhframe21,food_prices_increase,wbreg,food_share_data,data2day)
		finalhhframe21.index       = finalhhframe21.index.astype(object).astype(str)+"s"
		
		finalhhframetemp2          = finalhhframetemp.copy()
		finalhhframe22             = finalhhframetemp.copy()
		sh_people_affected         = min(0.999,50*sh_people_affected)
		finalhhframetemp2['weight'] = finalhhframetemp['weight']*(1-sh_people_affected)
		finalhhframe22['weight']   = finalhhframe22['weight']*(sh_people_affected)
		finalhhframe22             = food_price_impact(finalhhframe22,food_prices_increase,wbreg,food_share_data,data2day)
		finalhhframe22.index       = finalhhframe22.index.astype(object).astype(str)+"ss"
		finalhhframetemp2          = finalhhframetemp2.append(finalhhframe21).append(finalhhframe22)
	return finalhhframetemp2

def temperature_impact(futureinc,shares_outside,temp_impact):
	futureinc['cat3workers']   = futureinc['cat3workers']*(1-shares_outside['ag'])  +futureinc['cat3workers']*(1+temp_impact)*shares_outside['ag']
	futureinc['cat4workers']   = futureinc['cat4workers']*(1-shares_outside['ag'])  +futureinc['cat4workers']*(1+temp_impact)*shares_outside['ag']
	futureinc['cat1workers']   = futureinc['cat1workers']*(1-shares_outside['serv'])+futureinc['cat1workers']*(1+temp_impact)*shares_outside['serv']
	futureinc['cat2workers']   = futureinc['cat2workers']*(1-shares_outside['serv'])+futureinc['cat2workers']*(1+temp_impact)*shares_outside['serv']
	futureinc['cat5workers']   = futureinc['cat5workers']*(1-shares_outside['manu'])+futureinc['cat5workers']*(1+temp_impact)*shares_outside['manu']
	futureinc['cat6workers']   = futureinc['cat6workers']*(1-shares_outside['manu'])+futureinc['cat6workers']*(1+temp_impact)*shares_outside['manu']
	return futureinc
			
def valuefromcursor(boundsrow,cursor):
	return boundsrow['min']*(1-cursor)+boundsrow['max']*(cursor)
	

	
	