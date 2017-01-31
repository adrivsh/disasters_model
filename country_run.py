from pandas import concat,Series,DataFrame,read_csv,HDFStore,set_option
from lib_for_growth_model import *
from lib_for_disasters import *
import numpy as np

def shock_country(finalhhframe,disasters_cursor,fa_poor,fa_nonpoor,vp,vr,disasters_param_bounds,countrycode,hazard_name):
	casual_var    = valuefromcursor(disasters_param_bounds.loc['casual_var',:],disasters_cursor)
	if (vp is None)|(vr is None):
		losses_poor = valuefromcursor(disasters_param_bounds.loc['losses_poor',:],disasters_cursor)
		losses_rich = valuefromcursor(disasters_param_bounds.loc['losses_rich',:],disasters_cursor)
	else:
		losses_poor = (1+casual_var)*vp
		losses_rich = (1+casual_var)*vr
	new_hhframe    = finalhhframe.copy()
	hazard_share_p = (1+casual_var)*fa_poor
	hazard_share_r = (1+casual_var)*fa_nonpoor
	new_hhframe    = shock(new_hhframe,hazard_share_p,hazard_share_r,losses_poor,losses_rich)
	return new_hhframe

def drought_country_impact(finalhhframe,countrycode,fa,disasters_cursor,wbreg,food_share_data,data2day,disasters_param_bounds):
	
	new_hhframe    = finalhhframe.copy()
	shares_outside       = get_shares_outside(finalhhframe)
	food_prices_increase = valuefromcursor(disasters_param_bounds.loc['food_prices_increase',:],disasters_cursor)
	casual_var           = valuefromcursor(disasters_param_bounds.loc['casual_var',:],disasters_cursor)
	drought_share        = (1+casual_var)*fa
	losses_poor          = valuefromcursor(disasters_param_bounds.loc['losses_poor',:],disasters_cursor)
	new_hhframe          = shock_drought(new_hhframe,drought_share,losses_poor,food_prices_increase,wbreg,food_share_data,data2day)
	return new_hhframe