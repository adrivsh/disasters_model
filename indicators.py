from pandas import Series,DataFrame
import numpy as np
from perc import *
from lib_for_data_reading import *

def calc_indic(futurehhframe,data2day,extrpoor_threshold):
	
	indicators                 = DataFrame()
	percentiles                = perc_with_spline(futurehhframe.Y,futurehhframe.weight*futurehhframe.nbpeople,np.arange(0,1,0.01))
	quintilescum               = wp(futurehhframe.Y,futurehhframe.weight*futurehhframe.nbpeople,[0.2,0.4,1])
	quintilespc                = quintilescum/quintilescum[-1]
	indicators['GDP']          = [sum(futurehhframe.Y*futurehhframe.weight*futurehhframe.nbpeople)]
	indicators['avincome']     = [np.average(futurehhframe.Y,weights=futurehhframe.weight*futurehhframe.nbpeople)]
	indicators['incbott10']    = [poverty_indic(percentiles,0,10)]
	indicators['incbott20']    = [poverty_indic(percentiles,0,20)]
	indicators['inc2040']      = [poverty_indic(percentiles,20,40)]
	indicators['incbott40']    = [poverty_indic(percentiles,0,40)]
	indicators['quintilecum1'] = [quintilescum[0]]
	indicators['quintilecum2'] = [quintilescum[1]]
	indicators['quintilepc1']  = [quintilespc[0]]
	indicators['quintilepc2']  = [quintilespc[1]]
	indicators['extrpoor']     = [poor_people(futurehhframe.Y,futurehhframe.weight*futurehhframe.nbpeople,extrpoor_threshold*data2day)]
	indicators['below2']       = [poor_people(futurehhframe.Y,futurehhframe.weight*futurehhframe.nbpeople,2*data2day)]
	indicators['below4']       = [poor_people(futurehhframe.Y,futurehhframe.weight*futurehhframe.nbpeople,4*data2day)]
	indicators['below6']       = [poor_people(futurehhframe.Y,futurehhframe.weight*futurehhframe.nbpeople,6*data2day)]
	indicators['below8']       = [poor_people(futurehhframe.Y,futurehhframe.weight*futurehhframe.nbpeople,8*data2day)]
	indicators['below10']      = [poor_people(futurehhframe.Y,futurehhframe.weight*futurehhframe.nbpeople,10*data2day)]
	indicators['gini']         = [gini(futurehhframe.Y,futurehhframe.weight*futurehhframe.nbpeople)]
	indicators['tot_pop']      = [sum(futurehhframe.weight*futurehhframe.nbpeople)]
	indicators['gapextrpoor']       = [poverty_gap(futurehhframe.Y,futurehhframe.weight*futurehhframe.nbpeople,extrpoor_threshold*data2day)]
	indicators['gap2']         = [poverty_gap(futurehhframe.Y,futurehhframe.weight*futurehhframe.nbpeople,2*data2day)]
	indicators['gap4']         = [poverty_gap(futurehhframe.Y,futurehhframe.weight*futurehhframe.nbpeople,4*data2day)]
	indicators['gap6']         = [poverty_gap(futurehhframe.Y,futurehhframe.weight*futurehhframe.nbpeople,6*data2day)]
	indicators['gap8']         = [poverty_gap(futurehhframe.Y,futurehhframe.weight*futurehhframe.nbpeople,8*data2day)]
	indicators['gap10']        = [poverty_gap(futurehhframe.Y,futurehhframe.weight*futurehhframe.nbpeople,10*data2day)]
		
	ag = (futurehhframe['cat3workers']>0)|(futurehhframe['cat4workers']>0)
	
	indicators['childrenag']  = sum(futurehhframe.ix[ag,'children']*futurehhframe.ix[ag,'weight'])
	indicators['childrenonag']= sum(futurehhframe.ix[~ag,'children']*futurehhframe.ix[~ag,'weight'])
	indicators['peopleag']    = sum(futurehhframe.ix[ag,'nbpeople']*futurehhframe.ix[ag,'weight'])
	indicators['peoplenonag'] = sum(futurehhframe.ix[~ag,'nbpeople']*futurehhframe.ix[~ag,'weight'])
	
	quintilesag               = wp(reshape_data(futurehhframe.ix[ag,'Y']),reshape_data(futurehhframe.ix[ag,'weight']*futurehhframe.ix[ag,'nbpeople']),[0.2,1],cum=True)
	quintilesnonag            = wp(reshape_data(futurehhframe.ix[~ag,'Y']),reshape_data(futurehhframe.ix[~ag,'weight']*futurehhframe.ix[~ag,'nbpeople']),[0.2,1],cum=True)
	quintilesagpc             = quintilesag/quintilesag[-1]
	quintilesnonagpc          = quintilesnonag/quintilesnonag[-1]
	
	indicators['incsharebott20ag'] = quintilesagpc[0]
	indicators['incsharebott20nonag'] = quintilesnonagpc[0]
		
	indicators['poorag']      = [poor_people(futurehhframe.ix[ag,'Y'],futurehhframe.ix[ag,'weight']*futurehhframe.ix[ag,'nbpeople'],extrpoor_threshold*data2day)]
	indicators['poornonag']   = [poor_people(futurehhframe.ix[~ag,'Y'],futurehhframe.ix[~ag,'weight']*futurehhframe.ix[~ag,'nbpeople'],extrpoor_threshold*data2day)]
		
	indicators['avincomeag'] = [np.average(futurehhframe.ix[ag,'Y'],weights=futurehhframe.ix[ag,'weight']*futurehhframe.ix[ag,'nbpeople'])]
	indicators['avincomenonag']   = [np.average(futurehhframe.ix[~ag,'Y'],weights=futurehhframe.ix[~ag,'weight']*futurehhframe.ix[~ag,'nbpeople'])]
	
	return indicators	
	
def poor_people(income,weights,povline):
	isbelowline = (income<povline)
	thepoor     = weights.values*isbelowline.values
	nbpoor      = thepoor.sum()
	return nbpoor
	
def find_perc(y,w,theperc,density):
	'''
	The very sophisticated way of finding percentiles
	'''
	normalization = integrate.quad(density,0,np.inf)
	estime = wp(y,w,[theperc],cum=False)
	def find_root(x,normalization,density,theperc):
		integrale = integrate.quad(density,0,x)
		return integrale[0]/normalization[0]-theperc
	out = optimize.fsolve(find_root, estime, args=(normalization,density,theperc))
	return out
	
def poverty_indic_kde(income,weights,threshold,density):
	'''
	The very sophisticated way of finding percentiles and the average income of people between percentiles
	'''
	if type(threshold)==float:
		inclim20        = find_perc(income,weights,threshold,density)[0]
		isbelowline     = (income<=inclim20)
	elif type(threshold)==list:
		minlim = find_perc(income,weights,threshold[0],density)[0]
		maxlim = find_perc(income,weights,threshold[1],density)[0]
		isbelowline     = (income<=maxlim)&(income>=minlim)
		if sum(isbelowline)==0:
			isbelowline     = (income==min(income, key=lambda x:abs(x-maxlim)))|(income==min(income, key=lambda x:abs(x-minlim)))
	out = np.average(income[isbelowline],weights=weights[isbelowline])
	return out
	
def poverty_indic_spec(income,weights,threshold):
	'''
	For special cases
	'''
	if type(threshold)==list:
		minlim = threshold[0]
		maxlim = threshold[1]
		isbelowline     = (income<=maxlim)&(income>=minlim)
		if sum(isbelowline)==0:
			isbelowline     = (income==min(income, key=lambda x:abs(x-maxlim)))|(income==min(income, key=lambda x:abs(x-minlim)))
	else:
		isbelowline     = (income<=threshold)
	out = np.average(income[isbelowline],weights=weights[isbelowline])
	return out
	
def poverty_indic(percentiles,limit1,limit2):
	out = percentiles[limit1:limit2].sum()/(limit2-limit1)
	return out
	
def gini(income,weights):
	inc = np.asarray(reshape_data(income))
	wt  = np.asarray(reshape_data(weights))
	i   = np.argsort(inc) 
	inc = inc[i]
	wt  = wt[i]
	y   = np.cumsum(np.multiply(inc,wt))
	y   = y/y[-1]
	x   = np.cumsum(wt)
	x   = x/x[-1]
	G   = 1-sum((y[1::]+y[0:-1])*(x[1::]-x[0:-1]))
	return G
	
def poverty_gap(income,weights,povline):
	isbelowline = (income<povline)
	gap         = sum((1-income[isbelowline]/povline)*weights[isbelowline]/sum(weights))
	return gap
