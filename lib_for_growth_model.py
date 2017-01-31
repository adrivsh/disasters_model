from pandas import Series,DataFrame
import numpy as np
import cvxopt
from cvxopt.solvers import qp
from perc import *
import statsmodels.api as sm
# import statsmodels.formula.api as sm
from scipy.interpolate import UnivariateSpline
from scipy import integrate, optimize
from kde import gaussian_kde
from lib_for_data_reading import *
from lib_for_disasters import *
import scipy

from statsmodels.nonparametric.kde import kdensity

def calc_pop_desc(characteristics,weights):
	pop_description             = DataFrame(columns=characteristics.columns)
	pop_description.ix['pop',:] = np.dot(characteristics.T,weights)
	return pop_description

	
def keep_characteristics_to_reweight(finalhhframe):
	characteristics                 = DataFrame()
	characteristics['old']          = finalhhframe['old']
	characteristics['children']     = finalhhframe['children']
	characteristics['unemployed']   = finalhhframe['cat7workers']
	characteristics['skillworkers'] = finalhhframe['cat2workers']+finalhhframe['cat4workers']+finalhhframe['cat6workers']
	characteristics['agworkers']    = finalhhframe['cat3workers']+finalhhframe['cat4workers']
	characteristics['manuworkers']  = finalhhframe['cat5workers']+finalhhframe['cat6workers']
	characteristics['servworkers']  = finalhhframe['cat1workers']+finalhhframe['cat2workers']
	return characteristics
	
def build_new_description(ini_pop_desc,ssp_pop,ssp,year,countrycode,shareag,sharemanu,shareemp,ischildren=False):
	'''builds a new description vector for the projected year, from ssp data and exogenous share for skilled people and agri people'''
	pop_tot,pop_0014,pop_1564,pop_65up,skilled_adults=get_pop_data_from_ssp(ssp_pop,ssp,year,countrycode)
	new_pop_desc                 = ini_pop_desc.copy()
	new_pop_desc['children']     = ini_pop_desc['children']
	if ischildren:
		new_pop_desc['children'] = pop_0014
	new_pop_desc['old']          = pop_65up
	new_pop_desc['skillworkers'] = skilled_adults*shareemp
	new_pop_desc['agworkers']    = pop_1564*shareag*shareemp
	new_pop_desc['manuworkers']  = pop_1564*sharemanu*shareemp
	new_pop_desc['servworkers']  = pop_1564*(1-shareag-sharemanu)*shareemp
	new_pop_desc['unemployed']   = pop_1564*(1-shareemp)
	return new_pop_desc,pop_0014
	
def build_new_weights(ini_pop_desc,future_pop_desc,characteristics,ini_weights,ismosek=True):
	'''optimize new weights to match current households and new population description'''
	t_tilde = cvxopt.matrix((future_pop_desc.values-ini_pop_desc.values).astype(np.float,copy=False))
	aa      = cvxopt.matrix(characteristics.values.astype(np.float,copy=False))
	w1      = 1/(ini_weights.values)**2
	n       = len(w1)
	P       = cvxopt.spdiag(cvxopt.matrix(w1))
	G       = -cvxopt.matrix(np.identity(n))
	h       = cvxopt.matrix(ini_weights.values.astype(np.float,copy=False))
	q       = cvxopt.matrix(0.0,(n,1))
	if ismosek:
		result = qp(P,q,G,h,aa.T,t_tilde.T,solver='mosek')['x']
	else:
		result = qp(P,q,G,h,aa.T,t_tilde.T)['x']
	if result is None:
		new_weights = 0*ini_weights
	else:
		new_weights = ini_weights+list(result)
	return new_weights
	
def find_new_weights(characteristics,ini_weights,future_pop_desc):
	ini_pop_desc = calc_pop_desc(characteristics,ini_weights)
	weights_proj = build_new_weights(ini_pop_desc,future_pop_desc,characteristics,ini_weights,ismosek=True)
	weights_proj = Series(np.array(weights_proj),index=ini_weights.index.values,dtype='float64')
	return weights_proj
	
def Y_from_inc(futurehhframe,inc):
	'''
	Calculates total hh income as the sum of each people's income in the household (estimated income)
	'''
	listofvariables = list(inc.index)
	out             = 0*futurehhframe['nbpeople']
	for var in listofvariables:
		out += inc[var]*futurehhframe[var]
	Ycalc           = Series(out,index=out.index)
	return Ycalc
	
def before_tax(inc,finalhhframe):
	'''Calculates the pre-tax revenues, assuming that elderly and unemployed incomes come from redistribution only. The error term (difference btw calculated and actual income) is included in the taxed revenue. We therefore calculate a pre-tax error term.
	Note: obsolete in the latest version of the model.
	'''
	inc_bf=inc.copy()
	errorterm=finalhhframe['totY']-Y_from_inc(finalhhframe,inc)
	gdpobserved=GDP(finalhhframe['totY'],finalhhframe['weight'])
	pensions=GDP(Y_from_inc(finalhhframe,inc[['old']]),finalhhframe['weight'])
	benefits=GDP(Y_from_inc(finalhhframe,inc[['cat7workers']]),finalhhframe['weight'])
	p=pensions/gdpobserved
	b=benefits/gdpobserved
	for thecat in range(1,7):
		string='cat{}workers'.format(int(thecat))
		inc_bf[string]=inc[string]*1/(1-p-b)
	inc_bf['old']=0
	inc_bf['cat7workers']=0
	errorterm=errorterm*1/(1-p-b)
	return b,p,errorterm,inc_bf
	
def keep_workers(inputs):
	thebool=(inputs.index!='old')&(inputs.index!='cat7workers')
	return thebool

	
def after_pensions(inc,errorterm,p,finalhhframe):
	'''
	Transfers income from workers to retirees. The error term is taxed also, only for households that are not only composed of unemployed or elderlies.
	'''
	inigdp = GDP(Y_from_inc(finalhhframe,inc)+errorterm,finalhhframe['weight'])
	select = ~((finalhhframe['cat7workers']+finalhhframe['old'])==(finalhhframe['nbpeople']-finalhhframe['children']))
	inc_af = inc.copy()
	for thecat in range(1,7):
		string         = 'cat{}workers'.format(int(thecat))
		inc_af[string] = inc[string]*(1-p)
	errorterm[select] = errorterm[select]*(1-p)
	totrev            = inigdp-GDP(Y_from_inc(finalhhframe,inc_af)+errorterm,finalhhframe['weight'])
	pensions          = totrev/sum(finalhhframe['old']*finalhhframe['weight'])
	inc_af['old']     = inc['old']+pensions
	return inc_af,errorterm
	
def after_bi(inc,errorterm,b,finalhhframe):
	'''Recalculates the after-basic-income incomes. All categories are taxed (including unemployed and retirees) and all adults receive the basic income. The error term is taxed also.'''
	inc_af  = inc.copy()
	gdpcalc = GDP(Y_from_inc(finalhhframe,inc)+errorterm,finalhhframe['weight'])
	bI      = b*gdpcalc/sum((finalhhframe['nbpeople']-finalhhframe['children'])*finalhhframe['weight'])
	for thecat in range(1,8):
		string         = 'cat{}workers'.format(int(thecat))
		inc_af[string] = inc[string]*(1-b)+bI
	inc_af['old'] = inc['old']*(1-b)+bI
	errorterm     = errorterm*(1-b)
	return inc_af,errorterm
	
def future_income_simple(inputs,year,finalhhframe,futurehhframe,inc,inimin,ini_year,shares_outside,temp_impact,price_increase):
	'''
	Projects the income of each household based on sectoral growth given by the scenario and the structure of the household.
	The error term grows like the average growth in the household, because we don't know what the unexplained income comes from.
	'''
	errorterm = finalhhframe['Y']*finalhhframe['nbpeople']-Y_from_inc(finalhhframe,inc)
	futureinc = inc.copy()
	b = inputs['b']

	p = inputs['p']
		
	futureinc['cat1workers'] = inc['cat1workers']*(1+inputs['grserv'])**(year-ini_year)
	futureinc['cat3workers'] = inc['cat3workers']*(1+inputs['grag'])**(year-ini_year)*(1+price_increase)
	futureinc['cat5workers'] = inc['cat5workers']*(1+inputs['grmanu'])**(year-ini_year)
	
	futureinc['cat2workers'] = futureinc['cat1workers']*inputs['skillpserv']                  
	futureinc['cat4workers'] = futureinc['cat3workers']*inputs['skillpag']
	futureinc['cat6workers'] = futureinc['cat5workers']*inputs['skillpmanu']
	
	futureinc = temperature_impact(futureinc,shares_outside,temp_impact)
	
	pure_income_gr           = Y_from_inc(futurehhframe,futureinc)/Y_from_inc(finalhhframe,inc)
	pure_income_gr.fillna(0, inplace=True)
	futurerrorterm           = pure_income_gr*errorterm
	futureinc,futurerrorterm = after_pensions(futureinc,futurerrorterm,p,futurehhframe)
	futureinc,futurerrorterm = after_bi(futureinc,futurerrorterm,b,futurehhframe)
	out                      = Y_from_inc(futurehhframe,futureinc)+futurerrorterm
	out[out<=0]              = inimin
	income_proj              = Series(out.values,index=out.index)/futurehhframe['nbpeople']
	return income_proj,futureinc
	
def future_income_simple_no_cc(inputs,year,finalhhframe,futurehhframe,inc,inimin,ini_year):
	'''
	Projects the income of each household based on sectoral growth given by the scenario and the structure of the household.
	The error term grows like the average growth in the household, because we don't know what the unexplained income comes from.
	'''
	errorterm = finalhhframe['Y']*finalhhframe['nbpeople']-Y_from_inc(finalhhframe,inc)
	futureinc = inc.copy()
	b = inputs['b']
	p = inputs['p']
		
	futureinc['cat1workers'] = inc['cat1workers']*(1+inputs['grserv'])**(year-ini_year)
	futureinc['cat3workers'] = inc['cat3workers']*(1+inputs['grag'])**(year-ini_year)
	futureinc['cat5workers'] = inc['cat5workers']*(1+inputs['grmanu'])**(year-ini_year)
	
	futureinc['cat2workers'] = futureinc['cat1workers']*inputs['skillpserv']                  
	futureinc['cat4workers'] = futureinc['cat3workers']*inputs['skillpag']
	futureinc['cat6workers'] = futureinc['cat5workers']*inputs['skillpmanu']
	
	pure_income_gr           = Y_from_inc(futurehhframe,futureinc)/Y_from_inc(finalhhframe,inc)
	pure_income_gr.fillna(0, inplace=True)
	futurerrorterm           = pure_income_gr*errorterm
	futureinc,futurerrorterm = after_pensions(futureinc,futurerrorterm,p,futurehhframe)
	futureinc,futurerrorterm = after_bi(futureinc,futurerrorterm,b,futurehhframe)
	out                      = Y_from_inc(futurehhframe,futureinc)+futurerrorterm
	out[out<=0]              = inimin
	income_proj              = Series(out.values,index=out.index)/futurehhframe['nbpeople']
	return income_proj,futureinc

			

def GDP(income,weights):
	GDP = np.nansum(income*weights)
	return GDP
	
def actual_productivity_growth(finalhhframe,inc,futurehhframe,futureinc,year,ini_year):
	
	out=list()
	
	for cat1,cat2 in (['cat1workers','cat2workers'],['cat3workers','cat4workers'],['cat5workers','cat6workers']):
		prod_ini  = (sum(finalhhframe[cat1]*finalhhframe['weight'])*inc[cat1]+sum(finalhhframe[cat2]*finalhhframe['weight'])*inc[cat2])/sum((finalhhframe[cat1]+finalhhframe[cat2])*finalhhframe['weight'])
		prod_last = (sum(futurehhframe[cat1]*futurehhframe['weight'])*futureinc[cat1]+sum(futurehhframe[cat2]*futurehhframe['weight'])*futureinc[cat2])/sum((futurehhframe[cat1]+futurehhframe[cat2])*futurehhframe['weight'])
		
		prod_gr   = (prod_last/prod_ini)**(1/(year-ini_year))-1
		out.append(prod_gr)
	return tuple(out)
			
def distrib2store(income,weights,nbdots,tot_pop):
	categories         = np.arange(0, 1+1/nbdots, 1/nbdots)
	y                  = np.asarray(wp(reshape_data(income),reshape_data(weights),categories,cum=False))
	inc_o              = (y[1::]+y[0:-1])/2
	o2store            = DataFrame(columns=['income','weights'])
	o2store['income']  = list(inc_o)
	o2store['weights'] = list([tot_pop/nbdots]*len(inc_o))
	return o2store
	
def indicators_from_pop_desc(ini_pop_desc):
	children      = ini_pop_desc['children']
	ag            = float(ini_pop_desc['agworkers'])
	manu          = float(ini_pop_desc['manuworkers'])
	serv          = float(ini_pop_desc['servworkers'])
	work          = ag+manu+serv
	adults        = work+float(ini_pop_desc['unemployed'])
	earn_income   = adults+float(ini_pop_desc['old'])
	tot_pop       = earn_income+children
	skilled       = float(ini_pop_desc['skillworkers'])
	
	shareemp_ini  = float(1-ini_pop_desc['unemployed']/adults)
	shareag_ini   = ag/work
	sharemanu_ini = manu/work
	
	share_skilled = skilled/adults
	
	return shareemp_ini,shareag_ini,sharemanu_ini,share_skilled
	
def scenar_ranges(ranges,finalhhframe,countrycode,ssp_gdp,codes_tables,ssp_pop,year,ini_year):
	'''
	This is a messy function at the moment. It sets the ranges of uncertainties. For redistribution (p and b) it is just a fixed range. For structural change, the ranges depend on the initial shares and are calculated in find_range_struct. For growth rates,xxx
	'''
	characteristics = keep_characteristics_to_reweight(finalhhframe)
	ini_pop_desc    = calc_pop_desc(characteristics,finalhhframe['weight'])
	shareemp_ini,shareag_ini,sharemanu_ini,share_skilled = indicators_from_pop_desc(ini_pop_desc)
	
	ag            = float(ini_pop_desc['agworkers'])
	manu          = float(ini_pop_desc['manuworkers'])
	serv          = float(ini_pop_desc['servworkers'])
	work          = ag+manu+serv
	adults        = work+float(ini_pop_desc['unemployed'])

	gr4=(get_gdp_growth(ssp_gdp,year,4,country2r32(codes_tables,countrycode),ini_year))**(1/(year-ini_year))-1
	gr5=(get_gdp_growth(ssp_gdp,year,5,country2r32(codes_tables,countrycode),ini_year))**(1/(year-ini_year))-1
	ssp_growth = np.mean([gr4,gr5])

	pop_tot,pop_0014,pop_1564_4,pop_65up,skilled_adults=get_pop_data_from_ssp(ssp_pop,4,year,countrycode)
	pop_tot,pop_0014,pop_1564_5,pop_65up,skilled_adults=get_pop_data_from_ssp(ssp_pop,5,year,countrycode)
	pop_growth = np.mean([(pop_1564_5/adults)**(1/(year-ini_year))-1,(pop_1564_5/adults)**(1/(year-ini_year))-1])
	
	ranges.ix['shareag',['min','max']]   = find_range_struct(shareag_ini,'ag')
	ranges.ix['sharemanu',['min','max']] = find_range_struct(sharemanu_ini,'ind')
	ranges.ix['shareemp',['min','max']]  = find_range_struct(shareemp_ini,'emp')
		
	select_gr=['grag','grmanu','grserv']
	ranges.ix[select_gr,['min','max']]=[ssp_growth-pop_growth-0.05,ssp_growth-pop_growth+0.01]
	
	if countrycode in ['ALB','BIH','BTN','CHN','DOM','ECU','EGY','FSM','GEO','JAM','KGZ','MAR','MDA','MKD','MNG','MOZ','NPL','PHL']:
		ranges.ix[select_gr,['min','max']]=[ssp_growth-pop_growth-0.06,ssp_growth-pop_growth]
	if countrycode in ['BDI','BDG','LBR']:
		ranges.ix[select_gr,['min','max']]=[ssp_growth-pop_growth-0.04,ssp_growth-pop_growth+0.03]
	if countrycode in ['TCD','ZMB']:
		ranges.ix[select_gr,['min','max']]=[ssp_growth-pop_growth-0.02,ssp_growth-pop_growth+0.04]
		
	ranges.ix[['skillpag','skillpserv','skillpmanu'],['min','max']] = [1,5]
	ranges.ix['p',['min','max']]=[0.001,0.2]
	ranges.ix['b',['min','max']]=[0.001,0.2]
	
	return ranges
	
	
def find_range_struct(ini_share,sector):
	x    = [0,0.01,0.1,0.3,0.5,0.7,0.9,1]
	if sector=='ag':
		ymin = [0,0.001,0.01,0.1,0.2,0.3,0.4,0.6]
		ymax = [0,0.2,0.15,0.4,0.5,0.6,0.8,0.8]
	elif sector=='ind':
		ymin = [0,0.1,0.15,0.1,0.2,0.3,0.35,0.4]
		ymax = [0,0.25,0.3,0.35,0.4,0.5,0.7,0.8]
	elif sector=='emp':
		ymin = [0,0.007,0.07,0.25,0.4,0.6,0.75,0.8]
		ymax = [0,0.4,0.5,0.6,0.7,0.9,0.99,1]
	w         = [1,2,2,2,1,1,1,1]
	smin      = UnivariateSpline(x, ymin, w)
	smax      = UnivariateSpline(x, ymax, w)
	range_out = [max(float(smin(ini_share)),0),min(float(smax(ini_share)),1)]
	return range_out
	
def correct_shares(shareag,sharemanu):
	if shareag+sharemanu>1:
		tot=shareag+sharemanu
		shareag=shareag/tot-0.001
		sharemanu=sharemanu/tot-0.001
	return shareag,sharemanu
	
def futurehh(finalhhframe,pop_0014,ischildren=False):
	'''
	ischildren is True if the number of children is taken into account in the re-weighting process. Otherwise, we re-scale the number of people based on the new number of children.
	'''
	futurehhframe=finalhhframe.copy()
	if not ischildren:
		futurehhframe['children']=finalhhframe['children']*pop_0014/sum(finalhhframe['children']*finalhhframe['weight'])
	futurehhframe['nbpeople']=finalhhframe['nbpeople']+futurehhframe['children']-finalhhframe['children']
	futurehhframe.drop(['Y','weight'], axis=1, inplace=True)
	return futurehhframe
	
def estime_income(hhcat,finalhhframe):
	'''
	Estimates income brought by each category of adults/elderly. The objective is not to have a good model of income but rather to find a starting point for making the income grow based on the household's composition. If the income of the unemployed or elderly is found negative, it is put equal to zero and we re-estimate.
	If the coefficients are non significant, we try different categories (by grouping existing categories) and keep the new coefficients only if they become significant.
	We ignore the richest 5%.
	'''
	select     = finalhhframe.Y<float(perc_with_spline(finalhhframe.Y,finalhhframe.weight*finalhhframe.nbpeople,0.95))
	X         = finalhhframe.ix[select,['cat1workers','cat2workers','cat3workers','cat4workers','cat5workers','cat6workers','cat7workers','old']].copy()
	X['serv'] = X['cat1workers']+X['cat2workers']
	X['ag']   = X['cat3workers']+X['cat4workers']
	X['manu'] = X['cat5workers']+X['cat6workers']
	X.drop(['cat1workers','cat2workers','cat3workers','cat4workers','cat5workers','cat6workers'],axis=1,inplace=True)
	result3   = sm.WLS(Y, X, weights=1/w).fit()
	a3        = result3.pvalues
	nonsign3  = a3[a3>0.05].index
	if (len(nonsign3)==0):
		inctemp            = result3.params
		inc['cat2workers'] = inctemp['serv']
		inc['cat4workers'] = inctemp['ag']
		inc['cat6workers'] = inctemp['manu']
		inc['cat1workers'] = inctemp['serv']
		inc['cat3workers'] = inctemp['ag']
		inc['cat5workers'] = inctemp['manu']
	else:
		X         = finalhhframe.ix[select,['cat1workers','cat2workers','cat3workers','cat4workers','cat5workers','cat6workers','cat7workers','old']].copy()
		X['ag']   = X['cat3workers']+X['cat4workers']
		X['nonag'] = X['cat1workers']+X['cat2workers']+X['cat5workers']+X['cat6workers']
		X.drop(['cat1workers','cat2workers','cat3workers','cat4workers','cat5workers','cat6workers'],axis=1,inplace=True)
		result2        = sm.WLS(Y, X, weights=1/w).fit()
		a2             = result2.pvalues
		nonsign2       = a2[a2>0.05].index
		inctemp           = result2.params
		inc['cat2workers']= inctemp['nonag']
		inc['cat4workers']= inctemp['ag']
		inc['cat6workers']= inctemp['nonag']
		inc['cat1workers']= inctemp['nonag']
		inc['cat3workers']= inctemp['ag']
		inc['cat5workers']= inctemp['nonag']
	return inc