import numpy as np 
import pandas as pandas 
import scipy.signal
import os
import sys
sys.path.append("/Users/duminghao/Research")
import Asteroseismology as se
import matplotlib.pyplot as plt 
from astropy.io import fits, ascii
from astropy.stats import LombScargle
from scipy.optimize import leastsq
import emcee
import corner

__all__ = ['Get_LC_Info', 'Data_Preparation', 'Power_Spectrum_Kepler', 'Background_Fit', 'Power_Spectrum_S4tess']

def Get_LC_Info(lc_path: str):

	se.tools.count_lc_files(lc_path, move_file=True)

def Outliers_Correction(data: float):

	data = Remove_None_Points(data)
	time = data[:,0]
	flux = data[:,1]
	
	# Remove Outlier
	two_points_dif = flux[1:] - flux[:-1]
	sigma = np.std(two_points_dif)
	index = np.argwhere(np.absolute(two_points_dif-np.mean(two_points_dif)) > 3*sigma)
	new_index = index + 1
	if index[0] == 0:
		new_index = np.insert(new_index, 0, 0)
	data[new_index,1] =  np.inf
	data = Remove_None_Points(data)

	return data

def Remove_None_Points(data: float):

	index1 = np.where(np.isnan(data))[0]
	index2 = np.where(np.isinf(data))[0]
	index = np.union1d(index1,index2)
	data = np.delete(data,index,0)

	return data
	

def Jumps_Drifts_Correction(data, period): # Period = 6 / 30 day for main sequences
	# Remove Jumps and Drifts by median high pass filter
	print('--- High Pass Filtering ---')
	bins = np.median(data[:,0][1:]-data[:,0][:-1])
	kernel = int(period/bins)
	yf = scipy.signal.medfilt(data[:,1],kernel)
	data[:,1] = data[:,1] / yf
	data[:,2] = data[:,2] / yf
	return data	


def Data_Preparation(lc_path: str, period: float=0.2, plot_lc: bool=False):

	print('--- Data Preparing ---')
	fn, fp = se.io.traverse_dir(lc_path+'LCs', 'fits')
	fn = sorted(fn, key = lambda fn:fn[14:26])
	# time = np.array([])
	# flux = np.array([])
	# flux_err =np.array([])
	o_data = np.array([])
	output = np.array([])
	quater = []
	start_t = []
	end_t = []
	for i in fn :
		hdulist = fits.open(lc_path+se.sep+'LCs'+se.sep+i)
		# header = hdulist[0].header
		data = hdulist[1].data
		header = hdulist[0].header
		quater.append('%s.%s' %(header['QUARTER'],header['SEASON']))
		# names = data.names
		temp1 = data['TIME']
		start_t.append(temp1[0])
		end_t.append(temp1[-1])
		temp2 = data['PDCSAP_FLUX']
		temp3 = data['PDCSAP_FLUX_ERR'] 
		temp = np.vstack((temp1,temp2,temp3)).T
		o_data = np.append(o_data, temp).reshape(-1,3)
		temp = Outliers_Correction(temp)
		output = np.append(output, temp).reshape(-1,3)

		# time = np.append(time,temp1)
		# flux = np.appedn(flux,temp2)
		# flux_err = np.append(flux_err,temp3)
		# output = np.vstack((time,flux,flux_err)).T
		# output = Outliers_Correction(output)
	print('--- Outputing ---')
	quarters = np.vstack((np.array(quater, dtype=float), np.array(start_t), np.array(end_t))).T
	np.savetxt(lc_path+'Quarter_Time.txt' , quarters, comments='#', 
		header='Quarter, Start_time, End_time')
	output = Jumps_Drifts_Correction(output,period)
	np.savetxt(lc_path+'Original_LC.txt' , o_data, comments='#', header='Time, Flux, Flux_err')
	np.savetxt(lc_path+'Correct_LC.txt' , output, comments='#', header='Time, Flux, Flux_err')
	if plot_lc == True:
		Plot_Light_Curve(lc_path, data, output, quarters=quarters)

	print('--- Data Prepared ---')
		
	return output



def Plot_Light_Curve(lc_path, o_data, c_data, quarters=None):

	fig = plt.figure()
	ax = fig.add_subplot(111)
	# ax.plot(o_data['TIME'],o_data['SAP_FLUX'])
	# ax.plot(o_data['TIME'],o_data['PDCSAP_FLUX'])
	ax.plot(c_data[:,0], c_data[:,1], color='black')
	if quarters is not None:
		for i in quarters[:,1]:
			ax.axvline(i, color='gray')
	plt.savefig(lc_path+'Light_Curves.png')
	plt.close()
	return print('--- Light Curve Plotted ---')

	
def Plot_PS(fp, freq, power, power_smooth=None, power_bg_ls=None, power_bg_mc=None, xlim=None, ptype=None):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(freq, power, color='gray')
	ax.plot(freq, power_smooth, color='red')
	if xlim is not None:
		ax.set_xlim((xlim[0],xlim[1]))
	if power_bg_ls is not None:
		ax.plot(freq, power_bg_ls, color='blue')
	if power_bg_mc is not None:
		ax.plot(freq, power_bg_mc, color='green')
	if ptype == 'log':
		ax.set_xscale('log')
		ax.set_yscale('log')

	plt.savefig(fp)

 
def Bg_Damping_Factor(freq, fnyq):       #kallinger 2014
	power = (np.sinc(np.pi/2 * freq/fnyq)) ** 2
	return power

def Super_Lorentz_Profile(freq, a, b):
	epsilon = 2 * np.sqrt(2) / np.pi
	return epsilon * a ** 2 * b / (1 + (freq/b)**2)

def Gauss_Profile(freq, height, fc, sigma):
	return height * np.exp(-(freq - fc)**2 / (2.0*sigma**2.0))

def Init_Bg_Model_Paras(freq, power, numax, teff, lc_type):
	if lc_type == 'sc':
		index = np.intersect1d(np.argwhere(freq > 7000.0),np.argwhere(freq < 8000.0))
	elif lc_type == 'lc':
		index = np.intersect1d(np.argwhere(freq > 270.0),np.argwhere(freq < 280.0))
	white_noise = np.median(power[index])
	a_sun = 3.59 / 1.33
	b1_sun = 758.0
	b2_sun = 24698.0
	# a1 = numax ** (-0.6) / se.numax_sun ** (-0.6) * a_sun
	# a2 = numax ** (-0.6) / se.numax_sun ** (-0.6) * a_sun
	# b1 = numax ** (0.970) / se.numax_sun ** (0.970) * b1_sun
	# b2 = numax ** (0.992) / se.numax_sun ** (0.992) * b2_sun
	a1 = 3382 * numax ** (-0.609)
	a2 = a1
	b1 = 0.317 * numax ** (0.970)
	b2 = 0.948 * numax ** (0.992)

	height = 3.49 * numax ** (-0.75) * teff * (3.5*0.75-2) / (se.numax_sun ** (-0.75) * se.teff_sun * (3.5*0.75-2))  # stello+ 2011
	dnu_guess = (numax/3050)**0.77 * 135.1 # Stello+2009
	sigma  = 2 * dnu_guess

	prior = np.array([white_noise, a1, b1, a2, b2, height, numax, sigma])
	print(prior)
	return prior

def BackGround_Model(freq, paras, fnyq, slp_number=2, type='withgauss'):         ##kallinger 2014 

	power_white_noise = paras[0]
	power_slp1 = Super_Lorentz_Profile(freq, paras[1], paras[2])
	power_slp2 = Super_Lorentz_Profile(freq, paras[3], paras[4])
	
	if slp_number == 2:
		power_slp = power_slp1 + power_slp2
	elif slp_number == 1:
		power_slp = power_slp1

	if type == 'withgauss':
		power_gauss = Gauss_Profile(freq, paras[5], paras[6], paras[7])
		power = power_slp + power_gauss
	elif type == 'withoutgauss':
		power = power_slp

	power *= Bg_Damping_Factor(freq, fnyq)
	power += power_white_noise
	return power

def smooth(x, window_len = 11, window = "hanning"):
	# stole from https://scipy.github.io/old-wiki/pages/Cookbook/SignalSmooth
	if x.ndim != 1:
		raise ValueError("smooth only accepts 1 dimension arrays.")
	if x.size < window_len:
		raise ValueError("Input vector needs to be bigger than window size.")
	if window_len < 3:
		return x
	if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
		raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

	s = x #np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
	if window == "flat":
		w = np.ones(window_len,"d")
	else:
		w = eval("np."+window+"(window_len)") 
	
	y = np.convolve(w/w.sum(),s,mode="same")
	return y

def SmoothWrapper(x, y, period, windowtype, samplinginterval):
	#samplinginterval = np.median(x[1:-1] - x[0:-2])
	xp = np.arange(np.min(x),np.max(x),samplinginterval)
	yp = np.interp(xp, x, y)
	window_len = int(period/samplinginterval)
	if window_len % 2 == 0:
		window_len = window_len + 1
	ys = smooth(yp, window_len, window = windowtype)
	yf = np.interp(x, xp, ys)
	return yf

def Power_Spectrum_Kepler(file_path: str, kicid:str, period: float, lc_type: str, 
	dnu: float, numax: float, teff: float, plotflag: bool=False):

	if lc_type == 'sc':
		fnyq = 8496.35
		fmax = fnyq / 1e6
	elif lc_type == 'lc':
		fnyq = 283.21
		fmax = fnyq / 1e6

	lc = np.loadtxt(file_path+se.sep+'Correct_LC.txt')
	quarter_info = np.loadtxt(file_path+se.sep+'Quarter_Time.txt')
	t = lc[:,0]
	t = t * 24.0 * 3600.0
	flux = lc[:,1]
	flux_err = lc[:,2] 
	dt = np.median(t[1:] - t[:-1])
	#time_bin = np.median(dt)
	
	fmin = 1.0 / period * 11.57 / 1e6
	res  = fmax / len(t)
	freq = np.arange(fmin, fmax, res)
	power = LombScargle(t, flux).power(freq,normalization='psd')
	freq *= 1e6
	power *= dt * 1e6 # to psd

	samplinginterval = np.median(freq[1:-1] - freq[0:-2])
	power_smooth = SmoothWrapper(freq, power, period, "bartlett", samplinginterval)
	if plotflag == True:
		Plot_PS(file_path+'%s_power.png' %kicid, 
			freq, power, power_smooth, power_bg_ls, power_bg_mc=None, xlim=None, ptype='log')

	ascii.write([freq, power], file_path+'%s.power' %kicid,
		format="no_header",delimiter=",",overwrite=True)

def wgn(x, snr):
    Ps = np.sum(abs(x)**2)/len(x)
    Pn = Ps/(10**((snr/10)))
    noise = np.random.chisquare(2,len(x)) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise


def Power_Spectrum_S4tess(fp: str, id_: str, lc: float, period: float, parameter: float, snr: float, plotflag: bool=False):

	psfig_path = fp + '/psfig'
	ps_path = fp + '/ps'
	dnu, numax, teff, gran_sig, gran_tau = parameter
	fnyq = 4166.67
	fmax = fnyq / 1e6

	t = lc['TIME']
	t = t * 24.0 * 3600.0
	flux = lc['FLUX']
	#flux_noise = flux * np.random.randn(len(flux))
	#flux_noise = wgn(flux, snr=2)
	#flux_noise = Add_noise(flux, noise, snr=2)
	#flux_err = lc[:,2] 
	dt = np.median(t[1:] - t[:-1])
	#time_bin = np.median(dt)
	
	fmin = 1.0 / period * 11.57 / 1e6
	res  = fmax / len(t)
	freq = np.arange(fmin, fmax, res)
	freq *= 1e6
	p_gran = (4 * gran_sig**2 * gran_tau) / (1+(2*np.pi*freq*gran_tau)**2)	
	samplinginterval = np.median(freq[1:-1] - freq[0:-2])
	index = (freq > numax*0.5) & (freq < numax*1.5)
	freq_cut = freq[index]


	power = LombScargle(t, flux).power(freq,normalization='psd')
	power *= dt * 1e6 # to psd
	power = power - p_gran
	power_cut = power[index]
	power_smooth = SmoothWrapper(freq, power, period, "bartlett", samplinginterval)
	power_smooth_cut = power_smooth[index]

	power_noise = wgn(power,snr)
	power_noise *= dt * 1e6
	power_noise = power_noise - p_gran
	power_noise_smooth = SmoothWrapper(freq, power_noise, period, "bartlett", samplinginterval)	
	power_noise_cut = power_noise[index]
	power_noise_smooth_cut = power_noise_smooth[index]

	# if plotflag == True:
	# 	Plot_PS(psfig_path+'/%s_power1.png' %id_, 
	# 		freq_cut, power_cut, power_smooth_cut, )#xlim=[numax*0.5,numax*1.5])

	# ascii.write([freq_cut, power_cut, power_smooth_cut], ps_path+'/%s.power' %id_,
	# 	format="no_header",delimiter=",",overwrite=True)
	if plotflag == True:
		Plot_PS(psfig_path+'/%s_power1.png' %id_, 
			freq_cut, power_noise_cut, power_noise_smooth_cut, )#xlim=[numax*0.5,numax*1.5])

	ascii.write([freq_cut, power_noise_cut, power_noise_smooth_cut], ps_path+'/%s.power' %id_,
		format="no_header",delimiter=",",overwrite=True)



def Background_Fit(file_path: str, kicid:str, lc_type: str, dnu: float, numax: float, teff: float, 
	ftype: str):
	
	# Fit the background
	if lc_type == 'sc':
		fnyq = 8496.35
		fmax = fnyq / 1e6
	elif lc_type == 'lc':
		fnyq = 283.21
		fmax = fnyq / 1e6

	data = np.loadtxt(file_path+se.sep+'%s.power' %kicid, comments='#' )#delimiter=',')
	freq = data[:,0]
	power = data[:,1]
	period = dnu / 15.0
	samplinginterval = np.median(freq[1:-1] - freq[0:-2])
	power_smooth = SmoothWrapper(freq, power, period, "bartlett", samplinginterval)
	bgparas_guess = Init_Bg_Model_Paras(freq, power_smooth, numax, teff, lc_type)
	
	print('--- Fitting Background ---')
	if ftype == 'LS':
		
		def residuals_bg(paras):
			return power - BackGround_Model(freq, paras, fnyq, type='withoutgauss')
		
		bgpara, cov = leastsq(residuals_bg, bgparas_guess, maxfev=3000)
		power_bg = BackGround_Model(freq, bgpara, fnyq, type='withoutgauss')
		power_bg_with_gauss = BackGround_Model(freq, bgpara, fnyq, type='withgauss')
		np.savetxt(file_path+'%s_bg_ls.para' %kicid, bgpara)

	elif ftype == 'MC':
		if os.path.exists(file_path+'%s_bg_ls.para' %kicid):
			theta = np.loadtxt(file_path+'%s_bg_ls.para' %kicid)
			theta = theta[:5]		
		else:
			theta = bgparas_guess[:5]
			power_bg_ls = None


		def lnlikelihood(theta, freq, power, fnyq):
			model = BackGround_Model(freq, theta, fnyq, slp_number=2, type='withoutgauss')
			return - np.sum(np.log(model)+power/model)	
		
		def lnprior(theta):
			w, a1, b1, a2, b2 = theta
			w_l = bgparas_guess[0] * 0.5
			w_u = bgparas_guess[0] * 1.5
			# a1_l, a1_u = 0, 100
			# a2_l, a2_u = 0, 100
			a1_l = bgparas_guess[1] * 0.5#0.1
			a1_u = bgparas_guess[1] * 1.5#10
			a2_l = bgparas_guess[3] * 0.5#0.1
			a2_u = bgparas_guess[3] * 1.5# 10
			b1_l = bgparas_guess[2] * 0.5#0.1
			b1_u = bgparas_guess[2] * 1.5# 10
			b2_l = bgparas_guess[4] * 0.5#0.1
			b2_u = bgparas_guess[4] * 1.5# 10

			if w_l < w < w_u and a1_l < a1 < a1_u and a2_l < a2 < a2_u and b1_l < b1 < b1_u and b2_l < b2 < b2_u :
				return 0.0
			else:
				return -np.inf

		def lnprob(theta, freq, power, fnyq):
			lpri = lnprior(theta)
			if not np.isfinite(lpri):
				return -np.inf
			else:
				return lpri + lnlikelihood(theta, freq, power, fnyq)

		print('--- MCMC ---')
		ndim, nwalkers = 5, 300
		pos0 = [theta + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(freq,power,fnyq))
		nburn, nsteps = 200, 2000
		width = 30
		#result = sampler.sample(pos0, iterations=nburn, thin=10)
		for j, result in enumerate(sampler.sample(pos0, iterations=nburn, thin=10)):
			n = int((width+1) * float(j) / nburn)
			sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
		sys.stdout.write("\n")

		pos, lnpost, rstate = result
		sampler.reset()

		#result = sampler.sample(pos, nsteps, lnprob0=lnpost)
		for j, result in enumerate(sampler.sample(pos, iterations=nsteps, lnprob0=lnpost)):
				#pos, lnpost, rstate = result
			n = int((width+1) * float(j) / nsteps)
			sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
		sys.stdout.write("\n")
		samples = sampler.chain[:,nburn:,:].reshape((-1,ndim)) #.chain is of shape (nwalker, nsteps, ndim)

		# evidence = sampler.thermodynamic_integration_log_evidence()
		# np.savetxt(file_path+"evidence_h1_%s.txt" %kicid, evidence, delimiter=",", fmt=("%0.8f"), header="bayesian_evidence") 

		result_mcmc = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
			zip(*np.percentile(samples, [16, 50, 84],axis=0)))))
		np.savetxt(file_path+"%s_bg_mc.para" %kicid, result_mcmc)

		label = ['w','a1','b1','a2','b2']
		fig = corner.corner(samples, label=label, quantiles=(0.16,0.5,0.84), truths=result_mcmc[:,0])
		fig.savefig(file_path+'corner_%s.png' %kicid)
		plt.close()                           

		power_bg_ls = BackGround_Model(freq, theta, fnyq, slp_number=2, type='withoutgauss')
		power_bg_mc = BackGround_Model(freq, result_mcmc[:,0], fnyq, slp_number=2, type='withoutgauss')

		Plot_PS(file_path+'%s_power.png' %kicid, freq, power, power_smooth, power_bg_ls, power_bg_mc, ptype = 'log')
		xx = power - power_bg_mc
		np.savetxt('./ps_without_bg.power', xx)

