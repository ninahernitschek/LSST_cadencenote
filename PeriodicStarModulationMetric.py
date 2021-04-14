from builtins import zip
import numpy as np
from lsst.sims.maf.metrics.baseMetric import BaseMetric
from scipy.optimize import curve_fit
from lsst.sims.maf.utils import m52snr
import warnings
import random

__all__ = ['periodicStar', 'PeriodicStarModulationMetric']

""" This metric is based on the sims_maf_contrib PeriodicStar metric (https://github.com/LSST-nonproject/sims_maf_contrib/blob/master/mafContrib/periodicStarMetric.py).
    It was modified in a way to reproduce attempts to identify period/ phase modulation (Blazhko effect)
    in RR Lyrae stars.    
    We are not implementing a period/ phase modulation in the light curve, but rather use short baselines (e.g.: 20 days) of observations to test how well we can recover the period, phase and amplitude. We do this as such an 
    attempt is also useful for other purposes, i.e. if we want to test whether we can just recover period, phase and amplitude from short baselines at all, without necessarily having in mind to look for period/ phase modulations.
    Like in the PeriodicStar metric, the light curve of an RR Lyrae star, or a periodic star in general, is approximated as a simple sin wave. Other solutions might make use of light curve templates to generate light curves.
    Two other modifications we introduced for the PeriodicStarModulationMetric are:
    In contrast to the PeriodicStar metric, we allow for a random phase offset to mimic observation starting at random phase.
Also, we vary the periods and amplitudes within +/- 10 % to allow for a more realistic sample of variable stars.
    """



class periodicStar(object):
    def __init__(self, filternames):
        self.filternames = filternames

    def __call__(self, t,x0,x1,x2,x3,x4,x5,x6,x7,x8):
        """ Approximate a periodic star as a simple sin wave.
        t: array with "time" in days, and "filter" dtype names.
        x0: Period (days)
        x1: Phase (days)
        x2: Amplitude (mag)
        x3: mean u mag
        x4: mean g mag
        x5: mean r mag
        x6: mean i mag
        x7: mean z mag
        x8: mean y mag
        """
        filter2index = {'u':3, 'g':4, 'r':5, 'i':6,
                        'z':7,'y':8}
        filterNames = np.unique(self.filternames)
        mags = np.zeros(t.size,dtype=float)
        mags = x2*np.sin((t+x1)/x0*2.*np.pi)
        x=[x0,x1,x2,x3,x4,x5,x6,x7,x8]
        for f in filterNames:
            good = np.where(self.filternames == f)
            mags[good] += x[filter2index[f]]
        return mags


class PeriodicStarModulationMetric(BaseMetric):
    """ At each slicePoint, run a Monte Carlo simulation to see how well a periodic source can be fit.
    Assumes a simple sin-wave light-curve, and generates Gaussain noise based in the 5-sigma limiting depth
    of each observation.
    Light curves are evaluated piecewise to test how well we can recover the period, phase and amplitude from shorter baselines.
    We allow for a random phase offset to mimic observation starting at random phase.
    Also, we vary the periods and amplitudes within +/- 10 % to allow for a more realistic sample of variable stars.
    """

        
    def __init__(self, metricName='PeriodicStarModulationMetric', mjdCol='observationStartMJD',
                 m5Col='fiveSigmaDepth', filterCol='filter', period=10., amplitude=0.5,
                 phase=2.,random_phase=False, time_interval=50,
                 nMonte=1000, periodTol=0.05, ampTol=0.10, means=[20.,20.,20.,20.,20.,20.],
                 magTol=0.10, nBands=3, seed=42, **kwargs):
        """
        period: days (default 10)
        amplitude: mags (default 1)
        phase: days (default 2.)
        random_phase: a random phase is assigned (default False)
        time_interval: days (default 50); the interval over which we want to evaluate the light curve
        dust: use dust reddening, needs mwdust (https://github.com/jobovy/mwdust) installed (default False)
        nMonte: number of noise realizations to make in the Monte Carlo
        periodTol: fractional tolerance on the period to demand for a star to be considered well-fit
        ampTol: fractional tolerance on the amplitude to demand
        means: mean magnitudes for ugrizy
        magTol: Mean magnitude tolerance (mags)
        nBands: Number of bands that must be within magTol
        seed: random number seed
        """
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        super(PeriodicStarModulationMetric, self).__init__(col=[self.mjdCol, self.m5Col,self.filterCol],
                                                 units='Fraction Detected',
                                                 metricName=metricName,**kwargs)
        self.period = period
        self.amplitude = amplitude
        self.time_interval = time_interval
        if(random_phase==True):
            self.phase = np.NaN
        else:
            self.phase = phase
            
        self.nMonte = nMonte
        self.periodTol = periodTol
        self.ampTol = ampTol
        self.means = np.array(means)
        self.magTol = magTol
        self.nBands = nBands
        np.random.seed(seed)
        self.filter2index = {'u':3, 'g':4, 'r':5, 'i':6, 'z':7,'y':8}
       
        
    def run(self, dataSlice, slicePoint=None):

        # Bail if we don't have enough points
        if dataSlice.size < self.means.size+3:
            return self.badval
        
        # Generate input for true light curve
        
        lightcurvelength = dataSlice.size
        
        np.set_printoptions(suppress=True) 
        
        t = np.empty(lightcurvelength, dtype=list(zip(['time','filter'],[float,'|U1'])))
        t['time'] = (dataSlice[self.mjdCol]-dataSlice[self.mjdCol].min())
        t['filter'] = dataSlice[self.filterCol]
        m5 = dataSlice[self.m5Col]
        
        
        lightcurvelength_days = self.time_interval
        
        # evaluate light curves piecewise in subruns
        subruns = int(np.max(t['time']) / lightcurvelength_days)
        
        print('number of subruns: ', subruns)
        fracRecovered_list=[]
        
        for subrun_idx in range(0,subruns):
        
            good = (    (t['time']>=(lightcurvelength_days*(subrun_idx))) & (t['time']<=    (lightcurvelength_days*(subrun_idx+1)))   )
                  
            t_subrun = t[good]
            m5_subrun = m5[good]
         
            if(t_subrun['time'].size>0):

                # If we are adding a distance modulus to the magnitudes
                if 'distMod' in list(slicePoint.keys()):
                    mags = self.means + slicePoint['distMod']
                else:
                    mags = self.means
                
                #slightly different periods and amplitudes (+/- 10 %) to mimic true stars
                #random phase offsets to mimic observation starting at random phase

                true_period=random.uniform(0.9,1.1)*self.period
                true_amplitude=random.uniform(0.9,1.1)*self.amplitude
                
                if(np.isnan(self.phase)): 
                    #a random phase (in days) should be assigned
                    true_phase=random.uniform(0,1)*self.period
                else:
                    true_phase = self.phase
                
                trueParams = np.append(np.array([true_period, true_phase, true_amplitude]), mags)
                true_obj = periodicStar(t_subrun['filter'])
                trueLC = true_obj(t_subrun['time'], *trueParams)

                # Array to hold the fit results
                fits = np.zeros((self.nMonte,trueParams.size),dtype=float)
                for i in np.arange(self.nMonte):
                    snr = m52snr(trueLC,m5_subrun)
                    dmag = 2.5*np.log10(1.+1./snr)
                    noise = np.random.randn(trueLC.size)*dmag
                    # Suppress warnings about failing on covariance
                    fit_obj = periodicStar(t_subrun['filter'])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # If it fails to converge, save values that should fail later
                        try:
                            parmVals, pcov = curve_fit(fit_obj, t_subrun['time'], trueLC+noise, p0=trueParams,
                                                       sigma=dmag)
                        except:
                            parmVals = trueParams*0+np.inf
                    fits[i,:] = parmVals

                # Throw out any magnitude fits if there are no observations in that filter
                ufilters = np.unique(dataSlice[self.filterCol])
                if ufilters.size < 9:
                    for key in list(self.filter2index.keys()):
                        if key not in ufilters:
                            fits[:,self.filter2index[key]] = -np.inf

                # Find the fraction of fits that meet the "well-fit" criteria
                periodFracErr = np.abs((fits[:,0]-trueParams[0])/trueParams[0])
                ampFracErr = np.abs((fits[:,2]-trueParams[2])/trueParams[2])
                magErr = np.abs(fits[:,3:]-trueParams[3:])
                nBands = np.zeros(magErr.shape,dtype=int)
                nBands[np.where(magErr <= self.magTol)] = 1
                nBands = np.sum(nBands, axis=1)
                nRecovered = np.size(np.where( (periodFracErr <= self.periodTol) &
                                               (ampFracErr <= self.ampTol) &
                                               (nBands >= self.nBands) )[0])
                
                
                fracRecovered = float(nRecovered)/self.nMonte
            
                #print('fracRecovered indiviual ', fracRecovered)
                
                fracRecovered_list.append(fracRecovered)
            

        fracRecovered = np.sum(fracRecovered_list)/(len(fracRecovered_list))

        #print('fracRecovered mean ', fracRecovered)
        return fracRecovered
    
    