# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:09:03 2015

@author: elinor
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import re
from astropy import units as u
import astropy.table as t

def radius(x,y,center,scale=1,e=0,theta_e=0):
    x = x.copy(); y = y.copy()
    x = (x-center[0])*scale
    y = (y-center[1])*scale
    xtag = x*np.cos(theta_e)+y*np.sin(theta_e)
    ytag = -x*np.sin(theta_e)+y*np.cos(theta_e)
    r = np.sqrt(xtag**2*(1.0-e) + ytag**2/(1.0-e))
    return r
def radius_onsky(a,d,ac,dc):
    r = np.arccos(np.cos(d.to(u.rad))*np.cos(dc.to(u.rad))*np.cos(a.to(u.rad)-ac.to(u.rad))+np.sin(d.to(u.rad))*np.sin(dc.to(u.rad)))
    #r.unit=u.rad
    return r.to(u.arcmin)

    
def im2arcmin(x,center,scale):
    """r = im2arcmin(x,center,scale)"""
    r = (x-center) * scale
    return r

def arcmin2im(r,center,scale):
    """r = im2arcmin(x,center,scale)"""
    x = r/scale + center
    return x

####### BINNING ###############    
def logBins(minr, maxr, nbinr):
    l_bin = np.logspace(np.log10(minr), np.log10(maxr), nbinr+1)
    l_bin_min = l_bin[:-1]
    l_bin_max = l_bin[1:]
    l_bin_center = 2.*(l_bin_max**3 - l_bin_min**3)/(3.*(l_bin_max**2 - l_bin_min**2)) # area average
    return l_bin_min, l_bin_max, l_bin_center
   
def logBinsHarmonic(minr, maxr, nbinr):
    l_bin = np.logspace(np.log10(minr), np.log10(maxr), nbinr+1)
    l_bin_min = l_bin[:-1]
    l_bin_max = l_bin[1:]
    l_bin_center = (l_bin_max + l_bin_min) / 2.0
    return l_bin_min, l_bin_max, l_bin_center

def Wbinning(x, val, err=0, w=None, minr=0, maxr=1e10, nbinr=10):
    """ log-spaced  binned weighted average"""
    bins = np.logspace(np.log10(minr), np.log10(maxr), nbinr+1)
    bin_min, bin_max, bincenter = logBinsHarmonic(minr, maxr, nbinr)
    valbin, binedge = np.histogram( x, weights = np.nan_to_num(val*w), bins=bins)
    val2bin, binedge = np.histogram( x, weights = np.nan_to_num((val*w)**2), bins=bins)
    wbin,  binedge = np.histogram( x, weights = w, bins=bins)
    w2bin,  binedge = np.histogram( x, weights = w**2, bins=bins)
    wsig2bin, binedge = np.histogram( x, weights = np.nan_to_num(err**2*w**2), bins=bins) 
    binerr2 = np.sqrt(val2bin)/wbin
    binvalw = valbin/wbin
    binerr = np.sqrt(wsig2bin)/wbin
    Nwbin = wbin**2/w2bin
    STD = np.sqrt(val2bin/wbin - binvalw**2)
    return binvalw , binerr, binerr2, bincenter, bin_min, bin_max, Nwbin, STD, wbin



def Wbinning2(x, val, err=0, w=None, minr=0, maxr=1e10, nbinr=10):
    """ log-spaced  binned weighted average"""
    bins = np.logspace(np.log10(minr), np.log10(maxr), nbinr+1)
    bin_min, bin_max, bincenter = logBinsHarmonic(minr, maxr, nbinr)
    valbin, binedge = np.histogram( x, weights = np.nan_to_num(val*w), bins=bins)
    val2bin, binedge = np.histogram( x, weights = np.nan_to_num((val*w)**2), bins=bins)
    wbin,  binedge = np.histogram( x, weights = w, bins=bins)
    w2bin,  binedge = np.histogram( x, weights = w**2, bins=bins)
    wsig2bin, binedge = np.histogram( x, weights = np.nan_to_num(err**2*w**2), bins=bins) 
    binerr2 = np.sqrt(val2bin)/wbin
    binvalw = valbin/wbin
    binerr = np.sqrt(wsig2bin)/wbin
    #Nwbin = wbin**2/w2bin
    Nbin, binedge = np.histogram( x,  bins=bins) 
    #STD = np.sqrt(val2bin/wbin - binvalw**2)
    return binvalw , binerr, binerr2, bincenter, bin_min, bin_max, Nbin, wbin,w2bin

def Wbinning_flex(x, val, err=0, w=None, bins=None):
    """ input bins  binned weighted average"""
    valbin, binedge = np.histogram( x, weights = np.nan_to_num(val*w), bins=bins)
    val2bin, binedge = np.histogram( x, weights = np.nan_to_num((val*w)**2), bins=bins)
    wbin,  binedge = np.histogram( x, weights = w, bins=bins)
    w2bin,  binedge = np.histogram( x, weights = w**2, bins=bins)
    wsig2bin, binedge = np.histogram( x, weights = np.nan_to_num(err**2*w**2), bins=bins) 
    binerr2 = np.sqrt(val2bin)/wbin
    binvalw = valbin/wbin
    binerr = np.sqrt(wsig2bin)/wbin
    #Nwbin = wbin**2/w2bin
    Nbin, binedge = np.histogram( x,  bins=bins) 
    #STD = np.sqrt(val2bin/wbin - binvalw**2)
    return binvalw , binerr, binerr2, Nbin, wbin,w2bin


def Wbinning_bserr(x, val, err=0, w=None, minr=0, maxr=1e10, nbinr=10,Nbs=100):
    """ log-spaced  binned weighted average with boostrap errors"""
    import astropy.table as t
    bins = np.logspace(np.log10(minr), np.log10(maxr), nbinr+1)
    bin_min, bin_max, bincenter = logBinsHarmonic(minr, maxr, nbinr)
    valbin, binedge = np.histogram( x, weights = np.nan_to_num(val*w), bins=bins)
    val2bin, _ = np.histogram( x, weights = np.nan_to_num((val*w)**2), bins=bins)
    wbin,  _ = np.histogram( x, weights = w, bins=bins)
    w2bin,  _ = np.histogram( x, weights = w**2, bins=bins)
    binvalw = valbin/wbin
    binerr2 = np.sqrt(val2bin)/wbin # regular error
    binvalw_bs = np.ones([Nbs,nbinr])
    for i in range(Nbs):
        bs = np.random.choice(t.Table([x,val,w],names=('x','val','w')),size=len(x),replace=True)
        xbs = bs['x']; valbs = bs['val']; wbs = bs['w']
        valbin_i, _ = np.histogram( xbs, weights = np.nan_to_num(valbs*wbs), bins=bins)
        wbin_i,  _ = np.histogram( xbs, weights = wbs, bins=bins)
        binvalw_bs[i] = valbin_i/wbin_i
    binerr = np.std(binvalw_bs,axis=0) # bootstrap errors
    #Nwbin = wbin**2/w2bin
    Nbin, _ = np.histogram( x,  bins=bins) 
    #STD = np.sqrt(val2bin/wbin - binvalw**2)
    return binvalw , binerr, binerr2, bincenter, bin_min, bin_max, Nbin, wbin,w2bin


def Wbinning_tozero(x, val, err=0, w=None, minr=0, maxr=1e10, nbinr=10):
    """ log-spaced  binned weighted average"""
    bins = np.logspace(np.log10(minr), np.log10(maxr), nbinr+1)
    bins[0]=0
    #bin_min, bin_max, bincenter = logBins(minr, maxr, nbinr)
    bin_min = bin[:-1]
    bin_max = bin[1:]
    bincenter = 2.*(bin_max**3 - bin_min**3)/(3.*(bin_max**2 - bin_min**2)) # area average

    valbin, binedge = np.histogram( x, weights = np.nan_to_num(val*w), bins=bins)
    val2bin, binedge = np.histogram( x, weights = np.nan_to_num((val*w)**2), bins=bins)
    wbin,  binedge = np.histogram( x, weights = w, bins=bins)
    w2bin,  binedge = np.histogram( x, weights = w**2, bins=bins)
    wsig2bin, binedge = np.histogram( x, weights = np.nan_to_num(err**2*w**2), bins=bins) 
    binerr2 = np.sqrt(val2bin)/wbin
    binvalw = valbin/wbin
    binerr = np.sqrt(wsig2bin)/wbin
    #Nwbin = wbin**2/w2bin
    Nbin, binedge = np.histogram( x,  bins=bins) 
    #STD = np.sqrt(val2bin/wbin - binvalw**2)
    return binvalw , binerr, binerr2, bincenter, bin_min, bin_max, Nbin, wbin,w2bin

def Wsum(x, val, err=0, w=None, minr=0, maxr=1e10, nbinr=10):
    """ log-spaced  binned weighted average"""
    bins = np.logspace(np.log10(minr), np.log10(maxr), nbinr+1)
    #bin_min, bin_max, bincenter = logBinsHarmonic(minr, maxr, nbinr)
    valbin, binedge = np.histogram( x, weights = np.nan_to_num(val*w), bins=bins)
    val2bin, binedge = np.histogram( x, weights = np.nan_to_num((val*w)**2), bins=bins)
    wbin,  binedge = np.histogram( x, weights = w, bins=bins)
    w2bin,  binedge = np.histogram( x, weights = w**2, bins=bins)
    wsig2bin, binedge = np.histogram( x, weights = np.nan_to_num(err**2*w**2), bins=bins)
    Nbin, binedge = np.histogram( x,  bins=bins)    
    #return bincenter, valbin , val2bin, wbin, w2bin, wsig2bin, Nbin
    return valbin , val2bin, wbin, w2bin, wsig2bin, Nbin

def WMbin(x, val,  w=None, minr=0, maxr=1e10, nbinr=10):
    """ log-spaced  binned weighted average"""
    bins = np.logspace(np.log10(minr), np.log10(maxr), nbinr+1)
    #bin_min, bin_max, bincenter = logBinsHarmonic(minr, maxr, nbinr)
    valbin, binedge = np.histogram( x, weights = np.nan_to_num(val*w), bins=bins)
    wbin,  binedge = np.histogram( x, weights = w, bins=bins)
    return valbin/wbin
def WMbinFlex(x, val,  w=None, bins=None):
    """ log-spaced  binned weighted average"""
    if bins is None:
        bins = np.logspace(np.log10(0.2), np.log10(5), 10+1)
    #bin_min, bin_max, bincenter = logBinsHarmonic(minr, maxr, nbinr)
    valbin, binedge = np.histogram( x, weights = np.nan_to_num(val*w), bins=bins)
    wbin,  binedge = np.histogram( x, weights = w, bins=bins)
    return valbin/wbin

     
def Wbinning_lin(x, val, err=0, w=1, minr=0, maxr=1e10, nbinr=10):
    """ lin-spaced  binned weighted average"""
    bins = np.linspace(minr, maxr, nbinr+1)
    bincenter = (bins[1:] + bins[:-1])/2
    valbin, binedge = np.histogram( x, weights = np.nan_to_num(val*w), bins=bins)
    val2bin, binedge = np.histogram( x, weights = np.nan_to_num((val*w)**2), bins=bins)
    wbin,  binedge = np.histogram( x, weights = w, bins=bins)
    w2bin,  binedge = np.histogram( x, weights = w**2, bins=bins)
    wsig2bin, binedge = np.histogram( x, weights = np.nan_to_num(err**2*w**2), bins=bins)
    Nbin, binedge = np.histogram( x,  bins=bins)    
    binerr2 = np.sqrt(val2bin)/wbin
    binvalw = valbin/wbin
    binerr = np.sqrt(wsig2bin)/wbin
    Nwbin = wbin**2/w2bin
    return binvalw , binerr, binerr2, bincenter, Nwbin


    
def Circles(ax, x, y, r, **kwargs):
    phi = np.linspace(0.0,2*np.pi,100)
    if np.isscalar(r):
        r = r * np.ones(x.shape)
    na = np.newaxis
    
    # the first axis of these arrays varies the angle, 
    # the second varies the circles
#    x_line = x[na,:]+r[na,:]*np.sin(phi[:,na])
#    y_line = y[na,:]+r[na,:]*np.cos(phi[:,na])
    x_line = x+r*np.sin(phi[:,na])
    y_line = y+r*np.cos(phi[:,na])
    
    h = ax.plot(x_line,y_line,'-', **kwargs)
    return h, x_line, y_line
        
    
def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()
    
def linear_fit(cat):
    """ function [a]=linear_fit(cat)
    % if cat given, plots, mark upper left and lower right & calculate, 
    % else (empty) mark 2 points for line on existing plot
    % cat=[] or [mag color]
    """
    cat = np.array(cat)
    if cat.size == 0:
        
        tellme('select two points along the line')
        plt.waitforbuttonpress()
        p1, p2 = plt.ginput(2, timeout=-1)
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0]) #slope a
        intercept = p1[1] - slope*p1[0] #cutoff b
        return slope, intercept
    


def savemat(matname, variables):
    sio.savemat(matname, dict( (name,eval(name)) for name in variables))


def localMaxima(data,sep,threshold,ax=None,**kwargs):
    import scipy.ndimage as simage
    data_max = simage.filters.maximum_filter(data, sep)
    maxima = (data == data_max)
    diff = (data_max  > threshold)
    maxima[diff == 0] = 0 #?
    row, column = np.where(maxima)
    if ax is not None:
        #ax.imshow(data, origin='lower')    
        ax.scatter(column,row, s=50, c='k', facecolors='none', **kwargs )
        plt.show()
    return row, column

def z_Pz(z=None, Pz=1.):
    if z is None:
        z = np.arange(0,7.01,0.01)
    return np.sum(Pz * z,axis=1) / np.sum(Pz,axis=1)

#def z_hist(z=None, Pz=1., bins=np.arange(0,4,0.05), **kwargs):
#    zpz = z_Pz(z=z, Pz=Pz)
#    n,x, = plt.hist(zpz,bins, **kwargs)
def zhist_PDF(z=np.arange(0,7.01,0.01), Pz=1., bins=np.arange(0,4,0.05), **kwargs):
    zmat = np.tile(z,[len(Pz),1])
    zmatflat = zmat.flatten()
    Pzflat = np.array(Pz).flatten()
    plt.hist(zmatflat,weights=Pzflat,bins=bins,histtype='stepfilled', normed=True, **kwargs)
def zhist_weighted(z=np.arange(0,7.01,0.01), Pz=1., bins=np.arange(0,4,0.05), **kwargs):
    zmat = np.tile(z,[len(Pz),1])
    zmatflat = zmat.flatten()
    Pzflat = np.array(Pz).flatten()
    plt.hist(zmatflat,weights=Pzflat,bins=bins,histtype='stepfilled', normed=True, **kwargs)
    
def sss(x):
    y = np.sqrt(np.nansum(np.square(x)))
    return y    
def SNR(x,cov):
    return np.sqrt(np.vdot(np.dot(x,x.T),np.linalg.inv(cov)))
    #return np.sqrt(np.sum(np.dot(x,x.T)/cov))
def SNR2(a,cova): # explicit??
    SNR = np.zeros([len(a),len(a)])
    Cinv = np.linalg.inv(cova)
    for i in range(len(a)):
        for j in range(len(a)):
            SNR[i,j] = a[i]*Cinv[i,j]*a[j]
    return np.sqrt(np.nansum(SNR))

    
def wmean(x,var):
    w = 1./var
    mean = np.vdot(x,w)/np.sum(w)
    err = np.sqrt(1./np.sum(w))
    return mean, err

def wmean_cov(a,cova):
    W = np.linalg.inv(cova)
    mean = np.sum(np.dot(W,a))/np.sum(W)    
    err =   np.sqrt( 1./(np.sum(W)) )
    return mean, err
def wmean_varcov(a,cova):
    var = np.diag(cova).reshape([len(a),1])
    w = 1./var
    mean = np.sum(np.vdot(w,a))/np.sum(w)    
    err =   np.sqrt( np.vdot(np.dot(w,w.T),cova) / np.sum(w)**2 )
    return mean, err
    
def ceil_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1

def floor_to_odd(f):
    return np.floor(f) // 2 * 2 + 1

def floor_to_even(f):
    return np.floor(f) // 2 * 2 
    
def ceil_to_even(f):
    return np.ceil(f) // 2 * 2 
    
def ismember(A,B):
    B_unique_sorted, B_idx = np.unique(B, return_inverse=True)
    A_in_B_bool = np.in1d(A,B_unique_sorted, assume_unique=True)
    return A_in_B_bool
    
def issorted(l):
    return all(l[i] <= l[i+1] for i in xrange(len(l)-1))
    
    
def randDist(X,bins,P):
    """ randomly select without replacements subsample Y from X (where X has non-uniform distribution R(X))
    with distribution described by P(X)  
    resolution defined by bins
    returns Y"""
    # current distribution
    # hist in
    Rx,_ = np.histogram(X,bins=bins)
    # hist in*out
    Px,_ = np.histogram(X,bins=bins,weights=P)
    # multiply by 1/input distribution
    Px = Px /Rx  
    # peak normalization
    imax = (Px == np.max(Px))
    Px = Px * Rx[imax]/Px[imax]
    Px = Px/np.max(Px[Px>Rx]/Rx[Px>Rx])
    #Y = np.empty([len(Px),1])
    Y = np.array([])
    for i in range(len(Px)):
        Xi = X[ (X>=bins[i]) & (X<bins[i+1])]        
        Y = np.append(Y,np.random.choice(Xi, replace=False,size=np.floor(Px[i])*0.9))
    return Y
def matchDistribution(sample,X,X2,bins):
    """ randomly select subsample from sample so that its variable X match the distribution of variable X2"""
    Rx,_ = np.histogram(X2,bins=bins)
    Y = sample[0:0] 
    for i in range(len(bins)-1):
        Xi = sample[ (X>=bins[i]) & (X<=bins[i+1])]    
        if (len(Xi)>0) & (Rx[i]>0):
            print len(Xi)
            print(Rx[i])
            Y = t.vstack([Y,t.Table(np.random.choice(Xi, replace=False,size=Rx[i]))])
    return Y
    

def get_ellipse_coords(a=0.0, b=0.0, x=0.0, y=0.0, angle=0.0, k=2):
    """ Draws an ellipse using (360*k + 1) discrete points; based on pseudo code
    given at http://en.wikipedia.org/wiki/Ellipse
    k = 1 means 361 points (degree by degree)
    a = major axis distance,
    b = minor axis distance,
    x = offset along the x-axis
    y = offset along the y-axis
    angle = clockwise rotation [in degrees] of the ellipse;
        * angle=0  : the ellipse is aligned with the positive x-axis
        * angle=30 : rotated 30 degrees clockwise from positive x-axis
    """
    pts = np.zeros((360*k+1, 2))

    beta = -angle * np.pi/180.0
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    alpha = np.radians(np.r_[0.:360.:1j*(360*k+1)])
 
    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)
    
    pts[:, 0] = x + (a * cos_alpha * cos_beta - b * sin_alpha * sin_beta)
    pts[:, 1] = y + (a * cos_alpha * sin_beta + b * sin_alpha * cos_beta)

    return pts
    
def RegexList(list,regex):
    """ RegexList(list,regex) - return lines from a list (or array) that contain regular expression"""
    filter = re.compile('(' + regex + ')').search
    return [l for l in list for m in (filter(l),) if m]
    
    
def table2columns(table):
    cols=[]
    for col in table.colnames:
        cols.append(table[col])
    return cols
    
def processStats():
    """Collect Linux-specific process statistics

    Parses the /proc/self/status file (N.B. Linux-specific!) into a dict
    which is returned.
    """
    import os
    result = {}
    if os.path.isfile('/proc/self/status'):
        with open("/proc/self/status") as f:
            for line in f:
                key, _, value = line.partition(":")
                result[key] = value.strip()
        return result
    else:
        return None
    
def Ndecimal(num):
    import re
    return len(re.split('\.',np.str(num))[-1])
    
def plotCov(resample,minbin=1,maxbin=None):
    """ plot correlation matrix for sample"""
    from matplotlib.ticker import ScalarFormatter
    if maxbin is None:
        maxbin = len(resample)
    nbin = resample.shape[0]
    nsub = resample.shape[1]
    redges = np.logspace(np.log10(minbin),np.log10(maxbin),nbin)
    corrcoeff = np.corrcoef(resample) # covariance from xi_sim
    
    ax=plt.subplot(111)
    
    palette = plt.cm.coolwarm
    palette.set_bad('k', 1.)
    plt.pcolor(redges,redges,corrcoeff,cmap=palette)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$r$ [$h^{-1}$Mpc]')
    plt.ylabel(r'$r$ [$h^{-1}$Mpc]')
    plt.clim([-1,1])
    
    ax.set_xticks(np.arange(minbin,maxbin,nbin))
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_yticks(np.arange(minbin,maxbin,nbin))
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    cb=plt.colorbar()
    cb.set_label('R')
    plt.tight_layout()
    plt.axis('equal')
    plt.axis('tight')
    return ax
    
def conf_int(sample, p=0.68, mean=None):
    if mean is None:
        mean = np.mean(sample,axis=0)
    sort = np.sort(sample,axis=0)
    M = len(sort)/2.
    low = mean - sort[int(np.round(M - M*p))]
    high = sort[int(np.round(M + M*p))] - mean
    return low, high
    
    
def skyScatter(ra,dec,orig=180,ax=None,**kwargs):
    """ scatter plot of RA/DEC on a mollweide sky projection
    RA/DEC given in degrees 
    orig specifies central RA
    ax if to overlay on axisting axes
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    if ax is None:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection="mollweide")
    coord = SkyCoord( ra=ra,dec=dec,  unit=(u.deg,u.deg) ) # added for convenience
    ra = coord.ra.wrap_at(180*u.degree)
    dec=coord.dec
    x = np.remainder(ra.deg+360-orig,360) # shift RA values
    ind = x>180
    x[ind] -=360    # scale conversion to [-180, 180]
    x=-x
    sch = ax.scatter(np.radians(x), dec.radian,**kwargs)
    #ax.set_xticklabels(['14h','16h','18h','20h','22h','0h','2h','4h','6h','8h','10h'])
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels+360+orig,360)
     
    ax.grid(True)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("RA [deg]")
    ax.xaxis.label.set_fontsize(12)
    ax.set_ylabel("Dec [deg]")
    ax.yaxis.label.set_fontsize(12)
    plt.legend(bbox_to_anchor=(1.05, 1.2), loc='upper right')
    return ax,sch

def gnomProj(alpha, delta, alpha0, delta0, reso, xdim, ydim):
    """ Hironao's projected map"""
    alpha_rad = np.radians(alpha)
    delta_rad = np.radians(delta)
    alpha0_rad = np.radians(alpha0)
    delta0_rad = np.radians(delta0)
    cosc = np.sin(delta0_rad)*np.sin(delta_rad) + np.cos(delta0_rad)*np.cos(delta_rad)*np.cos(alpha_rad - alpha0_rad)
    x = np.cos(delta_rad)*np.sin(alpha_rad - alpha0_rad)/cosc
    y = (np.cos(delta0_rad)*np.sin(delta_rad) - np.sin(delta0_rad)*np.cos(delta_rad)*np.cos(alpha_rad - alpha0_rad))/cosc
    x = -x/(reso*np.pi/180./60.) + 0.5*xdim
    y = y/(reso*np.pi/180./60.) + 0.5*ydim
    return x, y
    
def readCovUmetsu(file,N=10):
    cov = np.loadtxt(file)
    cmat = cov[:,2].reshape([N,N])
    return cmat