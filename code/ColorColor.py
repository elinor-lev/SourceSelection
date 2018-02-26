# -*- coding: utf-8 -*-
"""
Utilities for selecting background sources for clean cluster weak lensing measurement
-------------------------------------
Created on Mon Nov 23 16:38:44 2015
Last edited: Feb 26 2018

@author: elinor


"""
import corner
import numpy as np
import matplotlib.pyplot as plt
import useful_el as usel
import plotting as epl
import astropy.table as table 
import scipy.io as sio
import WL
import astropy.units as u

def CCplot(table,r,g1,g2,b, CCparams=[0,0], levels=[0.683,0.955], ext='mag_forced_cmodel', 
           axis=None, plotpoints=True,plotcontours=False,plotCCseq=False, contour_args=None,  **kwargs):
    """ plot color-color diagram for a sample of galaxies given in table. 
    y-axis is b-g2 color, x-axis is g1-r colors, where r,g1,g2,b define the 
    filter column name first letter (e.g., 'z','i','r','g') and the rest of the column name goes in ext, 
    e.g. ext='mag_forced_cmodel'.
    Can plot density contours in CC space, or scatter plot.
    use axis to define axes limits."""
    x = table[g1+ext] - table[r+ext]
    y = table[b+ext] - table[g2+ext]
    if axis is None:
        axis = np.array([[np.nanmin(x), np.nanmax(x)], [np.nanmin(y), np.nanmax(y)]])
    else:
        axis = np.array(axis)
    if plotpoints:
        obj = plt.scatter(x, y,rasterized=True, **kwargs)
        plt.axis(axis.flatten())
    elif plotcontours:
        obj = corner.hist2d(x, y,range=axis,bins=100,plot_datapoints=True,smooth=1,levels=levels,no_fill_contours=True,
                            plot_density=False,fill_contours=False,**kwargs)

    else:
        obj = corner.hist2d(x, y,range=axis,bins=100,plot_datapoints=False,smooth=1,levels=levels,**kwargs)
    if plotCCseq:
        c1 = np.arange(axis[0][0],axis[0][1])
        plt.plot(c1, CCparams[0]*c1+CCparams[1],'--') 
    plt.xlabel(g1 + ' - ' + r + ' [mag]')
    plt.ylabel(b + ' - ' + g2 + ' [mag]') 
    return obj
    
def CCRplot(table,r,g1,g2,b,X,Y,center,scale,ext='mag_forced_cmodel',axis=None):
    """ plot mean clustercentric distance in color-color space for a sample of 
    galaxies given in table. """
    x = table[g1+ext] - table[r+ext]
    y = table[b+ext] - table[g2+ext]
    rad = usel.radius(table[X],table[Y],center,scale=scale) # in arcmin
    if axis is None:
        axis = [[np.nanmin(x), np.nanmax(x)], [np.nanmin(y), np.nanmax(y)]]
    obj = epl.mean2d(x, y,weights = rad, range=axis, bins=50,plot_datapoints=False,smooth=None,plot_contours=False,)
    plt.xlabel(g1 + ' - ' + r + ' [mag]')
    plt.ylabel(b + ' - ' + g2 + ' [mag]') 
    return obj
    
def CMplot(table,r,g,ext='mag_forced_cmodel', axis=None,plotpoints = True,**kwargs):
    """ plot color-magnitude diagram for a sample of galaxies given in table. """
    
    y = table[g+ext] - table[r+ext]
    x = table[r+ext]
    if axis is None:
        axis = [[np.nanmin(x), np.nanmax(x)], [np.nanmin(y), np.nanmax(y)]]    
    if plotpoints:
        obj = plt.scatter(x, y,s=1,**kwargs)
        plt.axis(axis.flatten())
    else:
        obj = corner.hist2d(x, y,range=axis,bins=100,plot_datapoints=False,smooth=0.5,levels=[0.683,0.955])
    plt.xlabel(r + ' [mag]')
    plt.ylabel(g + ' - ' + r + ' [mag]')
    return obj
    
    
    
def Params(table, r,g1,g2,b, X,Y, center, scale, ext='mag_forced_cmodel', axis=None):
    """ find the parameters that describe the cluster red-squence
    in color-color space, given b-g1 vs g1-r colors in table. 
    Also supply column names of position (ra/dec, X/Y etc) in X,Y, the pixelscale in scale, 
    and the cluster center coordinates (in the same system as X/Y) in center."""
    
    rad = usel.radius(table[X],table[Y],center,scale)
    plt.figure()
    plotargs = { 'facecolors':'0.25','edgecolor':'0.25'}
    obj = CMplot(table[rad<2],r,g1, axis=np.array([[18,27],axis[0]]),plotpoints=True,ext=ext,**plotargs)
    a1, b1 = usel.linear_fit([])
    x = np.arange(18,27)
    plt.plot(x, a1*x+b1,'--')
    plt.figure()
    obj = CCplot(table[rad<2],r,g1,g2,b, axis=axis,plotpoints=True,ext=ext, **plotargs)
    a2, b2 = usel.linear_fit([])
    x = np.arange(axis[0][0],axis[0][1])
    plt.plot(x, a2*x+b2,'--')
    return a1, b1, a2, b2

def saveCCParams(CCparams, filename = 'CCparams.mat'):
    aCM, bCM, aCC, bCC = CCparams
    sio.savemat(filename,{'aCM':aCM,'bCM':bCM,'aCC':aCC,'bCC':bCC})
def loadCCParams(filename = 'CCparams.mat'):
    CCparams = sio.loadmat(filename,squeeze_me=True)
    CCparams = [CCparams[key] for key in ['aCM','bCM','aCC','bCC']]
    return CCparams
    
    
def SelectSources(table, r,g1,g2,b, CCparams, rlim, blim, ext='mag_forced_cmodel'):
    """
    SelectSources(table, r,g1,g2,b, CCparams, rlim, blim)
    select red and blue background galaxies in color-color space from a sample of galaxies, 
    given parameters that describe the red sequence in CCparams, 
    and a set of cuts defined in rlim and blim, respectively.
    
    """
    aCM, bCM, aCC, bCC = CCparams
    Z = table[r+ext]
    RZ = table[g1+ext] - table[r+ext]
    BR = table[b+ext] - table[g2+ext]
    
    f1 = aCM*Z + bCM # color-mag R-I seq
    seqdif1 = RZ - f1
    CCf = aCC * RZ + bCC          #color-color cluster sequence
    CCf2 = -1./aCC*RZ - bCC/aCC**2 # line perpendicular to color-color sequence
    CCdif = BR-CCf                 # B-R -CC Sequence
    CCdif2 = (BR-CCf2)/(1.+1./aCC**2)# B-R -perCC Sequence

    
    r_rzlim    = ( RZ > rlim[0] ) # RZ lower limit, separate red from blue
    r_maglim   =  ( Z > rlim[1] ) & ( Z < rlim[2] ) # magnitude limit, redderst band
    r_CCseqlim = ( CCdif < rlim[3] ) &  ( CCdif2 < rlim[4] ) 
    red = table[ r_maglim & r_rzlim & r_CCseqlim ]
    # (seqdif1 < rlim[1] # doesn't exist...
    
    b_CMseqlim = (seqdif1 < blim[1]) & (seqdif1>blim[0]) & (BR<4)
    b_maglim   =  ( Z > blim[2] ) & ( Z < blim[3] ) # magnitude limit, redderst band
    b_rzlim    = ( RZ < blim[4]) # RZ upper limit, separate red from blue
    b_CCseqlim = ( CCdif2 < blim[5] )
    blue =  table[  b_rzlim & (b_CCseqlim |  b_CMseqlim )  & b_maglim ]
    return red, blue


def SelectNotForeground(table, r,g1,g2,b, CCparams, lim,ext="mag_cmodel"):
    aCM, bCM, aCC, bCC = CCparams
    Z = table[r+ext]
    RZ = table[g1+ext] - table[r+ext]
    BR = table[b+ext] - table[g2+ext]
    
    CCf = aCC * RZ + bCC          #color-color cluster sequence
    CCf2 = -1./aCC*RZ - bCC/aCC**2 # line perpendicular to color-color sequence
    CCdif = BR-CCf                 # B-R -CC Sequence
    CCdif2 = (BR-CCf2)/(1.+1./aCC**2)# B-R -perCC Sequence

    CCseqlim =   (CCdif2 < lim[0]) | (CCdif < lim[1])  |  (CCdif2 > lim[2]) | (CCdif > lim[3])    
    maglim = (Z > lim[4]) & ( Z < lim[5]) 
    back = table[ CCseqlim & maglim ]
    return back

def run_CCselection(source, columns,center,scale=60,redlim = [0.5, 21, 28, -0.6, 2], bluelim = [-3, -0.8, 22, 28, 0.5, 0.5], CClim = np.array([[-1, 2.5],[-0.5, 3.5]]) ):
    """ run a full process of color-color selection, starting from finding the 
    adequate CC parameters of the red-sequence, selecting red/blue background galaxies, 
    and plotting the results in CC space"""
    
    z, r, i, g, ra, dec    = columns
    #%% CC params
    aCM, bCM, aCC, bCC = Params(source, z,r,i,g, ra,dec, center, scale=scale, axis=CClim)
    CCparams = [aCM, bCM, aCC, bCC]
    #%% CC select
    red, blue = SelectSources(source, z,r,i,g, CCparams, redlim , bluelim)
    back = table.vstack([red,blue])
    #%% CC plot
    plt.figure()
    plotargs = { 'facecolors':'0.25','edgecolor':'0.25'}
    obj1 = CCplot(source,'z','r','i','g',axis=CClim,plotpoints=False,**plotargs)
    plotargs = { 'facecolors':'r','edgecolor':'r'}
    obj2 = CCplot(red,'z','r','i','g',axis=CClim,plotpoints=True,**plotargs)
    plotargs = { 'facecolors':'b','edgecolor':'b'}
    obj3 = CCplot(blue,'z','r','i','g',axis=CClim,plotpoints=True,**plotargs)
    return red, blue, back, CCparams
    
    
def CM_evtrack(typ,cols,filtset,savefig,mrk):
    """ CM_evtrack(typ,cols,filtset,savefig)
    #--------------------------------------------
    # plot evolutionary color tracks for galaxy type typ, in color-magnitude space
    # type - galaxy type matrix, 
    # cols - struct containing columns of matrix, one for each filter in filtset, and one called 'zb' for redshift.
    # filtset - 4 filter names, from red to blue, e.g. ['z','i','r','g'] for CC
    # plot: g-r vs. i-z.
    # saveplot - bool to save plot as 'g-r_vs_i-z.png'  """
    

    filt1,filt2,filt3 = filtset
    
    i = range(1,len(typ),8) #downsizing
    plt.scatter(typ[i,cols[filt2]] - typ[i,cols[filt1]], typ[i,cols[filt4]] - typ[i,cols[filt3]],
                s=10, c=typ[i,cols['zb']],marker = mrk)

    plt.ylabel(filt3 + '-' + filt2 + ' [mag]')
    plt.xlabel(filt1 +' [mag]')
    plt.axis([18, 27, -0.5, 3])

    caxis([0., 3.5])
    
def CC_evtrack(typ,cols,filtset,mrk):
    """ CC_evtrack(typ,cols,filtset,mrk)
    --------------------------------------------
    # plot evolutionary color tracks for galaxy type typ, in color-color space
     type - galaxy type matrix,
    % cols - struct containing columns of matrix, one for each filter in filtset, and one called 'zb' for redshift.
    % filtset - 4 filter names, from red to blue, e.g. ['z','i','r','g'] for CC
    % mrk - marker type
    """
    
    filt1,filt2,filt3,filt4 = filtset
    
    i = range(0,len(typ),8) #downsizing
    # figure;hold on
    plt.scatter(typ[i,cols[filt2]] - typ[i,cols[filt1]], typ[i,cols[filt4]] - typ[i,cols[filt3]],
                s=10, c=typ[i,cols['zb']],marker = mrk, edgecolors='none')
    plt.xlabel(filt2 + '-' + filt1 + ' [mag]')
    plt.ylabel(filt4 + '-' + filt3 + ' [mag]')
    plt.axis([-0.5, 2.5, -0.2, 3.5])

    plt.clim(0., 3.5)

    
def gt_ColorLim(sample,CCparams, bands=('z','r','i','g'), cols=('ra','dec','e1','e2','sigma_e','zs','Pz','ra_l','dec_l','zl'),
                lims=[],limN=0,binsize=0.05,nbin=10,minr=0.1*u.Mpc, maxr=0.5*u.Mpc,
                ext='mag_forced_cmodel',zvec=np.arange(0,7.01,0.01),  **kwargs):
    """ (old version) plot mean tangential shear (WL signal) as a function of color limit, for all galaxies inside minr<R<maxr 
    to find appropriate color cuts isolating background galaxies"""
    
    gtmean, gterr, color = [np.ones(nbin)*np.nan for _ in xrange(3) ]
    redlim, bluelim = np.array([0.5, 21, 28, -0.5, 2.]) , np.array([-3, -0.8, 22, 28, 0.5, 0.5]) #defaults
    x,y,e1,e2,sig_e,zs,Pz,xc,yc,zl = cols
    r,g1,g2,b = bands
    # red, lim1
    binstart=lims[limN]-binsize*nbin/2.;
    for i in range(nbin):
        lims_i = lims[:]
        lims_i[limN] = binstart + binsize*(i-1.)
        print(lims_i[limN])
        if len(lims)==5:
            redlim=lims_i
        elif len(lims)==6:
            bluelim=lims_i
        else:
            raise TypeError("unrecognized sample limits option")
            
        red, blue = SelectSources(sample, r,g1,g2,b, CCparams, redlim , bluelim, ext=ext)
        if len(lims)==5:
            mat=red
        else:
            mat=blue
        if len(mat)==0:
            continue
        # slightly easier with Pz=None    
        table = WL.DSigma_profile(mat[x], mat[y], mat[e1], mat[e2], mat[sig_e], mat[zs], mat[xc], mat[yc], mat[zl], wl=1., Pz=mat[Pz], zvec=zvec, minr=minr, maxr=maxr, nbinr=1, correctResponsivity=True,plotbool=False)
        gtmean[i] = table['DSigmat']
        gterr[i] = table['DSigmat_err (regauss)']
        color[i] = lims_i[limN]
    plt.figure()
    plt.errorbar(color, gtmean, yerr=gterr, **kwargs)
    plt.xlabel(r'$\Delta color$ #' + str(limN+1))
    plt.ylabel(r'$\Delta\Sigma$')
    return gtmean,gterr

def gt_ColorRad(sample,CCparams, bands=('z','r','i','g'), cols=('ra','dec','e1','e2','sigma_e','zs','Pz','ra_l','dec_l','zl'),
                lims=[],limN=0,binsize=0.05,nbin=10,minr=0.2*u.Mpc, maxr=5*u.Mpc, nbinr=5,
                ext='mag_cmodel',zvec=np.arange(0,7.01,0.01),  **kwargs):
    """ (old version) plot mean tangential shear (WL signal) as a function of color limit, for several radial anulii separately, 
    to find appropriate color cuts isolating background galaxies"""

    color = np.ones(nbin)*np.nan    
    gtmean, gterr = [np.ones([nbin,nbinr])*np.nan for _ in xrange(2) ]
    redlim, bluelim = np.array([0.5, 21, 28, -0.5, 2.]) , np.array([-3, -0.8, 22, 28, 0.5, 0.5]) #defaults
    x,y,e1,e2,sig_e,zs,Pz,xc,yc,zl = cols
    r,g1,g2,b = bands
    # red, lim1
    binstart=lims[limN]-binsize*nbin/2.;
    for i in range(nbin):
        lims_i = lims[:]
        lims_i[limN] = binstart + binsize*(i-1.)
        print(lims_i[limN])
        if len(lims)==5:
            redlim=lims_i
        elif len(lims)==6:
            bluelim=lims_i
        else:
            raise TypeError("unrecognized sample limits option")
            
        red, blue = SelectSources(sample, r,g1,g2,b, CCparams, redlim , bluelim, ext=ext)
        if len(lims)==5:
            mat=red
        else:
            mat=blue
        if len(mat)==0:
            continue
        # slightly easier with Pz=None    
        table = WL.DSigma_profile(mat[x], mat[y], mat[e1], mat[e2], mat[sig_e], mat[zs], mat[xc], mat[yc], mat[zl], wl=1., Pz=mat[Pz], zvec=zvec, minr=minr, maxr=maxr, nbinr=nbinr, correctResponsivity=True,plotbool=False)
        gtmean[i,:] = table['DSigmat']
        gterr[i,:] = table['DSigmat_err (regauss)']
        color[i] = lims_i[limN]

    #fig,ax = plt.subplots()
    cmap = plt.get_cmap('jet')
    colors = [cmap(i) for i in np.linspace(0.1, 1, nbinr)]
    plt.gca().set_color_cycle(colors[::-1])
    for j in range(nbinr):
        plt.errorbar(color, gtmean[:,j], yerr=gterr[:,j], label="%d" % (table['R [Mpc]'][j]*1e3)+r" h$^{-1}$kpc", **kwargs)
    plt.xlabel(r'$\Delta color$ #' + str(limN+1))
    plt.ylabel(r'$\Delta\Sigma$')
    #plt.legend()
    return gtmean,gterr
    
    
    
def gt_ColorRad_light(sample,CCparams, bands=('z','r','i','g'),lims=[],limN=0,binsize=0.05,nbin=10,
                      minr=0.2, maxr=5, nbinr=5,ext='mag_forced_cmodel',  **kwargs):
    """ plot mean tangential shear (WL signal) as a function of color limit, for several radial anulii separately, 
    to find appropriate color cuts isolating background galaxies.
    Sample needs to contain cluster-galaxy cross-correlated list"""    
    colorvec = np.ones(nbin)*np.nan    
    gtmean, gterr = [np.ones([nbin,nbinr])*np.nan for _ in xrange(2) ]
    redlim, bluelim = np.array([0.5, 21, 28, -0.5, 2.]) , np.array([-3, -0.8, 22, 28, 0.5, 0.5]) #defaults
    r,g1,g2,b = bands
    # red, lim1
    binstart=lims[limN]-binsize*nbin/2.;
    for i in range(nbin):
        lims_i = lims[:]
        lims_i[limN] = binstart + binsize*(i-1.)
        print(lims_i[limN])
        if len(lims)==5:
            redlim=lims_i
        elif len(lims)==6:
            bluelim=lims_i
        else:
            raise TypeError("unrecognized sample limits option")
            
        red, blue = SelectSources(sample, r,g1,g2,b, CCparams, redlim , bluelim, ext=ext)
        if len(lims)==5:
            mat=red
        else:
            mat=blue
        if len(mat)==0:
            continue
        table = WL.DSigma_profile_light(mat, minr=minr, maxr=maxr, nbinr=nbinr,  correctResponsivity=True,)
        #print(table)          
        gtmean[i,:] = table['dSigma+']
        gterr[i,:] = table['dSigma+_err']
        colorvec[i] = lims_i[limN]

    #fig,ax = plt.subplots()
    cmap = plt.get_cmap('jet')
    colors = [cmap(i) for i in np.linspace(0.1, 1, nbinr)]
    #plt.gca().set_color_cycle(colors[::-1])
    xoff=(colorvec[1]-colorvec[0])/10
    for j in range(nbinr):
        #plt.errorbar(color+j*xoff, gtmean[:,j]/gtmean[-1,j], yerr=gterr[:,j]/gtmean[-1,j], label="%d" % (table['R'][j]*1e3)+r" h$^{-1}$kpc", **kwargs)
        plt.fill_between(colorvec+j*xoff, gtmean[:,j]/gtmean[0,j]-gterr[:,j]/gtmean[0,j],gtmean[:,j]/gtmean[0,j]+gterr[:,j]/gtmean[0,j], label=None, color=colors[-1-j], alpha=0.2)        
        plt.plot(colorvec+j*xoff, gtmean[:,j]/gtmean[0,j], label="%d" % (table['R'][j]*1e3)+r" h$^{-1}$kpc",c=colors[-1-j], **kwargs)
    plt.xlabel(r'$\Delta color$ #' + str(limN+1))
    plt.ylabel(r'$\Delta\Sigma$')
    #plt.legend()
    return gtmean,gterr
###
    
    
def gt_ColorRad_lightflex(sample,CCparams, bands=('z','r','i','g'),lims=[],limN=0,binsize=0.05,nbin=10,rbins=np.logspace(np.log10(0.2),np.log10(2),4+1),ext='mag_forced_cmodel',  **kwargs):
   """ (latest version used) plot mean tangential shear (WL signal) as a function of color limit, in  radial bins defined by rbins, 
    to find appropriate color cuts isolating background galaxies.
    Sample needs to contain cluster-galaxy cross-correlated list"""

    colorvec = np.ones(nbin)*np.nan    
    gtmean, gterr = [np.ones([nbin,len(rbins)-1])*np.nan for _ in xrange(2) ]
    redlim, bluelim = np.array([0.5, 21, 28, -0.5, 2.]) , np.array([-3, -0.8, 22, 28, 0.5, 0.5]) #defaults
    r,g1,g2,b = bands
    # red, lim1
    binstart=lims[limN]-binsize*nbin/2.;
    for i in range(nbin):
        lims_i = lims[:]
        lims_i[limN] = binstart + binsize*(i-1.)
        print(lims_i[limN])
        if len(lims)==5:
            redlim=lims_i
        elif len(lims)==6:
            bluelim=lims_i
        else:
            raise TypeError("unrecognized sample limits option")
            
        red, blue = SelectSources(sample, r,g1,g2,b, CCparams, redlim , bluelim, ext=ext)
        if len(lims)==5:
            mat=red
        else:
            mat=blue
        if len(mat)==0:
            continue
        table = WL.DSigma_profile_light_flexbin_calib(mat, rbins,  )        
        gtmean[i,:] = table['dSigma+']
        gterr[i,:] = table['dSigma+_err']
        colorvec[i] = lims_i[limN]

    cmap = plt.get_cmap('jet')
    colors = [cmap(i) for i in np.linspace(0.1, 1, len(rbins)-1)]
    xoff=(colorvec[1]-colorvec[0])/10
    for j in range(len(rbins)-1):
        plt.fill_between(colorvec+j*xoff, gtmean[:,j]/np.mean(gtmean[0:3,j])-gterr[:,j]/np.mean(gtmean[0:3,j]),gtmean[:,j]/np.mean(gtmean[0:3,j])+gterr[:,j]/np.mean(gtmean[0:3,j]), label=None, color=colors[-1-j], alpha=0.2)        
        plt.plot(colorvec+j*xoff, gtmean[:,j]/np.mean(gtmean[0:3,j]), label="%d" % (table['R'][j]*1e3)+r" h$^{-1}$kpc",c=colors[-1-j], **kwargs)
    plt.xlabel(r'$\Delta color$ #' + str(limN+1))
    plt.ylabel(r'$\Delta\Sigma$')
    return gtmean,gterr
###
    
    


def Pztot(z,zl,Pz,zvec=None,dz=0.05):
    """ calculate the sum pf P(z) for z>zl; 
    [Oguri-san's photo-z P(z) cuts; see Medezinski et al 2018b]"""
    if zvec is None:
        zvec = np.arange(0,7.01,0.01)
    zmat = np.tile(zvec,[len(zl),1])
    zlmat = np.tile(zl,[len(zvec),1]).T
    Pzmask = np.ma.array( data=Pz, mask = (zmat<=zlmat+dz) )
    Psum = np.ma.sum(Pzmask, axis=1)/np.sum(Pz,axis=1)
    return Psum

def Pz_selection(z,zl,Pz,zvec=None,pcut=0.98,dz=0.05,zmax=2.5):
    """ select galaxies whose sum P(z) for z>zl is greater than some percentage (98%) AND z<zmax
    [Oguri-san's photo-z P(z) cuts;  see Medezinski et al 2018b]"""
    Psum = Pztot(z,zl,Pz,zvec=zvec,dz=dz)
    sel = ( (Psum>pcut) & (z<zmax) )
    return sel

def gt_pzlim(sample,dzvec,minr=0.2*u.Mpc, maxr=5*u.Mpc, nbinr=5,  **kwargs):
    """ plot mean tangential shear (WL signal) as a function of cluster redshift threshold dz"""
    dzvec= dzvec[::-1]
    nbin = len(dzvec)
    gtmean, gterr = [np.ones([nbin,nbinr])*np.nan for _ in xrange(2) ]
    for i in range(nbin):
        Psum_i = sample['Psum'][:,-1-i] 
        mat = sample[(Psum_i>0.98) & (sample['zs']<2.5)]
        table = WL.DSigma_profile_light_calib(mat, minr=minr, maxr=maxr, nbinr=nbinr,  )       
        gtmean[i,:] = table['dSigma+']
        gterr[i,:] = table['dSigma+_err']
        cmap = plt.get_cmap('jet')
    colors = [cmap(i) for i in np.linspace(0.1, 1, nbinr)]
    plt.gca().set_color_cycle(colors[::-1])
    xoff=(dzvec[1]-dzvec[0])/10
    for j in range(nbinr):
        plt.fill_between(dzvec+j*xoff, gtmean[:,j]/gtmean[0,j]-gterr[:,j]/gtmean[0,j],gtmean[:,j]/gtmean[0,j]+gterr[:,j]/gtmean[0,j], label=None, color=colors[-1-j], alpha=0.2)        
        plt.plot(dzvec+j*xoff, gtmean[:,j]/gtmean[0,j], label="%d" % (table['R'][j]*1e3)+r" h$^{-1}$kpc",c=colors[-1-j], **kwargs)

    plt.xlabel(r'$\Delta z$')
    plt.ylabel(r'$\Delta\Sigma$ (normalized)')
    return gtmean,gterr

def gt_pcutlim(sample,pcuts,minr=0.2*u.Mpc, maxr=5*u.Mpc, nbinr=5,  **kwargs):
    """ plot mean tangential shear (WL signal) as a function of P(z) percentage cut"""
    nbin = len(pcuts)
    gtmean, gterr = [np.ones([nbin,nbinr])*np.nan for _ in xrange(2) ]
    for i in range(nbin):
        mat = sample[(sample['Psum']>pcuts[i]) & (sample['zs']<2.5)]
        table = WL.DSigma_profile_light_calib(mat, minr=minr, maxr=maxr, nbinr=nbinr,  )
        #print(table)          
        gtmean[i,:] = table['dSigma+']
        gterr[i,:] = table['dSigma+_err']
    cmap = plt.get_cmap('jet')
    colors = [cmap(i) for i in np.linspace(0.1, 1, nbinr)]
    plt.gca().set_color_cycle(colors[::-1])
    xoff=(pcuts[1]-pcuts[0])/10
    for j in range(nbinr):
        plt.fill_between(pcuts+j*xoff, gtmean[:,j]/gtmean[0,j]-gterr[:,j]/gtmean[0,j],gtmean[:,j]/gtmean[0,j]+gterr[:,j]/gtmean[0,j], label=None, color=colors[-1-j], alpha=0.2)        
        plt.plot(pcuts+j*xoff, gtmean[:,j]/gtmean[0,j], label="%d" % (table['R'][j]*1e3)+r" h$^{-1}$kpc",c=colors[-1-j], **kwargs)

    plt.xlabel(r'$p_{\rm cut}$')
    plt.ylabel(r'$\Delta\Sigma$ (normalized)')

    return gtmean,gterr
