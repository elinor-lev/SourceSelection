# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:25:57 2015

@author: elinor
"""

import numpy as np
import useful_el as usel
import matplotlib.pyplot as plt
import cosmology as cosmo_el
from astropy import units as u
import astropy.table as t
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from matplotlib import ticker
import gc
import h5py

def weight_shear(g1,g2,w,a=0.4):
    U=1/(1/w+a**2)
    Unorm = len(U)/np.nansum(U)
    g1 = g1 * U*Unorm
    g2 = g2 * U*Unorm
    return g1, g2


def shear_rot(g1,g2,phi2=0):
    """ g1rot, g2rot = shear_rot(g1,g2,phi2)"""
    cos2 = np.cos(phi2)
    sin2 = np.sin(phi2)
    g1rot = cos2 * g1 - sin2 * g2 #from rotation x'=-(cos(2*theta)*x+sin(2*theta)*y)
    g2rot = sin2 * g1 + cos2 * g2
    return g1rot, g2rot
    
    
def gT(x,y,g1,g2,xc,yc):
    """ gT(x,y,g1,g2,center)
    tangential shear given position and frame center , in pixel coordinates!"""
    #x, y = (x - xc, y - yc)
    dx = x - xc
    dy = y - yc
    r = np.hypot(dx, dy)
    # minus the g1,g2 for im2shape
    cos2 = (dx**2 - dy**2) / r**2  #from cos(2*theta)=cos**2(theta)-sin**2(theta) theta is the angle of the radius from center
    sin2 = 2 * dx * dy / r**2    #from sin(2*theta)=2*cos(theta)*sin(theta)
    gt = - (g1 * cos2  + g2 * sin2) #from rotation x'=-(cos(2*theta)*x+sin(2*theta)*y)
    gr =    g1 * sin2 - g2 * cos2 
    #ge =   (sin2 * g1 + cos2 * g2)**2   
    #sigma_g = np.sqrt(1/(2*wg))
    #sigma_wg = np.sqrt(1/wg + a**2)
    return gt, gr, r
    
def gTsky(a,d,g1,g2,ac,dc):
    """ gTsky(x,y,g1,g2,center)
    tangential shear given position and frame center , in sky coordinates!
    angles must be in radians"""
    # requires high-precision
    unit=a.unit
    a = np.array(a,dtype=np.float64)
    d = np.array(d,dtype=np.float64)
    ac = np.array(ac,dtype=np.float64)
    dc = np.array(dc,dtype=np.float64)
    #
    r = np.arccos(np.cos(d)*np.cos(dc)*np.cos(a-ac)+np.sin(d)*np.sin(dc))
    cosp = np.sin(ac - a)*np.cos(d)/np.sin(r)
    sinp = (-np.cos(dc)*np.sin(d) + np.sin(dc)*np.cos(d)*np.cos(a-ac))/np.sin(r)
    cos2p = cosp**2 - sinp**2
    sin2p = 2.*sinp*cosp
    gt = - (g1 * cos2p + g2 * sin2p)
    gr =   (g1 * sin2p - g2 * cos2p)
    return gt, gr, r*unit

#def gTsky2(a,d,g1,g2,ac,dc):
#    """ gTsky(x,y,g1,g2,center)
#    tangential shear given position and frame center , in sky coordinates!
#    angles must be in radians"""
#    # requires high-precision
#    dx = a - ac
#    dy = d - dc
#    coords1 = SkyCoord(a,d)
#    coords2 = SkyCoord(ac,dc)
#    r = coords1.separation(coords2)
#    cos2p = (dx**2 - dy**2) / r.deg**2  #from cos(2*theta)=cos**2(theta)-sin**2(theta) theta is the angle of the radius from center
#    sin2p = 2 * dx * dy / r.deg**2    #from sin(2*theta)=2*cos(theta)*sin(theta)
#    gt = - (g1 * cos2p + g2 * sin2p)
#    gr =   (g1 * sin2p - g2 * cos2p)
#    return gt, gr, r
    
def gt_profile(x, y, g1, g2, sig_g, xc, yc, zl=None, minr=0.1, maxr=20, nbinr=10, cosmo=None, a=0.365, marker = 'o', correctResponsivity=False,**kwargs):
    """ gt_profile(x,y,g1,g1,gw, xc, yc, zl, minr, maxr, nbinr=10, cosmo=None, a=0.365)
    plot tangential shear profile
    """
    gt, gx, r = gT(x,y,g1,g2,xc,yc)
    unit = str(minr.unit)
    if ('pc' in "{0.unit:FITS}".format(minr)):
        r = ( (r.to(u.arcmin))*cosmo.kpc_comoving_per_arcmin(zl) ).to(getattr(u,unit))
        #d_co = cd.comoving_distance(zl, **cosmo)
        #r = ( r.to(u.rad) * d_co*u.Mpc/u.rad ).to(getattr(u,unit))
    else:
        r = r.to(getattr(u,unit))
    wg     =  1/(sig_g**2 + a**2)
    #sig_g = sig_g/np.sqrt(2) # for each component??
    if correctResponsivity: ## for regauss
        R = 1.-a**2
    else: ## no need for KSB
        R = 1 #??
    # binning over radial axis
    gtw, gterr, gterr2, x1, xmin, xmax, Nbin, STD, wbin = usel.Wbinning(r, gt, sig_g, wg, minr=minr/minr.unit, maxr=maxr/minr.unit, nbinr=nbinr)
    gxw, gxerr, gxerr2, _, _, _, _, _, _ = usel.Wbinning(r, gx, sig_g, wg, minr=minr/minr.unit, maxr=maxr/minr.unit, nbinr=nbinr)
    xerr = (xmax-xmin)/2
    gtw, gterr, gterr2, gxw, gxerr, gxerr2 = gtw/(2*R), gterr/(2*R), gterr2/(2*R), gxw/(2*R), gxerr/(2*R), gxerr2/(2*R) # correct for resopnsivity
    
    # plot gt,gx
    plt.subplots_adjust(hspace=0.001)
    top_ax = plt.subplot(211)
    top_ax.set_yticks([0], minor=True)
    top_ax.yaxis.grid(True, which='minor', ls='--')
    top_ax.errorbar(x1, gtw, yerr=gterr2, xerr=xerr, marker = marker, ls = 'None',**kwargs)
    plt.xscale('log')
    plt.ylabel(r'$g_+$')    
    bottom_ax = plt.subplot(212, sharex=top_ax)
    bottom_ax.errorbar(x1,gxw,yerr=gxerr2, xerr=xerr, marker = marker, ls = 'None',**kwargs)
    bottom_ax.set_yticks([0], minor=True)
    bottom_ax.yaxis.grid(True, which='minor', ls='--')
    plt.ylabel(r'$g_\times$')
    plt.setp(top_ax.get_xticklabels(), visible=False)
    plt.xlabel('R [{0}]'.format(unit))
    bottom_ax.xaxis.set_major_formatter(ticker.LogFormatter()) 
    nbins = len(top_ax.get_yticklabels()) # added 
    bottom_ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=nbins, prune='upper')) # adde
    plt.xlim([x1[0]-np.diff(x1)[0], x1[-1]+np.diff(x1)[-1]])
    
    return t.Table( [x1, xerr, gtw, gterr, gterr2, gxw, gxerr, gxerr2],
                   names = ('R [{0}]'.format(unit),'dR [{0}]'.format(unit) , 'gt', 'gterr (KSB+)', 'gterr (regauss)', 'gx', 'gxerr (KSB+)', 'gxerr (regauss)') )

def StackWeight(sig_g,zl,wl=1.,zs=None,Pz=None,cosmo=None, a=0.365):
    if Pz is None:
        Sigma_cr = cosmo_el.Sigma_crit_co(zl,zs,cosmo=cosmo).value
    else: 
        Sigma_cr = cosmo_el.Sigma_cr_co_PDF(zl, Pz, cosmo=cosmo)        
    wg     =  1./(sig_g**2 + a**2)
    w = wl * wg * Sigma_cr**-2
    return w
  
def DSigma_profile(x, y, g1, g2, sig_g, zs, xc, yc, zl, wl=1., Pz=None, zvec=np.arange(0,7.01,0.01), minr=0.1, maxr=10, nbinr=10, cosmo=None, a=0.365, marker = 'o', correctResponsivity=False,plotbool=True,**kwargs):
    """ DSigma_profile(x, y, g1, g2, sig_g, zs, xc, yc, zl, wl, minr=0.1, maxr=10, nbinr=10, cosmo=None, a=0.365, marker = 'o', correctResponsivity=False):

    plot density contrast profile
    """
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=100, Om0=0.27,)

    if Pz is None:
        Sigma_cr = cosmo_el.Sigma_crit_co(zl,zs,cosmo=cosmo).value
    else: 
        Sigma_cr = cosmo_el.Sigma_cr_co_PDF(zl, Pz, cosmo=cosmo, zvec=zvec)
    gt, gx, r = gT(x,y,g1,g2,xc,yc)
    unit = str(minr.unit)
    if ('pc' in "{0.unit:FITS}".format(minr)):
        r = ( (r.to(u.arcmin))*cosmo.kpc_comoving_per_arcmin(zl) ).to(getattr(u,unit))
    else:
        r = r.to(getattr(u,unit))
    wg     =  1./(sig_g**2 + a**2)
    if correctResponsivity: ## for regauss
        R = 1.-a**2
    else: ## no need for KSB
        R = 0.5 #??
    DSigma = gt * Sigma_cr #  gt -> DSigma
    Dx = gx * Sigma_cr # 45-rotated - zero consistency test
    sig_S = sig_g * Sigma_cr # error of mass density
    w = wg * wl * Sigma_cr**(-2)
    Inonan  = (~np.isnan(DSigma)) # to remove nan photo-z's
    # binning
    Stw, Sterr, Sterr2, x1, xmin, xmax, Nbin, STD, wbin= usel.Wbinning(r[Inonan], DSigma[Inonan], sig_S[Inonan], w[Inonan], minr=minr/minr.unit, maxr=maxr/maxr.unit, nbinr=nbinr)
    Sxw, Sxerr,Sxerr2, x1, xmin, xmax, Nbin, STD,wbin  = usel.Wbinning(r[Inonan], Dx[Inonan], sig_S[Inonan], w[Inonan], minr=minr/minr.unit, maxr=maxr/maxr.unit, nbinr=nbinr)
    # correct for responsivity    
    Stw, Sterr, Sterr2, Sxw, Sxerr, Sxerr2 = Stw/(2*R), Sterr/(2*R), Sterr2/(2*R), Sxw/(2*R), Sxerr/(2*R), Sxerr2/(2*R) # correct for resopnsivity
    xerr = (xmax-xmin)/2 # this is wrong in log-space.. good enough for presentation purposes?
    # weighted redshift?
    zlw, _, _, _, _, _, _, _, _ = usel.Wbinning(r[Inonan], zl[Inonan], sig_S[Inonan], w[Inonan], minr=minr/minr.unit, maxr=maxr/minr.unit, nbinr=nbinr)
    if plotbool:
        # plotting
        ax, gs = plotShear(x1,Stw,Sxw,xerr,Sterr2,Sxerr2,marker=marker,**kwargs)
        ax[1].set_xticks(np.around(np.logspace(np.log10(minr.value),np.log10(maxr.value),5),usel.Ndecimal(minr.value)))
    return t.Table( [x1, xerr, Stw, Sterr, Sterr2, Sxw, Sxerr, Sxerr2, Nbin, wbin, zlw],
                   names = ('R','dR' , 'dSigma+', 'dSigma+_err (KSB+)', 'dSigma+_err', 'dSigmax', 'dSigmax_err (KSB+)', 'dSigmax_err', 'N', 'W', 'zl_wmean') )
                   
def plotShear(r,gt,gx,rerr,gterr,gxerr,marker='o',gs=None,**kwargs):
    """ plotShear(r,gt,gx,rerr,gterr,gxerr,marker='o',gs=None,**kwargs)
    """
    from matplotlib import gridspec
    convert = (u.Mpc**(-2)).to(u.pc**(-2)) # plot in units Msun/pc**2
    #convert=1e-14
    if gs is None:
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    plt.subplots_adjust(hspace=0.001)
    
    # gt
    top_ax = plt.subplot(gs[0])
    #top_ax = plt.subplot(211)
    top_ax.set_yticks([0], minor=True)
    top_ax.yaxis.grid(True, which='minor', ls='--')
    top_ax.errorbar(r, (gt*convert), yerr=(gterr*convert), xerr=rerr, marker = marker, ls = 'None',**kwargs)
    plt.xscale('log')
    #plt.yscale('log')
    #plt.ylabel(r'$\Delta \Sigma_+\ [h 10^{14} M_\odot \mathrm{Mpc}^{-2}]$')
    plt.ylabel(r'$\Delta \Sigma_+\ [h M_\odot \mathrm{pc}^{-2}]$')
    # gx
    bottom_ax = plt.subplot(gs[1], sharex=top_ax)
    #bottom_ax = plt.subplot(212, sharex=top_ax)
    bottom_ax.set_yticks([0], minor=True)
    bottom_ax.yaxis.grid(True, which='minor', ls='--')
    bottom_ax.errorbar(r, (gx*convert), yerr=(gxerr*convert), xerr=rerr, marker = marker, ls = 'None',**kwargs)
    plt.ylabel(r'$\Delta\Sigma_\times$')
    plt.setp(top_ax.get_xticklabels(), visible=False)
    plt.xlabel(r'R [Mpc/$h$]')
    bottom_ax.xaxis.set_major_formatter(ticker.LogFormatter()) 
    nbins = len(top_ax.get_yticklabels()) # added 
    bottom_ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=usel.floor_to_odd(nbins/2.), prune='upper')) # adde
    plt.xlim([r[0]-np.diff(r)[0], r[-1]+np.diff(r)[-1]])
    #plt.tight_layout()
    #plt.subplots_adjust(hspace=0.001)
    return (top_ax, bottom_ax), gs
    

def plotShearEB(r,gt,gx,gterr,gxerr,ax=None,**kwargs):
    """ plotShearV2(r,gt,gx,,gterr,gxerr,marker='o',gs=None,**kwargs)
    """

    convert = (u.Mpc**(-2)).to(u.pc**(-2)) # plot in units Msun/pc**2
    #convert=1e-14
    if ax is None:
        fig, ax = plt.subplots()
    #top_ax = plt.subplot(211)
#    ax.set_yticks([0], minor=True)
#    ax.yaxis.grid(True, which='minor', ls='--')
    ax.errorbar(r, (gt*convert), yerr=(gterr*convert),  marker = 'o',color='k', ls = 'None',label='E',**kwargs)
    plt.xscale('log')
    #plt.yscale('log')
    #plt.ylabel(r'$\Delta \Sigma_+\ [h 10^{14} M_\odot \mathrm{Mpc}^{-2}]$')
    plt.ylabel(r'$\Delta \Sigma [h M_\odot \mathrm{pc}^{-2}]$')
    # gx
    ax.plot(r*10**0.01, (gx*convert),   marker = 'x',color='r', ls = 'None',label='B',**kwargs)
    plt.xlabel(r'R [Mpc/$h$]')
    ax.xaxis.set_major_formatter(ticker.LogFormatter()) 
    plt.xlim([r[0]-np.diff(r)[0], r[-1]+np.diff(r)[-1]])
    plt.axhline(y=0,ls=':',c='k')
    #plt.tight_layout()
    #plt.subplots_adjust(hspace=0.001)
    plt.legend()
    return ax
    
def plotShearNumber(r,gt,gx,ng,rerr,gterr,gxerr,marker='o',gs=None,**kwargs):
    from matplotlib import gridspec
    convert = (u.Mpc**(-2)).to(u.pc**(-2)) # plot in units Msun/pc**2
    #convert=1e-14
    if gs is None:
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1],hspace=0.001) 
    #plt.subplots_adjust(hspace=0.001)
    
    # gt
    top_ax = plt.subplot(gs[0])
    #top_ax = plt.subplot(211)
    top_ax.set_yticks([0], minor=True)
    top_ax.yaxis.grid(True, which='minor', ls='--')
    top_ax.errorbar(r, (gt*convert), yerr=(gterr*convert), xerr=rerr, marker = marker, ls = 'None',**kwargs)
    plt.xscale('log')
    #plt.yscale('log')
    #plt.ylabel(r'$\Delta \Sigma_+\ [h 10^{14} M_\odot \mathrm{Mpc}^{-2}]$')
    plt.ylabel(r'$\Delta \Sigma_+\ [h M_\odot \mathrm{pc}^{-2}]$')
    top_ax.yaxis.set_label_coords(-0.08, 0.5)
    plt.setp(top_ax.get_xticklabels(), visible=False)
    nbins = len(top_ax.get_yticklabels()) # added
    # gx
    mid_ax = plt.subplot(gs[1], sharex=top_ax)
    plt.setp(mid_ax.get_xticklabels(), visible=False)
    mid_ax.set_yticks([0], minor=True)
    mid_ax.yaxis.grid(True, which='minor', ls='--')
    mid_ax.errorbar(r, r*(gx*convert), yerr=r*(gxerr*convert), xerr=rerr, marker = marker, ls = 'None',**kwargs)
    mid_ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='upper')) # adde
    plt.ylabel(r'$R\Delta\Sigma_\times$',fontsize=14)
    mid_ax.yaxis.set_label_coords(-0.08, 0.5)

    #n_g
    bottom_ax = plt.subplot(gs[2], sharex=top_ax)
    bottom_ax.plot(r,ng/1e4,marker='None',ls='-',**kwargs)
    plt.ylabel(r"$n_{\rm g}/10^4$" "\n" "[$h^2$Mpc$^{-2}$]",fontsize=14)
    bottom_ax.yaxis.set_label_coords(-0.08, 0.5)
    plt.xlabel(r'R [Mpc/$h$]')
    #plt.yscale('log')
    #bottom_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.e'))
#    bottom_ax.xaxis.set_major_formatter(ticker.LogFormatter())  
    #bottom_ax.set_yticks([])
    #bottom_ax.yaxis.set_major_formatter(ticker.LogFormatter(labelOnlyBase=False))
    bottom_ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='upper',min_n_ticks=2)) # adde
    
    plt.xlim([r[0]-np.diff(r)[0], r[-1]+np.diff(r)[-1]])
    #plt.tight_layout()
    #plt.subplots_adjust(hspace=0.001)
    return (top_ax, mid_ax, bottom_ax), gs
    
def plotShearE(r,gt,rerr,gterr,marker='o',gs=None,**kwargs):
    convert = (u.Mpc**(-2)).to(u.pc**(-2)) # plot in units Msun/pc**2
    #convert=1e-14   
    if gs is None:
        gs = gridspec.GridSpec(1, 1,) 

    # gt
    top_ax = plt.subplot(gs[0])
    #top_ax = plt.subplot(211)
    top_ax.set_yticks([0], minor=True)
    top_ax.yaxis.grid(True, which='minor', ls='--')
    top_ax.errorbar(r, (gt*convert), yerr=(gterr*convert), xerr=rerr, marker = marker, ls = 'None',**kwargs)
    plt.xscale('log')
    #plt.yscale('log')
    #plt.ylabel(r'$\Delta \Sigma_+\ [h 10^{14} M_\odot \mathrm{Mpc}^{-2}]$')
    plt.ylabel(r'$\Delta \Sigma_+\ [h M_\odot \mathrm{pc}^{-2}]$')
    plt.setp(top_ax.get_xticklabels(), visible=False)
    plt.xlabel(r'R [Mpc/$h$]')
    plt.xlim([r[0]-np.diff(r)[0], r[-1]+np.diff(r)[-1]])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.001)
    return top_ax, gs
#%%
def stackSources(lens, source, 
                 lensColumns=['RADeg','decDeg','redshift'], 
                 sourceColumns=['id','ra2000','decl2000','shape_hsm_regauss_e1','shape_hsm_regauss_e2','shape_hsm_regauss_sigma','photoz','P(z)'], 
                 binmax=5*u.Mpc, cosmo=None):                     
    """ Cross-correlate lens and source tables
    create lens-source pairs table, returning for each  'id','ra','dec','e1','e2','sigma_e','zs','Pz','ra_l','dec_l','zl' """                
    raL, decL, zL =  lensColumns
    idS, raS, decS, e1, e2, esigma, zS, Pz =  sourceColumns

        
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=100, Om0=0.27,)

    galcoords = SkyCoord(source[raS],source[decS],unit='deg')
    source_stack = t.Table( t.hstack([source[sourceColumns][0:0], lens[lensColumns][0:0]]), names  = ('id','ra','dec','e1','e2','sigma_e','zs','Pz','ra_l','dec_l','zl') ) 
    source_stack['id'].dtype = 'int64'
    #source_stack['zsmean'] = ()
    for cluster in lens:
        cluster = np.array(cluster)
        z_cl = cluster[zL]
        clcenter = SkyCoord(cluster[raL],cluster[decL], unit = 'deg')
        Rarcmin = clcenter.separation(galcoords).to(u.arcmin)
        #if Rarcmin.min()>1*u.arcmin:
        #    continue
        Rmpc = (Rarcmin*cosmo.kpc_comoving_per_arcmin(z_cl)).to(u.Mpc) # in Mpc
        Iclgals = (Rmpc<=binmax) # binmax in Mpc
        if len(np.where(Iclgals)[0])==0:
            print "no galaxies for this cluster at redshift: %.3f " % z_cl
            continue
        print "cluster redshift: %.3f " % z_cl
        source_cl = t.Table( source[sourceColumns][Iclgals], names  = ('id','ra','dec','e1','e2','sigma_e','zs','Pz') )
        for nicklensColumn,lensColumn in zip(['ra_l','dec_l','zl'],lensColumns):
            source_cl[nicklensColumn] = cluster[lensColumn]    
        source_stack = t.vstack([source_stack,source_cl])
        del source_cl
        gc.collect()
    del source
    gc.collect()
    if len(source_stack)==0:
        raise TypeError("stacked table is empty")
    return source_stack
    
def DSigma(x, y, g1, g2, sig_g, zs, xc, yc, zl, wl=1., Pz=None, zvec=np.arange(0,7.01,0.01), cosmo=None, a=0.365, dz=0.2):
    """ DSigma_(x, y, g1, g2, sig_g, zs, xc, yc, zl, wl,  cosmo=None, a=0.365,):
        return shear
    """
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=100, Om0=0.27,)

    if Pz is None:
        Sigma_cr = cosmo_el.Sigma_crit_co(zl,zs,cosmo=cosmo).value
    else: 
        Sigma_cr = cosmo_el.Sigma_cr_co_PDF(zl, Pz, cosmo=cosmo, zvec=zvec)
    gt, gx, r = gTsky(x.to(u.rad),y.to(u.rad),g1,g2,xc.to(u.rad),yc.to(u.rad))
    r = ( (r.to(u.arcmin))*cosmo.kpc_comoving_per_arcmin(zl) ).to(u.Mpc)
    wg     =  1./(sig_g**2 + a**2)
    DSigma = gt * Sigma_cr #  gt -> DSigma
    Dx = gx * Sigma_cr # 45-rotated - zero consistency test
    #sig_S = sig_g * Sigma_cr # error of mass density
    sig_S = np.sqrt(sig_g**2 + rms_e**2) * Sigma_cr # error of mass density
    w = wg * wl * Sigma_cr**(-2)
    import ColorColor as CC
    Psum = CC.Pztot(zs,zl,Pz,zvec,dz=dz)
    return r,DSigma,Dx,sig_S,w,Sigma_cr,Psum
    
def stackSources_light(lens, source, 
                 lensColumns=['RADeg','decDeg','redshift'], 
                 sourceColumns=['id','ra2000','decl2000','shape_hsm_regauss_e1','shape_hsm_regauss_e2','shape_hsm_regauss_sigma','photoz','P(z)'], 
                 binmax=5*u.Mpc,cosmo=None,zvec=np.arange(0,7.01,0.01),dz=0.2,filename='temp.hdf5'):                     
    """ Cross-correlate lens and source tables
    create lens-source pairs table, returning for each  'id','ra','dec','e1','e2','sigma_e','zs','Pz','ra_l','dec_l','zl' """                
    raL, decL, zL =  lensColumns
    idS, raS, decS, e1, e2, esigma, zS, Pz =  sourceColumns

        
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=100, Om0=0.27,)
        
    lens[lensColumns[0]].unit = 'deg'; lens[lensColumns[1]].unit = 'deg'
    source[sourceColumns[1]].unit = 'deg'; source[sourceColumns[2]].unit = 'deg'
    
    galcoords = SkyCoord(source[raS],source[decS],unit='deg')
    
    First = True
    curr_length = 0
    for cluster in lens:
        cluster = np.array(cluster)
        z_cl = cluster[zL]
        clcenter = SkyCoord(cluster[raL],cluster[decL], unit = 'deg')
        Rarcmin = clcenter.separation(galcoords).to(u.arcmin)
        #if Rarcmin.min()>1*u.arcmin:
        #    continue
        Rmpc = (Rarcmin*cosmo.kpc_comoving_per_arcmin(z_cl)).to(u.Mpc) # in Mpc
        Iclgals = (Rmpc<=binmax) # binmax in Mpc
        if len(np.where(Iclgals)[0])==0:
            print "no galaxies for this cluster at redshift: %.3f " % z_cl
            continue
        print "cluster redshift: %.3f " % z_cl
        source_cl = t.Table( source[sourceColumns][Iclgals], names  = ('id','ra','dec','e1','e2','sigma_e','zs','Pz') )
        for nicklensColumn,lensColumn in zip(['ra_l','dec_l','zl'],lensColumns):
            source_cl[nicklensColumn] = cluster[lensColumn] 
        source_cl['ra_l'].unit=u.deg;source_cl['dec_l'].unit=u.deg;
        r,DSt,DSx,sig_S,w,Sigma_cr,Psum = DSigma(source_cl['ra'],source_cl['dec'],source_cl['e1'],source_cl['e2'],source_cl['sigma_e'],
                                                   source_cl['zs'],source_cl['ra_l'],source_cl['dec_l'],source_cl['zl'],Pz=source_cl['Pz'],zvec=zvec,cosmo=cosmo,dz=dz)
        source_cl_light = t.Table( [source_cl['id'],r,DSt,DSx,sig_S,w,Sigma_cr,Psum,source_cl['zs'],source_cl['zl'],source_cl['ra_l'],source_cl['dec_l']],names = ('object_id','R','dSigma+','dSigmax','dSigma_err','W','Sigma_cr','Psum','zs','zl','ra_l','dec_l') )            
        del source_cl                                       
        if First: # first lens in tract
                fbig = h5py.File(filename, "w")
                dtype = source_cl_light.dtype
                dset = fbig.create_dataset("stacked_tract", shape=(len(source)*len(lens),),maxshape=(len(source)*len(lens),), dtype=dtype)
                First = False
        # save to hdf5 file each loop
        dset[curr_length:curr_length+len(source_cl_light)] = np.array(source_cl_light)
        curr_length = curr_length+len(source_cl_light)
        del source_cl_light
        gc.collect()
        
    if curr_length==0: # will be no file to close?
        return 0
    else:
        dset.resize((curr_length,))
        fbig.close()
        return curr_length
            
    #return t.Table( [source_stack['id'],r,DSt,DSx,sig_S,w,Sigma_cr,Psum,source_stack['zs'],source_stack['zl']],names = ('R','dSigma+','dSigmax','dSigma_err','W','Sigma_cr','Psum','zs','zl') )

def DSigma_profile_light(sources, minr=0.1, maxr=10, nbinr=10, a=0.365, correctResponsivity=False,):
    """ DSigma_profile(sourcetable minr=0.1, maxr=10, nbinr=10, a=0.365, correctResponsivity=False):

    plot density contrast profile
    """
    if correctResponsivity: ## for regauss
        R = 1.-a**2
    else: ## no need for KSB
        R = 0.5 #??

    Inonan  = (~np.isnan(sources['dSigma+'])) # to remove nan photo-z's
    # binning
    Stw, Sterr, Sterr2, x1, xmin, xmax, Nbin, wbin, w2bin= usel.Wbinning2(sources['R'][Inonan], sources['dSigma+'][Inonan], sources['dSigma_err'][Inonan], sources['W'][Inonan], minr=minr, maxr=maxr, nbinr=nbinr)
    
    Sxw, Sxerr, Sxerr2, x1, xmin, xmax, Nbin, wbin, w2bin  = usel.Wbinning2(sources['R'][Inonan], sources['dSigmax'][Inonan], sources['dSigma_err'][Inonan], sources['W'][Inonan], minr=minr, maxr=maxr, nbinr=nbinr)
    # correct for responsivity    
    Stw, Sterr, Sterr2, Sxw, Sxerr, Sxerr2 = Stw/(2*R), Sterr/(2*R), Sterr2/(2*R), Sxw/(2*R), Sxerr/(2*R), Sxerr2/(2*R) # correct for resopnsivity
    xerr = (xmax-xmin)/2 # this is wrong in log-space.. good enough for presentation purposes?
    # weighted redshift?
    zlw, _, _, _, _, _, _, _, _ = usel.Wbinning(sources['R'][Inonan], sources['zl'][Inonan], sources['dSigma_err'][Inonan], sources['W'][Inonan], minr=minr, maxr=maxr, nbinr=nbinr)
    return t.Table( [x1, xerr, Stw, Sterr, Sterr2, Sxw, Sxerr, Sxerr2, Nbin, wbin, w2bin, zlw],
                  names = ('R','dR' , 'dSigma+', 'dSigma+_err (KSB+)', 'dSigma+_err', 'dSigmax', 'dSigmax_err (KSB+)', 'dSigmax_err', 'N', 'W','W2', 'zl_wmean') )

def DSigma_profile_light_calib(sources, minr=0.1, maxr=10, nbinr=10, ):
    """ DSigma_profile(sourcetab;e, minr=0.1, maxr=10, nbinr=10, ):

    plot density contrast profile
    """
    Inonan  = (~np.isnan(sources['dSigma+'])) # to remove nan photo-z's
    # binning
    Stw, Sterr, Sterr2, x1, xmin, xmax, Nbin, wbin, w2bin= usel.Wbinning2(sources['R'][Inonan], sources['dSigma+'][Inonan], sources['dSigma_err'][Inonan], sources['W'][Inonan], minr=minr, maxr=maxr, nbinr=nbinr)
    
    Sxw, Sxerr, Sxerr2, _, _, _, _, _, _  = usel.Wbinning2(sources['R'][Inonan], sources['dSigmax'][Inonan], sources['dSigma_err'][Inonan], sources['W'][Inonan], minr=minr, maxr=maxr, nbinr=nbinr)
    IS = sources['Sigma_cr']**(-1)
    var_e = sources['rms_e']**2
    ISw  = usel.WMbin(sources['R'][Inonan], IS[Inonan], sources['W'][Inonan], minr=minr, maxr=maxr, nbinr=nbinr)
    vare_w = usel.WMbin(sources['R'][Inonan], var_e[Inonan], sources['W'][Inonan], minr=minr, maxr=maxr, nbinr=nbinr)
    Kg   = usel.WMbin(sources['R'][Inonan], sources['m'][Inonan], sources['W'][Inonan], minr=minr, maxr=maxr, nbinr=nbinr)
    ct_w = usel.WMbin(sources['R'][Inonan], sources['ct'][Inonan]* sources['Sigma_cr'][Inonan], sources['W'][Inonan], minr=minr, maxr=maxr, nbinr=nbinr)
    cx_w = usel.WMbin(sources['R'][Inonan], sources['cx'][Inonan]* sources['Sigma_cr'][Inonan], sources['W'][Inonan], minr=minr, maxr=maxr, nbinr=nbinr)
    zlw  = usel.WMbin(sources['R'][Inonan], sources['zl'][Inonan],  sources['W'][Inonan], minr=minr, maxr=maxr, nbinr=nbinr)
    
    # correct for responsivity    
    Rg = 1.0 - vare_w
    Stw = 1./(2*Rg*(1.+Kg))*Stw \
         - 1./(1.+Kg) * ct_w
    Sterr2 = 1./(2*Rg*(1.+Kg))*Sterr2
    Sxw = 1/(2*Rg*(1.+Kg))*Sxw \
         - 1./(1.+Kg) * cx_w
    Sxerr2 = 1./(2*Rg*(1.+Kg))*Sxerr2
    Sterr = 1./(2*Rg*(1.+Kg))*Sterr
    Sxerr = 1./(2*Rg*(1.+Kg))*Sxerr
   
    xerr = (xmax-xmin)/2 # this is wrong in log-space.. good enough for presentation purposes?

    return t.Table( [x1, xerr,xmin,xmax, Stw, Sterr, Sterr2, Sxw, Sxerr, Sxerr2, Nbin, wbin, w2bin, zlw, ISw],
                  names = ('R','dR' ,'R1','R2', 'dSigma+', 'dSigma+_err', 'dSigma+_err2', 'dSigmax', 'dSigmax_err', 'dSigmax_err2', 'N', 'W','W2', 'zl_wmean','IS_wmean') )

def DSigma_profile_light_flexbin_calib(sources, bins, ):
    """ DSigma_profile(x, y, g1, g2, sig_g, zs, xc, yc, zl, wl, minr=0.1, maxr=10, nbinr=10,  marker = 'o', correctResponsivity=False):

    plot density contrast profile
    """
    Inonan  = (~np.isnan(sources['dSigma+'])) # to remove nan photo-z's
    # binning
    Stw, Sterr, Sterr2, Nbin, wbin, w2bin = usel.Wbinning_flex(sources['R'][Inonan], sources['dSigma+'][Inonan], sources['dSigma_err'][Inonan], sources['W'][Inonan], bins=bins)
    
    Sxw, Sxerr, Sxerr2, _, _, _,   = usel.Wbinning_flex(sources['R'][Inonan], sources['dSigmax'][Inonan], sources['dSigma_err'][Inonan], sources['W'][Inonan], bins=bins)
    IS = sources['Sigma_cr']**(-1)
    var_e = sources['rms_e']**2
    ISw  = usel.WMbinFlex(sources['R'][Inonan], IS[Inonan], sources['W'][Inonan], bins=bins)
    vare_w = usel.WMbinFlex(sources['R'][Inonan], var_e[Inonan], sources['W'][Inonan], bins=bins)
    Kg   = usel.WMbinFlex(sources['R'][Inonan], sources['m'][Inonan], sources['W'][Inonan], bins=bins)
    ct_w = usel.WMbinFlex(sources['R'][Inonan], sources['ct'][Inonan]* sources['Sigma_cr'][Inonan], sources['W'][Inonan], bins=bins)
    cx_w = usel.WMbinFlex(sources['R'][Inonan], sources['cx'][Inonan]* sources['Sigma_cr'][Inonan], sources['W'][Inonan], bins=bins)
    zlw  = usel.WMbinFlex(sources['R'][Inonan], sources['zl'][Inonan],  sources['W'][Inonan], bins=bins)
    
    # correct for responsivity    
    Rg = 1.0 - vare_w
    Stw = 1./(2*Rg*(1.+Kg))*Stw \
         - 1./(1.+Kg) * ct_w
    Sterr2 = 1./(2*Rg*(1.+Kg))*Sterr2
    Sxw = 1/(2*Rg*(1.+Kg))*Sxw \
         - 1./(1.+Kg) * cx_w
    Sxerr2 = 1./(2*Rg*(1.+Kg))*Sxerr2
   
    xerr = (bins[1:]-bins[:-1])/2 # this is wrong in log-space.. good enough for presentation purposes?
    xcen=(bins[1:]+bins[:-1])/2    


    return t.Table( [xcen, xerr, Stw, Sterr, Sterr2, Sxw, Sxerr, Sxerr2, Nbin, wbin, w2bin, zlw, ISw],
                  names = ('R','dR' , 'dSigma+', 'dSigma+_err (KSB+)', 'dSigma+_err', 'dSigmax', 'dSigmax_err (KSB+)', 'dSigmax_err', 'N', 'W','W2', 'zl_wmean','IS_wmean') )

def DSigma_profile_light_bs(sources, minr=0.1, maxr=10, nbinr=10, a=0.365, correctResponsivity=False,):
    """ DSigma_profile(x, y, g1, g2, sig_g, zs, xc, yc, zl, wl, minr=0.1, maxr=10, nbinr=10,  marker = 'o', correctResponsivity=False):

    plot density contrast profile
    """
    if correctResponsivity: ## for regauss
        R = 1.-a**2
    else: ## no need for KSB
        R = 0.5 #??

    Inonan  = (~np.isnan(sources['dSigma+'])) # to remove nan photo-z's
    # binning
    Stw, Sterr, Sterr2, x1, xmin, xmax, Nbin, wbin, w2bin= usel.Wbinning_bserr(sources['R'][Inonan], sources['dSigma+'][Inonan], sources['dSigma_err'][Inonan], sources['W'][Inonan], minr=minr, maxr=maxr, nbinr=nbinr)
    Sxw, Sxerr, Sxerr2, x1, xmin, xmax, Nbin, wbin, w2bin  = usel.Wbinning_bserr(sources['R'][Inonan], sources['dSigmax'][Inonan], sources['dSigma_err'][Inonan], sources['W'][Inonan], minr=minr, maxr=maxr, nbinr=nbinr)
    # correct for responsivity    
    Stw, Sterr, Sterr2, Sxw, Sxerr, Sxerr2 = Stw/(2*R), Sterr/(2*R), Sterr2/(2*R), Sxw/(2*R), Sxerr/(2*R), Sxerr2/(2*R) # correct for resopnsivity
    xerr = (xmax-xmin)/2 # this is wrong in log-space.. good enough for presentation purposes?
    # weighted redshift?
    zlw, _, _, _, _, _, _, _, _ = usel.Wbinning(sources['R'][Inonan], sources['zl'][Inonan], sources['dSigma_err'][Inonan], sources['W'][Inonan], minr=minr, maxr=maxr, nbinr=nbinr)
    return t.Table( [x1, xerr, Stw, Sterr, Sterr2, Sxw, Sxerr, Sxerr2, Nbin, wbin, w2bin, zlw],
                  names = ('R','dR' , 'dSigma+', 'dSigma+_err (BS)', 'dSigma+_err', 'dSigmax', 'dSigmax_err (BS)', 'dSigmax_err', 'N', 'W','W2', 'zl_wmean') )
                   
def DSigma_profile_light_tozero(sources, minr=0.1, maxr=10, nbinr=10, a=0.365, correctResponsivity=False,):
    """ DSigma_profile(x, y, g1, g2, sig_g, zs, xc, yc, zl, wl, minr=0.1, maxr=10, nbinr=10,  marker = 'o', correctResponsivity=False):

    plot density contrast profile
    """
    if correctResponsivity: ## for regauss
        R = 1.-a**2
    else: ## no need for KSB
        R = 0.5 #??

    Inonan  = (~np.isnan(sources['dSigma+'])) # to remove nan photo-z's
    # binning
    Stw, Sterr, Sterr2, x1, xmin, xmax, Nbin, wbin, w2bin= usel.Wbinning_tozero(sources['R'][Inonan], sources['dSigma+'][Inonan], sources['dSigma_err'][Inonan], sources['W'][Inonan], minr=minr, maxr=maxr, nbinr=nbinr)
    Sxw, Sxerr, Sxerr2, x1, xmin, xmax, Nbin, wbin, w2bin  = usel.Wbinning_tozero(sources['R'][Inonan], sources['dSigmax'][Inonan], sources['dSigma_err'][Inonan], sources['W'][Inonan], minr=minr, maxr=maxr, nbinr=nbinr)
    # correct for responsivity    
    Stw, Sterr, Sterr2, Sxw, Sxerr, Sxerr2 = Stw/(2*R), Sterr/(2*R), Sterr2/(2*R), Sxw/(2*R), Sxerr/(2*R), Sxerr2/(2*R) # correct for resopnsivity
    xerr = (xmax-xmin)/2 # this is wrong in log-space.. good enough for presentation purposes?
    # weighted redshift?
    zlw, _, _, _, _, _, _, _, _ = usel.Wbinning_tozero(sources['R'][Inonan], sources['zl'][Inonan], sources['dSigma_err'][Inonan], sources['W'][Inonan], minr=minr, maxr=maxr, nbinr=nbinr)
    return t.Table( [x1, xerr, Stw, Sterr, Sterr2, Sxw, Sxerr, Sxerr2, Nbin, wbin, w2bin, zlw],
                  names = ('R','dR' , 'dSigma+', 'dSigma+_err (KSB+)', 'dSigma+_err', 'dSigmax', 'dSigmax_err (KSB+)', 'dSigmax_err', 'N', 'W','W2', 'zl_wmean') )

def DSigma_profile_light_flexbin(sources, bins, a=0.365, correctResponsivity=False,):
    """ DSigma_profile(x, y, g1, g2, sig_g, zs, xc, yc, zl, wl, minr=0.1, maxr=10, nbinr=10,  marker = 'o', correctResponsivity=False):

    plot density contrast profile
    """
    if correctResponsivity: ## for regauss
        R = 1.-a**2
    else: ## no need for KSB
        R = 0.5 #??

    Inonan  = (~np.isnan(sources['dSigma+'])) # to remove nan photo-z's
    # binning
    Stw, Sterr, Sterr2, Nbin, wbin, w2bin= usel.Wbinning_flex(sources['R'][Inonan], sources['dSigma+'][Inonan], sources['dSigma_err'][Inonan], sources['W'][Inonan], bins=bins)
    Sxw, Sxerr, Sxerr2, Nbin, wbin, w2bin  = usel.Wbinning_flex(sources['R'][Inonan], sources['dSigmax'][Inonan], sources['dSigma_err'][Inonan], sources['W'][Inonan], bins=bins)
    # correct for responsivity    
    Stw, Sterr, Sterr2, Sxw, Sxerr, Sxerr2 = Stw/(2*R), Sterr/(2*R), Sterr2/(2*R), Sxw/(2*R), Sxerr/(2*R), Sxerr2/(2*R) # correct for resopnsivity
    xerr = (bins[1:]-bins[:-1])/2 # this is wrong in log-space.. good enough for presentation purposes?
    xcen=(bins[1:]+bins[:-1])/2    
    # weighted redshift?
    zlw, _, _, _, _, _,  = usel.Wbinning_flex(sources['R'][Inonan], sources['zl'][Inonan], sources['dSigma_err'][Inonan], sources['W'][Inonan], bins=bins)
    return t.Table( [xcen, xerr, Stw, Sterr, Sterr2, Sxw, Sxerr, Sxerr2, Nbin, wbin, w2bin, zlw],
                  names = ('R','dR' , 'dSigma+', 'dSigma+_err (KSB+)', 'dSigma+_err', 'dSigmax', 'dSigmax_err (KSB+)', 'dSigmax_err', 'N', 'W','W2', 'zl_wmean') )
                   
#%% number density profile   
def annulus_area(x,y, rbins, cellsize=None):
    """ A = annulus_area(x,y,cellsize=None, rbins)
    count area within annulus using grid method """
    r = np.hypot(x,y)
    if cellsize is None:
        minr = list()
        for xi,yi in zip(x,y):
            rsep = usel.radius(x[x!=xi],y[y!=yi],[xi,yi],1)
            minr.append(np.min(rsep[rsep>0]))
        cellsize = 2*np.percentile(minr,99) # determine gridsize from typical object separation
    spacelimit = np.array([[x.min(),x.max()],[y.min(),y.max()]])
    #print spacelimit
    bins = np.floor([(spacelimit[0,1]-spacelimit[0,0])/np.float64(cellsize), (spacelimit[1,1]-spacelimit[1,0])/np.float64(cellsize)])
    N, Y, X = np.histogram2d(y.flatten(), x.flatten(), bins=bins, range=spacelimit)
    R, Y, X = np.histogram2d(y.flatten(), x.flatten(), weights=r.flatten(), bins=bins, range=spacelimit)
    Rmat = R/N
    Nr, rbin = np.histogram(Rmat.flatten(),bins=rbins)
    Area = Nr * cellsize**2 
    return Area
    
def area(x,y,cellsize=1,spacelimit=None,debug=False):
    """A=area(x,y,cellsize,spacelimit)"""
    if spacelimit is None:
        spacelimit = np.array([[x.min(),x.max()],[y.min(),y.max()]])
    bins = np.floor([(spacelimit[0,1]-spacelimit[0,0])/np.float64(cellsize), (spacelimit[1,1]-spacelimit[1,0])/np.float64(cellsize)])
    N, xedges, yedges = np.histogram2d(x.flatten(), y.flatten(), bins=bins, range=spacelimit)
    if debug:
        print spacelimit
        plt.imshow(N,origin='low',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.colorbar()
    Area = len(N[N>0]) * cellsize**2
    return Area


def Nbin_weighted(x,w,bins):
    wsum, _, = np.histogram( x, weights = w, bins=bins)
    w2sum, _, = np.histogram( x, weights = w**2, bins=bins)
    N = (wsum)**2/w2sum
    return N
    
def n_profile(x, y, sig_g, center, zl=None, Pz=None, scale=1, minr=0.1, maxr=100, nbinr=10, cellsize=0.2, cosmo=None, a=0.365, jackknife=False, Nsim=100):
    rbins = np.logspace(np.log10(minr/minr.unit), np.log10(maxr/maxr.unit), nbinr+1)
    bin_min, bin_max, bincenter = usel.getBins(minr/minr.unit, maxr/maxr.unit, nbinr)*minr.unit
    unit = str(minr.unit)
    x = (x-center[0][:])*scale
    y = (y-center[1][:])*scale
    r = np.hypot(x,y)
    if zl is not None:
        x = ( (x.to(u.arcmin))*cosmo.kpc_comoving_per_arcmin(zl) ).to(getattr(u,unit))
        y = ( (y.to(u.arcmin))*cosmo.kpc_comoving_per_arcmin(zl) ).to(getattr(u,unit))
        r = ( (r.to(u.arcmin))*cosmo.kpc_comoving_per_arcmin(zl) ).to(getattr(u,unit))
        if Pz.ndim==1:
            Sigma_cr = cosmo_el.Sigma_crit_co(zl, Pz,cosmo=cosmo)
        elif Pz.ndim==2: 
            Sigma_cr = cosmo_el.Sigma_cr_co_PDF(zl, Pz, cosmo=cosmo)
    else:
        Sigma_cr = 1
                
    wg=  1./(sig_g**2 + a**2)    
    w = wg/(Sigma_cr**2)
    N = Nbin_weighted(r,w,rbins)
    Area  = annulus_area(x.value,y.value, rbins, cellsize) * minr.unit**2
    n = N/Area

    if jackknife:
        out=np.array([])
        s = len(x)
        for i in range(Nsim):
            print "jackknife # {0}".format(i)
            Rand = np.random.rand(s,1)
            Ind = np.int64(np.ceil(Rand*(s-1)))
            Nt = Nbin_weighted(r[Ind],w[Ind],rbins)
            Areat  = annulus_area(x.value[Ind],y.value[Ind], rbins, cellsize) * minr.unit**2
            ntemp = Nt/Areat
            out = np.append(out,ntemp)        
        dn = np.std(out.reshape([Nsim,len(ntemp)]),axis=0)
    else:
        dn = np.sqrt(N)/Area
    return n, dn, bincenter, (bin_max - bin_min)/2

def n_plot(r, dr, n, dn,**kwargs):

    plt.errorbar(r.value, n.value, yerr=dn.value, xerr=dr.value, marker = 's', ls = 'None',**kwargs)
    plt.xscale('log')
    plt.ylabel(r'$n_g\ [{0}]$'.format(n.unit))
    #plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.gca().xaxis.set_major_formatter(ticker.LogFormatter()) 
    plt.xlabel('R [{0}]'.format(r.unit))
    plt.xlim([r.value[0]-np.diff(r.value)[0], r.value[-1]+np.diff(r.value)[-1]])

####################################################
def crosscorr(lens, source, 
                 lensColumns=['RADeg','decDeg','redshift'], 
                 sourceColumns=['ra2000','decl2000'], 
                 binmax=5*u.Mpc, cosmo=None):                     
    """ Cross-correlate lens and source tables
    create lens-source pairs table, returning for each  'id','ra','dec','e1','e2','sigma_e','zs','Pz','ra_l','dec_l','zl' """                
    raL, decL, zL =  lensColumns
    raS, decS =  sourceColumns

        
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=100, Om0=0.27,)

    galcoords = SkyCoord(source[raS],source[decS],unit='deg')
    #source_stack = t.Table( data=None, names  = ('R','ra','dec','ra_l','dec_l','zl') ) 
    R = list()
    zl=list()
    for cluster in lens:
        #source_cl = t.Table( data=None, names  = ('R','ra','dec','ra_l','dec_l','zl') ) 
        cluster = np.array(cluster)
        z_cl = cluster[zL]
        clcenter = SkyCoord(cluster[raL],cluster[decL], unit = 'deg')
        Rarcmin = clcenter.separation(galcoords).to(u.arcmin)
        Rmpc = (Rarcmin*cosmo.kpc_comoving_per_arcmin(z_cl)).to(u.Mpc) # in Mpc
        Iclgals = (Rmpc<=binmax) # binmax in Mpc        
        if len(np.where(Iclgals)[0])==0:
            print "no galaxies for this cluster at redshift: %.3f " % z_cl
            continue
        print "cluster redshift: %.3f " % z_cl
        R.extend(Rmpc[Iclgals].value)
        zl.extend(z_cl*np.ones(np.sum(Iclgals)))
    del source
    gc.collect()
    return t.Table([np.array(R),np.array(zl)],names=('R','zl'))
    
    
def boost_corr(sample, rand, binmin,binmax,nbin):
    i=0
    n_zl = np.empty(len(np.unique(sample['zl'])))
    for zl in np.unique(sample['zl']):
        #n_zl[i] = np.sum(sample[sample['zl']==zl]['zs']>zl)*1./len(sample[sample['zl']==zl])
        n_zl[i] = np.sum(sample[sample['zl']==zl]['Psum']>0.98)*1./len(sample[sample['zl']==zl])
        i+=1
    i=0
    n_zl_r = np.empty([len(np.unique(sample['zl'])),nbin])
    bins = np.logspace(np.log10(binmin), np.log10(binmax), nbin+1)
    for zl in np.unique(sample['zl']):
        Nrand,_ =  np.histogram( rand[rand['zl']==zl]['R'],  bins=bins )
        #print Nrand
        W,_ =  np.histogram( sample[sample['zl']==zl]['R'], weights= sample[sample['zl']==zl]['W'], bins=bins )
        W2,_ =  np.histogram( sample[sample['zl']==zl]['R'], weights= sample[sample['zl']==zl]['W']**2, bins=bins )
        #Nsource = W**2/W2
        Nsource = W
        print Nsource
        n_zl_r[i] = Nsource*1./(Nrand)
        i+=1
    ClW = sample[(sample['R']>binmin) & (sample['R']<binmax)].group_by('zl').groups.aggregate(np.sum)['W']
    boost = np.nansum(n_zl_r.T/n_zl_r.T[-1]*ClW,axis=1)/np.nansum(ClW) ## weighted mean of clusters???
    return n_zl,n_zl_r, boost
 
   