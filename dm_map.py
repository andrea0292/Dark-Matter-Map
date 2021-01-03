import matplotlib.pyplot as plt
import os
import numpy as np
import healpy as hp
from tqdm import *
from astropy_healpix import HEALPix


class DM_map:
    def __init__(self):
        """
        Container class to compute the dark matter map to show with healpy
        """

        # Set constants, the first three are for the DM profile in the NFW and Burkert case

        self.rsun = 8.5 # in kpc
        self.rbstandard = 12.67 # kpc, see 1012.4515
        self.rsstandard = 24.42 # kpc
        self.cmperkpc = 3.08567758e21 # kpc to cm

    def fromGCtor(self, x, b, l): #Function to pass from longitudine and latidue to r. Returns r in kpc; x needs to be in kpc, b and l in radians
    
        return np.sqrt(x**2 + self.rsun**2 - 2 * x * self.rsun *np.cos(b)*np.cos(l))
        
    #Here I also define a function that given the resolution of my maps give me the corresponding values of the map in terms of b and l, 
    # and return also the area of a single pixel.

    def toGcoordinate(self, nside):
    
        npix = hp.nside2npix(nside) # number of pixels in the map 
    
        theta, phi = hp.pix2ang(nside, range(npix))
    
        bval = np.pi/2. - theta # checked with astropy_healpix; HPprova = HEALPix(nside=1024, order='ring') -> HPprova.healpix_to_lonlat([i])
        #lval = np.mod(phi+np.pi, 2.*np.pi)-np.pi
        pixarea = hp.nside2pixarea(nside)
    
        return bval, lval, pixarea

    def NFW(self, rs, gamma, x, b, l):
    
    
        def rho(r):
            ratio = r/rs
            return 1 / (ratio)**gamma / (1+ratio)**(3-gamma)
    
        norm = 0.4/ rho(self.rsun)
    
        rfunction = self.fromGCtor(x, b, l)
    
        return rho(rfunction) *norm

    def Burkert(self, rb, x, b, l):
    
        def rho(r):
            ratio = r/rb
            return 1 / (ratio) / (1+ratio)**2
    
        norm = 0.4/ rho(self.rsun)
    
        rfunction = self.fromGCtor(x, b, l)
    
        return rho(rfunction) *norm

    def integratedDM(self, whichprofile, rs, rb, gamma, b, l, xmax, dx): # dx = 0.01 seems to be a good value
    
        if whichprofile == 'NFW':
        
            def tointegrate(x):
            
                return self.NFW(rs, gamma, x, b, l)
        
        if whichprofile == 'Burkert':
        
            def tointegrate(x):
            
                return self.Burkert(rb, x, b, l)
        
        ary_x = np.arange(0,xmax,dx)
        tointegrate_ary = np.array([tointegrate(x) for x in ary_x])
    
        return self.cmperkpc * np.trapz(tointegrate_ary, ary_x) # in GeV/cm^2

    def DarkMap(self, nside, whichprofile, rs, rb, gamma, xmax, dx):
    
        #bvalues, lvalues, pixarea = self.toGcoordinate(nside)
        pixarea = hp.nside2pixarea(nside)
        npix = hp.nside2npix(nside)

        HP = HEALPix(nside=nside, order='ring')

        theta, phi = hp.pix2ang(nside, range(npix))

        bvalues = np.array([HP.healpix_to_lonlat([i])[1][0].value for i in range(len(theta))])
        lvalues = np.array([HP.healpix_to_lonlat([i])[0][0].value for i in range(len(phi))])
    

        mappa_ary = np.array([self.integratedDM(whichprofile, rs, rb, gamma, bvalues[p], lvalues[p], xmax, dx) for p in range(npix)])
        
        # Multiply by area of pixel to add sr unit
        mappa_ary *= pixarea
    
        return hp.ud_grade(np.array(mappa_ary ), 128, power=-2) # See https://healpy.readthedocs.io/en/latest/generated/healpy.pixelfunc.ud_grade.html
  

    