import numpy as np
from matplotlib import pyplot as plt
from numpy import fft as fftengn

import gaussian_random as gr



def main():
    L = 40
    n = 64
    d = 3

    # Calculate true power spectrum
    ktrue = np.logspace(np.log10(1e-3), np.log10(1e2), 50)
    pktrue = gr.powspec(ktrue)

    # Calculate true correlation function
    rtrue, cftrue = pk_to_2pcf(ktrue, pktrue, d)

    # Calculate output power spectrum from density field

    dens = gr.load_field('density_n{}_L{}_{}d.p'.format(n, L, d))
    dens /= 1./(2*np.pi)**d
    # fft to properly count pixels
    pk = np.abs(fftengn.fftshift(fftengn.fftn(dens)))**2

    #scale
    boxvol = np.float(L)**3
    pixelsize = boxvol/np.float(n)**3
    unscale = boxvol/pixelsize**2
    pk *= unscale

    # get all frequency values
    k = gr.get_ks(L, n, d, full=True)

    kavg, pkavg = radial_average(k, pk)
    plot_pk(*zip([ktrue, pktrue], [kavg, pkavg]))


    # Calculate correlation function from output power spectrum
    r, cf = pk_to_2pcf(kavg, pkavg, d)
    plot_cf(*zip([rtrue, cftrue], [r, cf]))

    plt.show()


def plot_cf(r, cf):
    plt.figure()
    rsize = np.array(r).shape[0]
    if rsize==1:
        r = [r]
        cf = [cf]
    for i in range(rsize):
        rcf = np.array([rrcf for rrcf in zip(r[i], cf[i]) if rrcf[1]>0])
        rr = rcf[:,0]
        cfcf = rcf[:,1]
        plt.loglog(rr, cfcf)
    plt.xlabel('r')
    plt.ylabel(r'$\xi$(r)')


def pk_to_2pcf(k, pk, d):
    print 'Pk tp 2PCF'
    ksps = np.array([kkpp for kkpp in zip(k, pk) if np.isfinite(kkpp[1])])
    ks = ksps[:,0]
    ps = ksps[:,1]
    cf = 1./(2*np.pi)**d * fftengn.fftshift(fftengn.irfft(ps))
    r = 2.*np.pi/ks
    return r, cf


def radial_average(x, y):
    print 'Averaging power'

    #x, y = zero_odds(x,y)

    xflat = x.flatten()
    yflat = y.flatten()
    xmin = np.min(xflat[np.nonzero(xflat)])
    xmax = np.max(xflat[np.isfinite(xflat)])

    xbins = np.logspace(np.log10(xmin), np.log10(xmax), 80)
    xavg = (xbins[:-1]+xbins[1:])/2.

    pos = np.digitize(xflat, xbins)
    ybinned = [[] for _ in range(len(xavg))]
    for i in range(len(pos)):
        ibin = pos[i]-1
        if 0<=ibin<len(ybinned):
            ybinned[ibin].append(yflat[i])

    yavg = [np.mean(ybin) for ybin in ybinned]
    return xavg, yavg


def zero_odds(x, y):
    n = x.shape[0]
    kx = fftengn.fftshift(fftengn.fftfreq(n, 1./n))
    ky = fftengn.fftshift(fftengn.fftfreq(n, 1./n))
    kz = fftengn.fftshift(fftengn.fftfreq(n, 1./n))[:n/2+1]
    #mask = np.zeros((n,n,n))
    for i in range(len(kx)):
        for j in range(len(ky)):
            for k in range(len(kz)):
                if kx[i]%2!=0 or ky[j]%2!=0 or kz[k]%2!=0:
                    x[i][j][k] = 0
                    y[i][j][k] = 0
    return x, y


def plot_pk(k, Pk):
    plt.figure()
    ksize = np.array(k).shape[0]
    if ksize==1:
        k = [k]
        Pk = [Pk]
    for i in range(ksize):
        plt.loglog(k[i], Pk[i])
    plt.xlabel('k')
    plt.ylabel('P(k)')


if __name__=='__main__':
    main()