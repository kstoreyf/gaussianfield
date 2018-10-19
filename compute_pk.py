import numpy as np
from matplotlib import pyplot as plt
from numpy import fft as fftengn

import gaussian_random as gr



def main():
    L = 40
    n = 256
    d = 3

    # Plot true power spectrum
    klog = np.logspace(np.log10(1e-3), np.log10(1e2), 50)
    pktrue = gr.powspec(klog)
    plot_pk(klog, pktrue)

    # Plot true correlation function
    rtrue, cftrue = pk_to_2pcf(klog, pktrue)
    plot_cf(rtrue, cftrue)

    # Calculate output power spectrum from density field
    dens = gr.load_field('density_n{}_L{}_{}d.p'.format(n, L, d))
    pk = np.abs(fftengn.fftshift(fftengn.rfftn(dens)))**2
    pk *= float(n)**(2*d)/(L**d) #rescale
    k = gr.get_ks(L, n, d)
    kavg, pkavg = radial_average(k, pk)
    plot_pk(kavg, pkavg)

    # Calculate correlation function from output power spectrum
    r, cf = pk_to_2pcf(kavg, pkavg)
    plot_cf(r, cf)

    plt.show()


def plot_cf(r, cf):
    plt.figure()
    rcf = np.array([rrcf for rrcf in zip(r, cf) if rrcf[1]>0])
    rr = rcf[:,0]
    cfcf = rcf[:,1]
    plt.loglog(rr, cfcf)
    plt.xlabel('r')
    plt.ylabel(r'$\xi$(r)')


def pk_to_2pcf(k, pk):
    print 'Pk tp 2PCF'
    ksps = np.array([kkpp for kkpp in zip(k, pk) if np.isfinite(kkpp[1])])
    ks = ksps[:,0]
    ps = ksps[:,1]
    cf = 1./(2*np.pi)**3 * fftengn.fftshift(fftengn.irfft(ps))
    r = 2.*np.pi/ks
    return r, cf


def radial_average(x, y):
    print 'Averaging power'
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


def plot_pk(k, Pk):
    plt.figure()
    plt.loglog(k, Pk)
    plt.xlabel('k')
    plt.ylabel('P(k)')


if __name__=='__main__':
    main()