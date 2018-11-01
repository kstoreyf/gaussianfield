import numpy as np
from matplotlib import pyplot as plt
from numpy import fft as fftengn

import gaussian_random as gr



pickle_dir = 'pickles_2018-10-28/'
plot_dir = 'plots_2018-10-28/'


def main():
    L = 40
    n = 256
    d = 3
    #tag = '_n{}_L{}_{}d'.format(n, L, d)
    tag = '_n{}_L{}_{}d_bump'.format(n, L, d)

    # Calculate true power spectrum
    ktrue = np.logspace(np.log10(1e-3), np.log10(1e2), 50)
    #pktrue = gr.powspec(ktrue)
    pktrue = gr.powspec_bump(ktrue)

    # Calculate true correlation function
    rtrue, cftrue = pk_to_2pcf(ktrue, pktrue, L, n, d)

    # Calculate output power spectrum from density field
    dens = gr.load_field('density'+tag+'.p'.format(n, L, d))
    # fft (not rfft) to properly count pixels
    pk = np.abs(fftengn.fftshift(fftengn.fftn(dens)))**2

    #scale
    boxvol = np.float(L)**d
    pix = (float(L)/float(n))**d
    pk *= pix**2 / boxvol

    # get all frequency values
    k = gr.get_ks(L, n, d, full=True)

    kavg, pkavg = radial_average(k, pk)
    ks = [ktrue, kavg]
    Pks = [pktrue, pkavg]
    labels = ['True', 'FT']
    plot_pk(ks, Pks, labels, saveto='powspec'+tag+'.png')


    # Calculate correlation function from output power spectrum
    r, cf = pk_to_2pcf(kavg, pkavg, L, n, d)
    rs = [rtrue, r]
    cfs = [cftrue, cf]
    labels = ['True', 'FT']
    plot_cf(rs, cfs, labels, saveto='cf'+tag+'.png')

    plt.show()



def pk_to_2pcf(k, pk, L, n, d):
    print 'Pk tp 2PCF'
    ksps = np.array([kkpp for kkpp in zip(k, pk) if np.isfinite(kkpp[1])])
    ks = ksps[:,0]
    ps = ksps[:,1]

    cf = fftengn.fftshift(fftengn.irfft(ps))

    #scale
    boxvol = np.float(L)**d
    pix = (float(L)/float(n))**d
    cf *= 1./pix * 1./boxvol
    cf *= 1./(2*np.pi)**d

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
    for i in range(len(kx)):
        for j in range(len(ky)):
            for k in range(len(kz)):
                if kx[i]%2!=0 or ky[j]%2!=0 or kz[k]%2!=0:
                    x[i][j][k] = 0
                    y[i][j][k] = 0
    return x, y


def plot_pk(k, Pk, label, saveto=None):
    plt.figure()
    ksize = len(k)
    if ksize==1:
        k = [k]
        Pk = [Pk]
        label = [label]
    for i in range(ksize):
        plt.loglog(k[i], Pk[i], label=label[i])
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.legend()
    if saveto:
        plt.savefig(plot_dir+saveto)


def plot_cf(r, cf, label, saveto=None):
    plt.figure()
    rsize = len(r)
    if rsize==1:
        r = [r]
        cf = [cf]
        label = [label]
    for i in range(rsize):
        rcf = np.array([rrcf for rrcf in zip(r[i], cf[i]) if rrcf[1]>0])
        rr = rcf[:,0]
        cfcf = rcf[:,1]
        plt.loglog(rr, cfcf, label=label[i])
    plt.xlabel('r')
    plt.ylabel(r'$\xi$(r)')
    plt.legend()
    if saveto:
        plt.savefig(plot_dir+saveto)


if __name__=='__main__':
    main()