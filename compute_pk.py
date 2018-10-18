import numpy as np
from matplotlib import pyplot as plt
from numpy import fft as fftengn

import gaussian_random as gr



def main():
    L = 40
    n = 256
    d = 3

    klog = np.logspace(np.log10(1e-3), np.log10(1), 50)
    plot_pk(klog, gr.powspec(klog))

    dens = gr.load_field('density_n{}_L{}_{}d.p'.format(n, L, d))

    pk = np.abs(fftengn.fftshift(fftengn.rfftn(dens)))**2
    k = gr.get_ks(L, n, d)

    kavg, pkavg = avg_power(k, pk)
    plot_pk(kavg, pkavg)
    gr.plot_pixel_hist(dens)

    plt.show()



def avg_power(k, pk):
    print 'Averaging power'
    kflat = k.flatten()
    pflat = pk.flatten()
    kmin = np.min(kflat[np.nonzero(kflat)])
    kmax = np.max(kflat)

    kbins = np.logspace(np.log10(kmin), np.log10(kmax), 50)

    k_avg = (kbins[:-1]+kbins[1:])/2.

    pos = np.digitize(kflat, kbins)

    pk_binned = [[] for _ in range(len(k_avg))]
    for i in range(len(pos)):
        ibin = pos[i]-1
        if 0<=ibin<len(pk_binned):
            pk_binned[ibin].append(pflat[i])

    pk_avg = [np.mean(pkbin) for pkbin in pk_binned]

    return k_avg, pk_avg


def plot_pk(k, Pk):
    plt.figure()
    plt.loglog(k, Pk)
    plt.xlabel('k')
    plt.ylabel('P(k)')


if __name__=='__main__':
    main()