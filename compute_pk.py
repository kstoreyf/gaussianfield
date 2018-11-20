import numpy as np
from matplotlib import pyplot as plt
from numpy import fft as fftengn

import gaussian_random as gr



pickle_dir = 'pickles_2018-11-05/'
plot_dir = 'plots_2018-11-05/'


def main():
    L = 40
    n = 64
    #n = 16384
    #n = 4096
    #n = 64
    d = 3
    #pstag = '_gaussian'
    pstag = ''
    #pstag = '_bump'
    tag = '_n{}_L{}_{}d{}'.format(n, L, d, pstag)
    if pstag=='':
        ps = gr.powspec
    elif pstag=='_bump':
        ps = gr.powspec_bump
    elif pstag=='_gaussian':
        ps = gr.powspec_gaussian


    k = gr.get_ks(L, n, d, full=True, shift=False)

    pktrue = ps(abs(k))
    cftrue = pk_to_2pcf(k, pktrue, L, n, d)

    # Calculate output power spectrum from density field
    dens = gr.load_field('density'+tag+'.p'.format(n, L, d))
    #pk = np.abs(fftengn.fftshift(fftengn.fftn(dens)))**2
    pk = np.abs(fftengn.fftn(dens))**2

    #scale
    boxvol = float(L)**d
    pix = (float(L)/float(n))**d
    pk *= pix**2 / boxvol

    #pk *= 1./boxvol

    # Calculate correlation function from output power spectrum
    cf = pk_to_2pcf(k, pk, L, n, d)

    r = get_rs(L, n, d, full=False)

    nbins = 200
    kavg, pktrueavg = radial_average(k, pktrue, nbins=nbins)
    kavg, pkavg = radial_average(k, pk, nbins=nbins)

    r = cut_half(r, n, d)
    cf = cut_half(cf, n, d)
    cftrue = cut_half(cftrue, n, d)

    ravg, cftrueavg = radial_average(r, np.abs(cftrue), nbins=nbins)
    ravg, cfavg = radial_average(r, np.abs(cf), nbins=nbins)


    # PLOT
    plotft = True

    # P(k)
    plt.figure()
    #plt.plot(kshift[n/2:], abs(kshift[n/2:])**3*pktrue[:n/2], 'k-', label='True (pos k, p<n/2)')
    plt.plot(kavg, pktrueavg, 'k-', label='True (pos k, p<n/2)')

    if plotft:
        plt.plot(kavg, pkavg, c='purple', label='Meausured (pos k, p<n/2)')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.legend()

    # P(k) - log
    plt.figure()
    #plt.loglog(kshift[n/2:], abs(kshift[n/2:])**3*pktrue[:n/2], 'k-', label='True (pos k)')
    plt.loglog(kavg, pktrueavg, 'k-', label='True (pos k)')
    if plotft:
        plt.loglog(kavg, pkavg, c='purple', label='Meausured (pos k)')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.legend()

    # xi(r)
    plt.figure()
    plt.plot(ravg, np.real(cftrueavg), 'b-', label='True real')
    plt.plot(ravg, np.imag(cftrueavg), 'r-', label='True imag')
    plt.plot(ravg, np.abs(cftrueavg), 'k-', label='True (all r)')
    if plotft:
        plt.plot(ravg, np.real(cfavg), 'c-', label='Measured real')
        plt.plot(ravg, np.imag(cfavg), c='hotpink', label='Measured imag')
        plt.plot(ravg, np.abs(cfavg), c='purple', label='Measured (all r)')
    plt.xlabel('r')
    plt.ylabel(r'$\xi$(r)')
    plt.legend()

    # xi(r) - log
    plt.figure()
    plt.loglog(ravg, np.real(cftrueavg), 'b-')
    plt.loglog(ravg, np.imag(cftrueavg), 'r-')

    plt.loglog(ravg, np.abs(cftrueavg), 'k-', label='True (r,cf<n/2)')
    if plotft:
        plt.loglog(ravg, np.abs(cfavg), c='purple', label='Measured (r,cf<n/2)')
    plt.xlabel('r')
    plt.ylabel(r'$\xi$(r)')
    plt.legend()

    # plt.figure()
    # plt.loglog(r, np.abs(cftrue), 'k-', label='True (r,cf<n/2)')
    # if plotft:
    #     plt.loglog(r, np.abs(cf), 'm-', label='Measured (r,cf<n/2)')
    # plt.xlabel('r')
    # plt.ylabel(r'$\xi$(r)')
    # plt.legend()


    # Plot power spectrum
    # karr = [kshift, k]
    # pkarr = [pktrue, pk]
    # labels = ['True', 'FT']
    # cols = ['k-', 'm-']
    #plot_pk(karr, pkarr, labels, cols, saveto='powspec'+tag+'.png')

    # Plot 2PCF
    # rarr = [r, r]
    # cfarr = [cftrue, cf]
    # labels = ['True', 'FT']
    # cols = ['k-', 'm-']
    #plot_cf(rarr, cfarr, labels, cols, saveto='cf'+tag+'.png')

    plt.show()



def pk_to_2pcf(k, pk, L, n, d):
    print 'Pk tp 2PCF'
    cf = fftengn.ifftn(fftengn.fftshift(pk))

    #scale
    pix = (float(L)/float(n))**d
    cf *= 1./pix

    return cf


def cut_half(arr, n, d):
    d = int(d)
    assert d in [1,2,3], 'd must be 1, 2, or 3'
    if d==1:
        arr = arr[:n/2+1]
        #return arr
    elif d==2:
        return arr[:,:n/2+1]
        #return arr[:n/2+1,:n/2+1]
    elif d==3:
        return arr[:,:,:n/2+1]


# TODO: not going down to zero k?
def radial_average(x, y, nbins=100):
    print 'Averaging power'

    xflat = x.flatten()
    yflat = y.flatten()
    xmin = np.min(xflat[np.nonzero(xflat)])
    xmax = np.max(xflat[np.isfinite(xflat)])

    # do i have an edge issue here?
    xbins = np.linspace(xmin, xmax, nbins)
    xavg = (xbins[:-1]+xbins[1:])/2.
    print min(xbins), max(xbins)
    print min(xavg), max(xavg)

    pos = np.digitize(xflat, xbins)
    ybinned = [[] for _ in range(len(xavg))]
    for i in range(len(pos)):
        ibin = pos[i]-1
        if 0<=ibin<len(ybinned):
            ybinned[ibin].append(yflat[i])

    yavg = [np.mean(ybin) for ybin in ybinned]
    return xavg, yavg


def get_rs3d(L, n, full=False):
    assert n % 2 == 0
    space = 2.*np.pi/float(L)
    rx = np.linspace(0, 2./space, n)
    ry = np.linspace(0, 2./space, n)
    rz = np.linspace(0, 2./space, n)
    if not full: #???
        rz = rz[:n/2+1]
    r = np.sqrt(rx[:,np.newaxis][np.newaxis,:]**2
                    + ry[:,np.newaxis][:,np.newaxis]**2
                    + rz[np.newaxis,:][np.newaxis,:]**2)
    return r


def get_rs2d(L, n, full=False):
    assert n % 2 == 0
    space = 2.*np.pi/float(L)
    rx = np.linspace(0, 2./space, n)
    ry = np.linspace(0, 2./space, n)
    if not full: #???
        ry = ry[:n/2+1]
    r = np.sqrt(rx[:,np.newaxis]**2
              + ry[np.newaxis,:]**2)
    return r


def get_rs1d(L, n, full=False):
    assert n % 2 == 0
    space = 2.*np.pi/float(L)
    rx = np.linspace(0, 2./space, n)
    if not full: #???
        rx = rx[:n/2+1]
    r = abs(rx) #should be unnecessary but for consistency
    return r


def get_rs(L, n, d, full=False):
    print 'Computing ks'
    d = int(d)
    assert d in [1,2,3], 'd must be 1, 2, or 3'
    if d==1:
        return get_rs1d(L, n, full=full)
    elif d==2:
        return get_rs2d(L, n, full=full)
    else:
        return get_rs3d(L, n, full=full)


def plot_pk(k, Pk, label, col, saveto=None):
    plt.figure()
    ksize = len(k)
    if ksize==1:
        k = [k]
        Pk = [Pk]
        label = [label]
        col = [col]
    for i in range(ksize):
        plt.loglog(k[i], Pk[i], col[i], label=label[i])
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.legend()
    if saveto:
        plt.savefig(plot_dir+saveto)


def plot_cf(r, cf, label, col, saveto=None):
    plt.figure()
    rsize = len(r)
    if rsize==1:
        r = [r]
        cf = [cf]
        label = [label]
        col = [col]
    for i in range(rsize):
        # rcf = np.array([rrcf for rrcf in zip(r[i], cf[i]) if rrcf[1]>0])
        # rr = rcf[:,0]
        # cfcf = rcf[:,1]
        plt.loglog(r[i], cf[i], col[i], label=label[i])
    plt.xlabel('r')
    plt.ylabel(r'$\xi$(r)')
    plt.legend()
    if saveto:
        plt.savefig(plot_dir+saveto)


if __name__=='__main__':
    main()