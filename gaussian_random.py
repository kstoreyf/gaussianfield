from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import fft as fftengn
from scipy.optimize import curve_fit


pickle_dir = 'pickles_2018-11-05/'
plot_dir = 'plots_2018-11-05/'


def main():

    L = 40 #length scale
    #n = 4096 #pixels on side of density field
    #n = 16384
    n = 256
    d = 2 #dimensions
    N = int(1e3) #num samples

    generate_field(L, n, d)
    #sample_field(L, n, N, d)
    plt.show()

def generate_field(L, n, d):
    #pstag = '_gaussian'
    pstag = ''
    #pstag = '_bump'
    tag = '_n{}_L{}_{}d{}'.format(n, L, d, pstag)
    if pstag=='':
        ps = powspec
    elif pstag=='_bump':
        ps = powspec_bump
    elif pstag=='_gaussian':
        ps = powspec_gaussian


    k = get_ks(L, n, d)
    Pk = ps(k)
    print np.mean(Pk), np.min(Pk), np.max(Pk)

    a = get_amplitudes(L, n, Pk, d)
    a2 = a*np.conj(a)
    print np.mean(a2), np.min(a2), np.max(a2)
    a3 = np.abs(a)**2
    print np.mean(a3), np.min(a3), np.max(a3)

    dens = density_field(a, L, n, d)
    print np.mean(dens), np.min(dens), np.max(dens)

    plot_pixel_hist(dens, fit=True)
    plot_density(dens, L, n, d, avg=False, saveto='density'+tag+'.png')
    plot_density(dens, L, n, d, avg=True, saveto='density_avg'+tag+'.png')
    plot_tiled(dens, L, n, d, avg=True, saveto='density_tiled'+tag+'.png')
    save_field(dens, 'density'+tag+'.p')


def sample_field(L, n, N, d):
    tag = '_n{}_L{}_{}d'.format(n, L, d)
    dens = load_field('density'+tag+'.p')
    q = exponentiate(dens, N, fac=5e7)
    samples = generate_samples(L, N, d)
    sampled_field, rejects = reject_samples(samples, q, L, n, d)
    plot_samples(sampled_field, rejects, d)
    save_field(sampled_field, 'samples'+tag+'_N{}.p'.format(N))


def get_ks(L, n, d, full=False, shift=True):
    print 'Computing ks'
    d = int(d)
    assert d in [1,2,3], 'd must be 1, 2, or 3'
    if d==1:
        return get_ks1d(L, n, full=full, shift=shift)
    elif d==2:
        return get_ks2d(L, n, full=full, shift=shift)
    else:
        return get_ks3d(L, n, full=full, shift=shift)


def get_amplitudes(L, n, Pk, d):
    print 'Computing amplitudes'
    d = int(d)
    assert d in [1,2,3], 'd must be 1, 2, or 3'
    if d==1:
        return get_amplitudes1d(L, n, Pk)
    elif d==2:
        return get_amplitudes2d(L, n, Pk)
    else:
        return get_amplitudes3d(L, n, Pk)

def density_field(a, L, n, d):
    print 'Transforming amplitudes to density field'

    boxvol = float(L)**d
    pix = (float(L)/float(n))**d

    #a *= np.sqrt(boxvol/pix**2)
    #dens = fftengn.ifftn(a)
    #dens = fftengn.ifftn(a) * boxvol**(3./2.) / pix
    dens = fftengn.ifftn(a) * boxvol ** (1./2.) / pix
    #dens = fftengn.ifftn(a) * boxvol / pix
    #dens *= 1./np.sqrt(2*np.pi)**d
    print 'dens:',np.mean(dens), np.min(dens), np.max(dens)
    assert np.max(abs(np.imag(dens)))<1e-10, 'Density field should be entirely real'
    return np.real(dens)


def get_ks3d(L, n, full=False, shift=True):
    assert n % 2 == 0
    kx = fftengn.fftfreq(n, 1./n)
    ky = fftengn.fftfreq(n, 1./n)
    kz = fftengn.fftfreq(n, 1./n)
    if shift:
        kx = fftengn.fftshift(kx)
        ky = fftengn.fftshift(ky)
        kz = fftengn.fftshift(kz)
    if not full:
        kz = kz[:n/2+1]
    k = 2.*np.pi/L * np.sqrt(kx[:,np.newaxis][np.newaxis,:]**2
                           + ky[:,np.newaxis][:,np.newaxis]**2
                           + kz[np.newaxis,:][np.newaxis,:]**2)
    return k


def get_amplitudes3d(L, n, Pk):
    areal = np.zeros((n,n,n))
    aim = np.zeros((n,n,n))
    for i in range(n):
        for j in range(n):
            for g in range(n/2+1):
                pk = Pk[i][j][g]
                if (i==0 or i==n/2) and (j==0 or j==n/2) and (g==0 or g==n/2):
                    areal[i][j][g] = np.random.normal(0, np.sqrt(pk))
                    aim[i][j][g] = 0
                else:
                    areal[i][j][g] = 2**(-0.5) * np.random.normal(0, np.sqrt(pk))
                    aim[i][j][g] = 2**(-0.5) * np.random.normal(0, np.sqrt(pk))
                    areal[(n-i)%n][(n-j)%n][(n-g)%n] = areal[i][j][g]
                    aim[(n-i)%n][(n-j)%n][(n-g)%n] = -aim[i][j][g]
    a = areal + 1.0j*aim
    a = fftengn.ifftshift(a)
    return a #(h/Mpc)**3



def get_ks2d(L, n, full=False, shift=True):
    assert n % 2 == 0
    kx = fftengn.fftfreq(n, 1./n)
    ky = fftengn.fftfreq(n, 1./n)
    if shift:
        kx = fftengn.fftshift(kx)
        ky = fftengn.fftshift(ky)
    if not full:
        ky = ky[:n/2+1]
    k = 2.*np.pi/L * np.sqrt(kx[:,np.newaxis]**2
                           + ky[np.newaxis,:]**2)
    return k


def get_amplitudes2d(L, n, Pk):
    areal = np.zeros((n,n))
    aim = np.zeros((n,n))
    for i in range(n):
        for j in range(n/2+1):
            pk = Pk[i][j]
            if (i==0 or i==n/2) and (j==0 or j==n/2):
                areal[i][j] = np.random.normal(0, np.sqrt(pk))
                aim[i][j] = 0
            else:
                areal[i][j] = 2**(-0.5) * np.random.normal(0, np.sqrt(pk))
                aim[i][j] =  2**(-0.5) * np.random.normal(0, np.sqrt(pk))
                areal[(n-i)%n][(n-j)%n] = areal[i][j]
                aim[(n-i)%n][(n-j)%n] = -aim[i][j]
    a = areal + 1.0j*aim
    a = fftengn.ifftshift(a)
    return a # (h/Mpc)**2


def get_ks1d(L, n, full=False, shift=True):
    assert n % 2 == 0
    kx = fftengn.fftfreq(n, 1./n)
    if shift:
        kx = fftengn.fftshift(kx)
    if not full:
        kx = kx[:n/2+1]
    k = 2.*np.pi/float(L) * abs(kx)
    return k


def get_amplitudes1d(L, n, Pk):
    areal = np.zeros(n)
    aim = np.zeros(n)
    for i in range(n/2+1):
        pk = Pk[i]
        if (i==0 or i==n/2):
            areal[i] = np.random.normal(0, np.sqrt(pk))
            aim[i] = 0
        else:
            areal[i] = 2**(-0.5) * np.random.normal(0, np.sqrt(pk))
            aim[i] = 2**(-0.5) * np.random.normal(0, np.sqrt(pk))
            areal[(n-i)%n] = areal[i]
            aim[(n-i)%n] = -aim[i]
    a = areal + 1.0j*aim
    a = fftengn.ifftshift(a)
    return a #(h/Mpc)


def exponentiate(dens, N, fac):
    print 'Exponentiating density field'
    q = np.exp(fac*dens)
    qrenorm = q/np.max(q)
    assert np.max(qrenorm)==1
    return qrenorm


def save_field(field, fn):
    field = np.array(field)
    field.dump(pickle_dir+fn)


def load_field(fn):
    return np.load(pickle_dir+fn)


def plot_density(dens, n, L, d, avg=True, saveto=None):
    plt.figure()
    if d==1:
        plt.plot(dens)
    if d>1:
        if d==3:
            if avg:
                dens_pix = np.mean(dens[0:12], axis=0)
            else:
                dens_pix = dens[0]

        if d==2:
            dens_pix = dens
        ax = plt.gca()
        cax = ax.imshow(dens_pix)
        maxd = np.max(abs(dens_pix))
        cax.set_clim(-maxd, maxd)
        plt.colorbar(cax)
        plt.xlabel('x')
        plt.ylabel('y')
    if saveto:
        plt.savefig(plot_dir+saveto)


def plot_tiled(dens, L, n, d, avg=True, saveto=None):
    plt.figure()
    if d==1:
        plt.plot(np.tile(dens, 2))
    if d>1:
        if d==3:
            if avg:
                dens_pix = np.mean(dens[0:12], axis=0)
            else:
                dens_pix = dens[0]
        if d==2:
            dens_pix = dens
        ax = plt.gca()
        cax = ax.imshow(np.tile(dens_pix, (2,2)))
        maxd = np.max(abs(dens_pix))
        cax.set_clim(-maxd, maxd)
        plt.colorbar(cax)
        plt.xlabel('x')
        plt.ylabel('y')
    if saveto:
        plt.savefig(plot_dir+saveto)


def reject_samples(samples, q, L, n, d):
    print "Rejecting samples"
    sampled_field = []
    rejects = []
    count = 0
    for s in samples:
        count += 1
        px = int(np.floor(s[0]*n/float(L)))
        if d>=2:
            py = int(np.floor(s[1]*n/float(L)))
            print px, py
        if d==3:
            pz = int(np.floor(s[2]*n/float(L)))

        if d==1:
            qval = q[px]
        elif d==2:
            qval = q[px][py]
        elif d==3:
            qval = q[px][py][pz]
        rand = np.random.uniform()
        if qval > rand:
            sampled_field.append(s)
        else:
            rejects.append(s)
        if count%10000==0:
            print count
    return sampled_field, rejects


def plot_samples(samples, rejects, d):
    fig = plt.figure()

    if d==3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    print len(zip(*samples))
    ax.scatter(*zip(*samples), s=1, c='limegreen')

    if len(rejects)>0:
        ax.scatter(*zip(*rejects), s=1, c='red')


def generate_samples(L, N, d):
    print 'Generating samples'
    sx = np.random.uniform(0, L, N)
    samples = sx
    if d>=2:
        sy = np.random.uniform(0, L, N)
        samples = zip(sx, sy)
    if d==3:
        sz = np.random.uniform(0, L, N)
        samples = zip(sx, sy, sz)
    return samples


def powspec(k):
    #print 'Computing power spectrum'
    if type(k)==float or type(k)==int and k==0:
        return 0.0
    a = 50
    b = 50
    c = -1
    d = 3
    N = 5 * 10 ** 4
    Pk = N * 1.0 / ((a * k) ** c + (b * k) ** d)
    if type(Pk)==float:
        assert Pk>=0
    else:
        assert Pk[Pk >= 0.0].size == Pk.size
    return Pk


def powspec_bump(k):
    #print 'Computing power spectrum'
    if type(k)==float or type(k)==int and k==0:
        return 0.0
    a = 50
    b = 50
    c = -1
    d = 3
    N = 5 * 10 ** 4
    Pk = N * 1.0 / ((a * k) ** c + (b * k) ** d)

    Pk += gaussian(k, *[0.1, 6, 1.5])

    if type(Pk)==float:
        assert Pk>=0
    else:
        assert Pk[Pk >= 0.0].size == Pk.size
    return Pk

def powspec_gaussian(k):
    #print 'Computing power spectrum'
    if type(k)==float or type(k)==int and k==0:
        return 0.0
    Pk = gaussian(k, *[100, 0, 15])

    if type(Pk)==float:
        assert Pk>=0
    else:
        assert Pk[Pk >= 0.0].size == Pk.size
    return Pk


def plot_powspec(k, Pk):
    plt.figure()
    plt.loglog(k, Pk)


def gaussian(x, *p):
    A, mu, sigma = p
    return np.array(A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)))


def plot_pixel_hist(dens, fit=False):
    dpix = np.real(dens).flatten()

    hist, bin_edges = np.histogram(dpix, bins=50)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure()
    plt.hist(dpix,bins=bin_centres)
    plt.xlabel('density amplitude')
    plt.ylabel('freq')

    if fit:
        p0 = [len(dpix)/10., 0., 0.5e-10]

        coeff, var_matrix = curve_fit(gaussian, bin_centres, hist, p0=p0)
        hist_fit = gaussian(bin_centres, *coeff)

        plt.plot(bin_centres, hist_fit, label='Fit: mean={:.3e}, stdev={:.3e}, var={:.3e}, '.format(
            coeff[1], coeff[2], coeff[2]**2))
        plt.legend()


if __name__=='__main__':
    main()