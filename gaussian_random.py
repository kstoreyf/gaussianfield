from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import fft as fftengn
from scipy.optimize import curve_fit


pickle_dir = 'pickles_2018-10-16/'
plot_dir = 'plots_2018-10-16/'


def main():

    L = 40 #length scale
    n = 256 #pixels on side of density field
    d = 3 #dimensions

    generate_field(L, n, d)
    plt.show()


def generate_field(L, n, d):
    tag = '_n{}_L{}_{}d'.format(n, L, d)
    k = get_ks(L, n, d)
    Pk = powspec(k)
    a = get_amplitudes(L, n, Pk, d)

    dens = density_field(a, d)
    plot_pixel_hist(dens, fit=True)
    plot_density(dens, d, saveto='density'+tag+'.png')
    save_field(dens, 'density'+tag+'.p')


def sample_field(L, n, N, d):
    tag = '_n{}_L{}_{}d'.format(n, L, d)
    dens = load_field('density'+tag+'.p')
    N = int(1e4) #num samples
    q = exponentiate(dens, N, fac=5e4)
    samples = generate_samples(L, N, d)
    sampled_field, rejects = reject_samples(samples, q, L, n, d)

    plot_samples(sampled_field, rejects, d)
    save_field(sampled_field, 'samples'+tag+'.p')


def get_ks(L, n, d):
    print 'Computing ks'
    d = int(d)
    assert d in [1,2,3], 'd must be 1, 2, or 3'
    if d==1:
        return get_ks1d(L, n)
    elif d==2:
        return get_ks2d(L, n)
    else:
        return get_ks3d(L, n)


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


def get_ks3d(L, n):
    assert n % 2 == 0
    kx = fftengn.fftshift(fftengn.fftfreq(n, 1./n))
    ky = fftengn.fftshift(fftengn.fftfreq(n, 1./n))
    kz = fftengn.fftshift(fftengn.fftfreq(n, 1./n))[:n/2+1]
    k = 2.*np.pi/L * np.sqrt(kx[:,np.newaxis][np.newaxis,:]**2
                           + ky[:,np.newaxis][:,np.newaxis]**2
                           + kz[np.newaxis,:][np.newaxis,:]**2)
    return k


def get_amplitudes3d(L, n, Pk):
    deltan = (float(L)/float(n))**-3
    areal = np.zeros((n,n,n))
    aim = np.zeros((n,n,n))
    for i in range(n):
        for j in range(n):
            for g in range(n/2+1):
                pk = Pk[i][j][g]
                if (i==0 or i==n/2) and (j==0 or j==n/2) and (g==0 or g==n/2):
                    areal[i][j][g] = np.random.normal(0, deltan) * np.sqrt(pk) * 1./L**3
                    aim[i][j][g] = 0
                else:
                    areal[i][j][g] = 2**(-0.5) * np.random.normal(0, deltan) * np.sqrt(pk) * 1./L**3
                    aim[i][j][g] = 2**(-0.5) * np.random.normal(0, deltan) * np.sqrt(pk) * 1./L**3
                    areal[(n-i)%n][(n-j)%n][(n-g)%n] = areal[i][j][g]
                    aim[(n-i)%n][(n-j)%n][(n-g)%n] = -aim[i][j][g]
    a = areal + 1.0j*aim
    a = fftengn.ifftshift(a)
    return a


def get_ks2d(L, n):
    assert n % 2 == 0
    kx = fftengn.fftshift(fftengn.fftfreq(n, 1./n))
    ky = fftengn.fftshift(fftengn.fftfreq(n, 1./n))[:n/2+1]
    k = 2.*np.pi/L * np.sqrt(kx[:,np.newaxis]**2
                           + ky[np.newaxis,:]**2)
    return k


def get_amplitudes2d(L, n, Pk):
    deltan = (float(L)/float(n))**-2
    areal = np.zeros((n,n))
    aim = np.zeros((n,n))
    for i in range(n):
        for j in range(n/2+1):
            pk = Pk[i][j]
            if (i==0 or i==n/2) and (j==0 or j==n/2):
                areal[i][j] = np.random.normal(0, deltan) * np.sqrt(pk) * 1./L**2
                aim[i][j] = 0
            else:
                areal[i][j] = 2**(-0.5) * np.random.normal(0, deltan) * np.sqrt(pk) * 1./L**2
                aim[i][j] = 2**(-0.5) * np.random.normal(0, deltan) * np.sqrt(pk) * 1./L**2
                areal[(n-i)%n][(n-j)%n] = areal[i][j]
                aim[(n-i)%n][(n-j)%n] = -aim[i][j]
    a = areal + 1.0j*aim
    a = fftengn.ifftshift(a)
    return a


def get_ks1d(L, n):
    assert n % 2 == 0
    kx = fftengn.fftshift(fftengn.fftfreq(n, 1./n))[:n/2+1]
    k = 2.*np.pi/L * abs(kx)
    return k


def get_amplitudes1d(L, n, Pk):
    deltan = (float(L)/float(n))**-1
    areal = np.zeros(n)
    aim = np.zeros(n)
    for i in range(n/2+1):
        pk = Pk[i]
        if (i==0 or i==n/2):
            areal[i] = np.random.normal(0, deltan) * np.sqrt(pk) * 1./L
            aim[i] = 0
        else:
            areal[i] = 2**(-0.5) * np.random.normal(0, deltan) * np.sqrt(pk) * 1./L
            aim[i] = 2**(-0.5) * np.random.normal(0, deltan) * np.sqrt(pk) * 1./L
            areal[(n-i)%n] = areal[i]
            aim[(n-i)%n] = -aim[i]
    a = areal + 1.0j*aim
    a = fftengn.ifftshift(a)
    return a


def density_field(a, d):
    print 'Transforming amplitudes to density field'
    dens = 1./(2*np.pi)**d * fftengn.ifftn(a)
    assert abs(np.max(np.imag(dens)))<1e-15, 'Density field should be entirely real'
    return np.real(dens)


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


def plot_density(dens, d, saveto=None):
    plt.figure()
    if d==1:
        plt.plot(dens)
    if d>1:
        if d==3:
            dens_pix = dens[0]
        if d==2:
            dens_pix = dens
        ax = plt.gca()
        cax = ax.imshow(dens_pix)
        plt.colorbar(cax)
    if saveto:
        plt.savefig(plot_dir+saveto)


def reject_samples(samples, q, L, n, d):
    print "Rejecting samples"
    sampled_field = []
    rejects = []
    count = 0
    for s in samples:
        count +=1
        px = int(np.floor(s[0]/n))
        if d>=2:
            py = int(np.floor(s[1]/n))
        if d==3:
            pz = int(np.floor(s[2]/n))

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


def gaussian(x, mean, std):
    return 1/np.sqrt(2*np.pi*std**2) * np.exp(-1*(x-mean)**2/(2*std**2))


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


def plot_powspec(k, Pk):
    plt.figure()
    plt.loglog(k, Pk)


def plot_pixel_hist(dens, fit=False):
    dpix = np.real(dens).flatten()

    hist, bin_edges = np.histogram(dpix, bins=50)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure()
    plt.hist(dpix,bins=bin_centres)
    plt.xlabel('density amplitude')
    plt.ylabel('freq')

    if fit:

        def gauss(x, *p):
            A, mu, sigma = p
            return np.array(A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)))

        p0 = [len(dpix)/10., 0., 1e-10]

        coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
        hist_fit = gauss(bin_centres, *coeff)

        plt.plot(bin_centres, hist_fit, label='Fit: mean={:.3e}, stdev={:.3e}, var={:.3e}, '.format(
            coeff[1], coeff[2], coeff[2]**2))
        plt.legend()


if __name__=='__main__':
    main()