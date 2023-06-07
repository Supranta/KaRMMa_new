import numpy as np
import pyccl
import matplotlib.pyplot as plt

def get_cl(Omega_c, sigma_8, nz_data, N_Z_BINS, gen_lmax):
    cosmo = pyccl.Cosmology(
        Omega_c=Omega_c,
        Omega_b=0.046,
        h=0.7,
        sigma8=sigma_8,
        n_s=0.97
    )
    tracers = []
    for i in range(N_Z_BINS):
        tracer_i = pyccl.tracers.WeakLensingTracer(cosmo, nz_data[i].T)
        tracers.append(tracer_i)

    ell = np.arange(gen_lmax + 1)
    cl = np.zeros((N_Z_BINS, N_Z_BINS, len(ell)))
    for i in range(N_Z_BINS):
        for j in range(i+1):
            cl_ij = pyccl.cls.angular_cl(cosmo, tracers[i], tracers[j], ell)
            cl[i,j] = cl_ij
            cl[j,i] = cl_ij

    cl[:,:,0] = 1e-21 * np.eye(N_Z_BINS)
    cl[:,:,1] = 1e-21 * np.eye(N_Z_BINS)
    
    return cl

def plot_nz(nz):
    nbins = nz.shape[0]

    plt.xlabel('$z$')
    plt.ylabel('$n(z)$')
    for i in range(nbins):
        plt.xlim(0., 2.)
        plt.plot(nz[i,:,0], nz[i,:,1], label='Bin %d'%(i+1))
    plt.legend()
    plt.show()    