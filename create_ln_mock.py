import sys
import numpy as np
import h5py as h5
from karmma import KarmmaSampler, KarmmaConfig
from karmma.utils import *
import karmma.transforms as trf
from scipy.stats import norm, poisson
import torch

torch.set_num_threads(8)

configfile     = sys.argv[1]
config         = KarmmaConfig(configfile)

try:
    lowpass_filter = bool(sys.argv[2])
    if(lowpass_filter):
        print("CREATING LOW-PASS FILTERED MAPS!")
except:
    lowpass_filter = False

nside    = config.analysis['nside']
gen_lmax = 3 * nside - 1
lmax     = 2 * nside - 1

N_Z_BINS = config.analysis['nbins']
shift    = config.analysis['shift']

nz_data  = config.analysis['nz']
sigma_e  = config.analysis['sigma_e']

cl = config.analysis['cl'][:,:,:(gen_lmax + 1)]
cl_emu = None

#============================================================
print("Initializing sampler....")
tmp = np.zeros((N_Z_BINS,hp.nside2npix(nside)))
tmp = KarmmaSampler(tmp, tmp, tmp, tmp, cl, shift, cl_emu, lmax, gen_lmax)
print("Done initializing sampler....")

ell, emm = hp.Alm.getlm(gen_lmax)

def eigvec_matmul(A, x):
    y = np.zeros_like(x)
    for i in range(N_Z_BINS):
        for j in range(N_Z_BINS):
            y[i] += A[i,j] * x[j]
    return y

def apply_cl(xlm, cl):
    L = np.linalg.cholesky(cl.T).T
    
    xlm_real = xlm.real
    xlm_imag = xlm.imag
    
    L_arr = np.swapaxes(L[:,:,ell[ell > -1]], 0,1)
    
    ylm_real = eigvec_matmul(L_arr, xlm_real) / np.sqrt(2.)
    ylm_imag = eigvec_matmul(L_arr, xlm_imag) / np.sqrt(2.)

    ylm_real[:,ell[emm==0]] *= np.sqrt(2)
    
    return ylm_real + 1j * ylm_imag

def get_xlm(xlm_real, xlm_imag):
    ell, emm = hp.Alm.getlm(gen_lmax)
    #==============================
    _xlm_real = np.zeros((N_Z_BINS, len(ell)))
    _xlm_imag = np.zeros_like(_xlm_real)
    _xlm_real[:,ell > 1] = xlm_real
    _xlm_imag[:,(ell > 1) & (emm > 0)] = xlm_imag
    xlm = _xlm_real + 1j * _xlm_imag
    #==============================
    return xlm
    
def generate_xlm():
    xlm_real = np.random.normal(size=(N_Z_BINS, (ell > 1).sum()))
    xlm_imag = np.random.normal(size=(N_Z_BINS, ((ell > 1) & (emm > 0)).sum()))

    xlm = get_xlm(xlm_real, xlm_imag)
    return xlm

def generate_mock_y_lm():
    xlm = generate_xlm()
    return apply_cl(xlm, tmp.y_cl)

mask    = hp.fitsfunc.read_map(config.maskfile)
mask_lo = hp.ud_grade(mask, nside)
boolean_mask = mask_lo.astype(bool)

def get_y_maps():
    y_lm = generate_mock_y_lm()
    y_maps = []
    for i in range(N_Z_BINS):
        y_map = hp.alm2map(np.ascontiguousarray(y_lm[i]), nside, lmax=gen_lmax, pol=False)
        y_maps.append(y_map)    
    return np.array(y_maps)    

def get_mock_data(y_maps):
    g1_list = []
    g2_list = []
    k_list = []
    for i in range(N_Z_BINS):
        k = np.exp(y_maps[i] + tmp.mu[i]) - shift[i]
        if(lowpass_filter):
            lowpass_ell_filter = (ell < lmax)
            k = get_filtered_map(k, lowpass_ell_filter, nside)
        k_list.append(k)
        g1, g2 = trf.conv2shear(torch.tensor(k), lmax)
        g1 = g1.numpy() * mask_lo
        g2 = g2.numpy() * mask_lo
        g1_list.append(g1)
        g2_list.append(g2)    

    g1 = np.array(g1_list)
    g2 = np.array(g2_list)    

    N_bar = config.analysis['nbar'] * hp.nside2pixarea(nside, degrees=True) * 60**2

    N = []
    for i in range(N_Z_BINS):
        N_i = poisson(N_bar[i]).rvs(hp.nside2npix(nside))
        N.append(N_i * mask_lo)
    N = np.array(N)

    sigma = config.analysis['sigma_e'] / np.sqrt(N + 1e-25)
    g1_obs = g1 + np.random.standard_normal(sigma.shape) * sigma
    g2_obs = g2 + np.random.standard_normal(sigma.shape) * sigma
    g1_obs = g1_obs * mask_lo
    g2_obs = g2_obs * mask_lo    
    k_arr  = np.array(k_list)

    return g1_obs, g2_obs, k_arr, N 

def check_file_exists(filename):
    import os.path
    if os.path.isfile(filename):
        print('DATAFILE ALREADY EXISTS!')
        return True
    return False
    
def save_datafile(g1_obs, g2_obs, k_arr, N):
    file_exists = check_file_exists(config.datafile)
    overwrite = False
    if(file_exists):      
        overwrite_response = input("WE WILL NEED TO OVERWRITE THE EXISTING DATAFILE. ARE YOU SURE YOU WANT TO PROCEED? (y/n)")
        overwrite_response = overwrite_response.lower()
        assert overwrite_response in ['y', 'n'], "Invalid response"
        overwrite = (overwrite_response == 'y')
        if not overwrite:
            print("Not overwriting the existing file")
            return
    if not file_exists or overwrite:
        if(file_exists):
            print("OVERWRITING FILE!")
        with h5.File(config.datafile, 'w') as f:
            f['g1_obs'] = g1_obs
            f['g2_obs'] = g2_obs
            f['kappa']  = k_arr
            f['N']      = N    
            f['mask']   = mask_lo    
        
y_maps = get_y_maps()
g1_obs, g2_obs, k_arr, N = get_mock_data(y_maps)
save_datafile(g1_obs, g2_obs, k_arr, N)
