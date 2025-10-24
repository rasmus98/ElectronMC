from electron_scattering import Environment, propagate_photons
import numba
numba.set_num_threads(8)
import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy import optimize
import tqdm

central_wavelength = 6563 * u.AA
freq_fact = (central_wavelength.to(u.J, equivalencies=u.spectral())/c.c**2/c.m_e).si

numruns = 100
taus = np.geomspace(0.003,20.0,numruns)
eps = 1e-2
dirname = "scattering_data_10K/"
fbase = "scattering_counts_tau_"
fend = ".dat"


for i,tau in tqdm.tqdm(enumerate(taus)):
    env_obj = Environment(freq_fact=freq_fact, tau=tau, vel=100)
    npkts = int(5e6/np.sqrt(tau))
    energies, interactions = propagate_photons(n_pkts=npkts, env_obj=env_obj)
    wavelengths = (energies * c.m_e * c.c**2).to(u.AA, equivalencies=u.spectral())
    wl_edges = np.linspace(np.min(wavelengths), np.max(wavelengths), 300)
    counts, _ = np.histogram(wavelengths, bins=wl_edges)
    x = wl_edges[:-1]+0.5*np.diff(wl_edges)
    fname = dirname + fbase + "{:.3f}".format(tau) + fend
    np.savetxt(fname,np.vstack([x.value,counts]))
