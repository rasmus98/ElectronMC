import numpy as np
from numpy.random import normal as randn
from collections import namedtuple

import numba

class Environment:
    def __init__(self, 
                 R_inner=1.0, 
                 R_outer=1.1, 
                 kappa=30.0, 
                 vel=30.0, 
                 freq_fact=5.e-6,
                 temp=1.e4 * 1.38e-23 / 9.11e-31 / 3e8 / 3e8):
        
        # Initialize default environment parameters
        self.R_inner = R_inner
        self.R_outer = R_outer
        self.kappa = kappa
        self.vel = vel  # velocity in km/s
        self.freq_fact = freq_fact  # Frequency factor for photon packets
        
        # Temperature and derived standard deviation for beta (velocity/c)
        self.temp = temp
        self.beta_stdev = np.sqrt(self.temp)

def class_instance_to_namedtuple(instance):
    # Get the class name to use as the typename
    typename = type(instance).__name__
    
    # Get the dictionary of instance attributes
    attr_dict = vars(instance)
    
    # Extract field names and values
    field_names = list(attr_dict.keys())
    attr_values = list(attr_dict.values())
    
    # Define the namedtuple with the field names
    ClassTuple = namedtuple(typename, field_names)
    
    # Create the namedtuple instance with the attribute values
    return ClassTuple(*attr_values)


@numba.njit(fastmath=True)
def initialize_packet(env_obj):
    """Initialize a photon packet with default values."""
    r_current = env_obj.R_inner  # Starting at inner boundary
    mu_current = 0.99999999  # Incoming photon direction.
    nu_current = randn(loc=1.0, scale=env_obj.vel / 299792.458) * env_obj.freq_fact
    return r_current, mu_current, nu_current

@numba.njit(fastmath=True)
def move_packet(env_obj, r_current, mu_current, nu_current):
    """Move the packet until it exits the scattering region or is absorbed."""
    inside = True
    escape = False
    interactions = 0

    while inside:
        # Draw a new optical depth and convert to physical distance
        tau_next = -1.0 * np.log(np.random.rand())
        s_next = tau_next / env_obj.kappa

        # Determine if heading towards inner boundary
        if mu_current < -np.sqrt(1.0 - (env_obj.R_inner / r_current)**2):
            # Heading towards inner boundary
            s_max = calculate_s_max_inner(env_obj, r_current, mu_current)
            if s_next > s_max:
                # Photon would hit inner boundary
                r_current, mu_current = handle_inner_boundary(env_obj, r_current, mu_current, s_max)
                continue  # Restart loop with updated position and direction

        # Check if photon will reach outer boundary before next interaction
        s_max = calculate_s_max_outer(env_obj, r_current, mu_current)
        if s_next > s_max:
            # Photon escapes
            inside = False
            escape = True
            break

        # Photon scatters before reaching any boundary
        r_current, mu_current, nu_current, interactions = scatter_photon(
            env_obj, r_current, mu_current, nu_current, s_next, interactions
        )

    return escape, nu_current, interactions

@numba.njit(fastmath=True)
def calculate_s_max_inner(env_obj, r_current, mu_current):
    """Calculate the maximum distance to the inner boundary."""
    discriminant = r_current**2 * (mu_current**2 - 1.0) + env_obj.R_inner**2
    s_max = -r_current * mu_current - np.sqrt(discriminant)
    return s_max

@numba.njit(fastmath=True)
def calculate_s_max_outer(env_obj, r_current, mu_current):
    """Calculate the maximum distance to the outer boundary."""
    discriminant = r_current**2 * (mu_current**2 - 1.0) + env_obj.R_outer**2
    s_max = -r_current * mu_current + np.sqrt(discriminant)
    return s_max
 
@numba.njit(fastmath=True)
def handle_inner_boundary(env_obj, r_current, mu_current, s_max):
    """Handle the interaction with the inner boundary."""
    # Move to just before the boundary to avoid numerical issues
    s_next = s_max * 0.999999
    # Update position and direction
    r_new = np.sqrt(r_current**2 + s_next**2 + 2.0 * s_next * r_current * mu_current)
    mu_new = -0.5 * (r_current**2 - r_new**2 - s_next**2) / (r_new * s_next)
    mu_new *= -1  # Reflect at the inner boundary
    return r_new, mu_new

@numba.njit(fastmath=True)
def scatter_photon(env_obj, r_current, mu_current, nu_current, s_next, interactions):
    """Scatter the photon within the scattering region."""
    # Move the packet
    r_new = np.sqrt(r_current**2 + s_next**2 + 2.0 * s_next * r_current * mu_current)
    mu_new = -0.5 * (r_current**2 - r_new**2 - s_next**2) / (r_new * s_next)
    if np.abs(mu_new) > 1.0:
        mu_new = np.sign(mu_new)
    if np.abs(mu_new) > 1.01:
        print("Error: mu_new out of bounds", mu_new)

    # Frequency and scattering calculations
    nu_current, mu_new = calculate_scattering(env_obj, nu_current, mu_new)

    interactions += 1
    return r_new, mu_new, nu_current, interactions

@numba.njit(fastmath=True)
def calculate_scattering(env_obj, nu_current, mu_current):
    """Calculate the scattering of the photon."""
    # Electron speed calculation (in units of c)

    # Photon direction in observer frame
    n_phot_x_in = 0.0
    n_phot_y_in = np.sqrt(1.0 - mu_current**2) * (1 if np.random.random() < 0.5 else -1)
    n_phot_z_in = mu_current

    # Rejection sampling based on increased flux
    v_ele_x, v_ele_y, v_ele_z = rejection_sampling(
        n_phot_x_in, n_phot_y_in, n_phot_z_in, env_obj.beta_stdev
    )

    # Transform to electron rest frame
    nu_new, n_phot_x_in, n_phot_y_in, n_phot_z_in, gamma = transform_to_electron_frame(
        nu_current, n_phot_x_in, n_phot_y_in, n_phot_z_in, v_ele_x, v_ele_y, v_ele_z
    )

    # Scatter photon in electron rest frame
    nu_new, n_phot_x, n_phot_y, n_phot_z = compton_scatter(
        nu_new, n_phot_x_in, n_phot_y_in, n_phot_z_in
    )

    # Transform back to observer frame
    nu_current, mu_new = transform_to_observer_frame(
        nu_new, n_phot_x, n_phot_y, n_phot_z, v_ele_x, v_ele_y, v_ele_z, gamma
    )

    return nu_current, mu_new

@numba.njit(fastmath=True)
def draw_electron_velocity(beta_stdev):
    """Draw electron velocities from a normal distribution."""
    v_ele_x = np.random.normal(0.0, beta_stdev)
    v_ele_y = np.random.normal(0.0, beta_stdev)
    v_ele_z = np.random.normal(0.0, beta_stdev)
    return v_ele_x, v_ele_y, v_ele_z

@numba.njit(fastmath=True)
def rejection_sampling(n_phot_x_in, n_phot_y_in, n_phot_z_in, beta_stdev):
    """Perform rejection sampling to account for increased flux."""
    while True:
        v_ele_x, v_ele_y, v_ele_z = draw_electron_velocity(beta_stdev)
        ndotv = n_phot_x_in * v_ele_x + n_phot_y_in * v_ele_y + n_phot_z_in * v_ele_z
        p_accept = 0.5 * (1 - ndotv)
        if np.random.rand() < p_accept:
            break
    return v_ele_x, v_ele_y, v_ele_z

@numba.njit(fastmath=True)
def transform_to_electron_frame(nu_current, n_phot_x_in, n_phot_y_in, n_phot_z_in, v_ele_x, v_ele_y, v_ele_z):
    """Transform photon to electron rest frame."""
    speed = np.sqrt(v_ele_x**2 + v_ele_y**2 + v_ele_z**2)
    gamma = 1.0 / np.sqrt(1.0 - speed**2)
    ndotv = n_phot_x_in * v_ele_x + n_phot_y_in * v_ele_y + n_phot_z_in * v_ele_z

    # Transform direction cosines
    factor = gamma * (1 - ndotv)
    n_phot_x_in = (n_phot_x_in - v_ele_x * (gamma - 1) * ndotv / speed**2) / factor
    n_phot_y_in = (n_phot_y_in - v_ele_y * (gamma - 1) * ndotv / speed**2) / factor
    n_phot_z_in = (n_phot_z_in - v_ele_z * (gamma - 1) * ndotv / speed**2) / factor

    # Transform frequency
    nu_new = nu_current * gamma * (1 - ndotv)

    return nu_new, n_phot_x_in, n_phot_y_in, n_phot_z_in, gamma

@numba.njit(fastmath=True)
def compton_scatter(nu_new, n_phot_x_in, n_phot_y_in, n_phot_z_in):
    """Perform Compton scattering in the electron rest frame."""
    while True:
        # Draw random scattering direction
        mu_new, phi = draw_random_direction()
        n_phot_x = np.sqrt(1 - mu_new**2) * np.cos(phi)
        n_phot_y = np.sqrt(1 - mu_new**2) * np.sin(phi)
        n_phot_z = mu_new

        # Calculate scattering angle cosine
        cos_theta_scattering = n_phot_x * n_phot_x_in + n_phot_y * n_phot_y_in + n_phot_z * n_phot_z_in
        
        # Calculate Compton recoil factor
        f_comp = 1.0 / (1 + nu_new * (1 - cos_theta_scattering))
        # Differential cross-section (Klein-Nishina formula)
        p_accept = 0.5 * f_comp**2 * (f_comp + 1.0 / f_comp - 1 + cos_theta_scattering**2)
        if np.random.rand() < p_accept:
            break  # Accept the scattering angle

    # Apply Compton shift to frequency
    nu_new = nu_new / (1 + nu_new * (1 - cos_theta_scattering))

    return nu_new, n_phot_x, n_phot_y, n_phot_z

@numba.njit(fastmath=True)
def draw_random_direction():
    """Draw random direction in electron rest frame."""
    mu_new = -1.0 + 2.0 * np.random.rand()
    phi = 2.0 * np.pi * np.random.rand()
    return mu_new, phi

@numba.njit(fastmath=True)
def transform_to_observer_frame(nu_new, n_phot_x, n_phot_y, n_phot_z, v_ele_x, v_ele_y, v_ele_z, gamma):
    """Transform photon back to observer frame after scattering."""
    ndotv = n_phot_x * v_ele_x + n_phot_y * v_ele_y + n_phot_z * v_ele_z
    ndotv *= -1.0  # Reverse sign for reverse transformation

    # Transform direction cosines back to observer frame
    factor = gamma * (1 - ndotv)
    n_phot_x = (n_phot_x + v_ele_x * (gamma - 1) * ndotv / v_ele_x**2) / factor
    n_phot_y = (n_phot_y + v_ele_y * (gamma - 1) * ndotv / v_ele_y**2) / factor
    n_phot_z = (n_phot_z + v_ele_z * (gamma - 1) * ndotv / v_ele_z**2) / factor

    # Transform frequency back to observer frame
    nu_current = nu_new * gamma * (1 - ndotv)
    mu_new = n_phot_z  # Update direction cosine

    return nu_current, mu_new

@numba.njit(parallel=True, fastmath=True)
def _propagate_photons(n_pkts, env_obj):
    """Main simulation loop to process all photon packets."""
    escaped = np.zeros(n_pkts, dtype=np.bool_)
    energies = np.zeros(n_pkts)
    interactions = np.zeros(n_pkts, dtype=np.int32)

    for i in numba.prange(n_pkts):
        # Initialize packet
        r_current, mu_current, nu_current = initialize_packet(env_obj)
        # Move packet through the medium
        escaped[i], energies[i], interactions[i] = move_packet(
            env_obj, r_current, mu_current, nu_current
        )

    return energies[escaped], interactions[escaped]

def propagate_photons(n_pkts, env_obj):
    """Main simulation loop to process all photon packets."""
    return _propagate_photons(n_pkts, class_instance_to_namedtuple(env_obj))

if __name__ == "__main__":
    import time
    env_obj = Environment()
    # warm up
    escaped, energies, interactions = propagate_photons(n_pkts=int(1e3), env_obj=env_obj)
    print("Starting test")
    for n_threads in [1, 2, 4, 8, 16, 32, 64]:
        numba.set_num_threads(n_threads)
        start = time.time()
        escaped, energies, interactions = propagate_photons(n_pkts=int(1e6), env_obj=env_obj)
        print("Took:", time.time() - start, "with", n_threads, "threads")
