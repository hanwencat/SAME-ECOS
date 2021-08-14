import numpy as np
import scipy.io as sio
import math
import itertools
import matplotlib.pyplot as plt

## Determine for boundary conditions
def T2_boundary(SNR=100, echo_3=30, echo_last=320):
    """
    Generate T2 boundaries for SAME-ECOS analysis.

    Args:
        SNR (float, optional): signal to noise ratio. Defaults to 100.
        echo_3 (float, optional): 3rd echo time (ms). Defaults to 30.
        echo_last (float, optional): last echo time (ms). Defaults to 320.

    Returns:
        T2_min, T2_max: T2 lower and upper boundaries (ms)
    """    
    
    T2_min = np.ceil(-echo_3 / np.log(1/SNR))
    T2_max = np.floor(-echo_last / np.log(1/SNR))

    return T2_min, T2_max


def T2_lower_boundary(SNR=100, num_T2=1, echo_1=10):
    """
    Determine T2_min if there are multiple T2 component.

    Args:
        SNR (float, optional): signal to noise. Defaults to 100.
        num_T2 (int, optional): number of T2 components. Defaults to 1.
        echo_1 (float, optional): first echo time (ms). Defaults to 10.

    Returns:
        T2_min: T2 lower boundary (ms)
    """    
    
    T2_min = np.rint(-echo_1*(2*num_T2+1) / np.log(1/SNR))
    return T2_min


def T2_components_resolution_finite_domain(SNR=100, T2_min=7, T2_max=1000):
    """
    Numerically determine the maximum number of T2 components that can be resolved for a given SNR at a certain T2 range.

    Args:
        SNR (float, optional): signal to noise ratio. Defaults to 100.
        T2_min (float, optional): T2 lower boundary (ms). Defaults to 7.
        T2_max (float, optional): T2 upper boundary (ms). Defaults to 1000.

    Returns:
        M: the maximum number of T2 components that can be resolved.
    """    
   
    M = 1
    f = -1
    while f < 0:
        M = M + 0.01
        f = M/np.log(T2_max/T2_min) * np.sinh(np.pi**2*M/np.log(T2_max/T2_min)) - (SNR/M)**2
    return M


def resloution_limit(T2_min=7, T2_max=2000, M=4):
    """
    Finite domain T2 resolution calculation.

    Args:
        T2_min (float, optional): T2 lower boundary (ms). Defaults to 7.
        T2_max (float, optional): T2 upper boundary (ms). Defaults to 2000.
        M (float, optional): maximum number of resolvable T2 components. Defaults to 4.

    Returns:
        resolution: the T2 resolution.
    """    
    
    resolution = (T2_max/T2_min)**(1/M)
    return resolution


def t2_basis_generator(T2_min=7, T2_max=1000, num_basis_T2=40):
    """
    Generate T2 basis set.

    Args:
        T2_min (float, optional): T2 lower boundary (ms). Defaults to 7.
        T2_max (float, optional): T2 upper boundary (ms). Defaults to 1000.
        num_basis_T2 (int, optional): number of basis t2s. Defaults to 40.

    Returns:
        t2_basis: generated basis t2s (ms).
    """    
    
    t2_basis = np.geomspace(T2_min, T2_max, num_basis_T2)    
    return t2_basis


def analysis_boundary_condition(SNR=100, echo_1=10, T2_max=2000):
    """
    Analytically determines the boundary condition for the analysis according to experimental conditions.
    For n T2 components, the residual signal of the shortest T2 component on the (2n+1)th echo has to be larger than the noise.
    The boundary conditions yield to the resolution limit formula.
    Returns the maximum number of T2 components allowed and a list for T2_min. Use with Caution! 

    Args:
        SNR (float, optional): signal to noise ratio. Defaults to 100.
        echo_1 (float, optional): 1st echo time (ms). Defaults to 10.
        T2_max (float, optional): T2 upper limit (ms). Defaults to 2000.

    Returns:
        num_T2-1, T2_min_list: maximum resolvable T2 components, a list of T2 lower boundaries
    """     

    T2_min_list = []
    num_T2 = 1
    while True:
        T2_min = -echo_1*(2*num_T2+1) / np.log(1/SNR)
        M = T2_components_resolution_finite_domain(SNR, T2_min, T2_max)    
        if M <= num_T2:
            break
        else:
            num_T2 = num_T2+1
            T2_min_list.append(T2_min)
    return num_T2-1, np.ceil(T2_min_list)



## Random generation for T2 peaks
def T2_location_generator_v3(T2_min=5, T2_max=2000, num_T2=3, M_max=5, scale='log'):
    """
    Randomly generate T2 locations.
    Generated T2 peaks should have uniform distribution (scale = linear or log) at the range of [T2_min, T2_max]

    Args:
        T2_min (float, optional): T2 lower boundary (ms). Defaults to 5. 
        T2_max (float, optional): T2 upper boundary (ms). Defaults to 2000.
        num_T2 (int, optional): number of T2 component. Defaults to 3.
        M_max (int, optional): global maximal number of resolvable T2 components. Defaults to 5.
        scale (str, optional): logarithmic or linear scale over the T2 range. Defaults to 'log'.

    Returns:
        T2_locations: An array of T2 locations (ms)
    """    
    
    T2_location = np.zeros(num_T2)
    resolution = resloution_limit(T2_min, T2_max, M_max)    
    T2_low = T2_min 
    for i in range(num_T2):
        ### uniform distribution on linear scale
        if scale == 'linear': 
            T2_location[i] = np.random.randint(T2_low, T2_max/(resolution**(num_T2-i-1))) 
        ### uniform distribution on logarithmic scale (https://stackoverflow.com/questions/43977717/how-do-i-generate-log-uniform-distribution-in-python/43977980)
        if scale == 'log':
            T2_location[i] = np.exp(np.random.uniform(np.log(T2_low),np.log(T2_max/(resolution**(num_T2-i-1)))))
        T2_low = T2_location[i]*resolution # set the new lower boundary        
    T2_location = T2_location.astype(int)     
    return T2_location
        
    
def T2_resolution_examiner(T2_location, resolution):
    """
    Examine whether two T2 peak locations obey the resolution limit.

    Args:
        T2_location (array): an array of T2 locations (ms)
        resolution (float): the resolution limit

    Returns:
        TF: True or False
    """    
    
    TF = True
    for x, y in itertools.combinations(T2_location,2):
        if x/y > 1/resolution and x/y < resolution:
            TF = False
            break        
    return TF


def required_frequency(T2_location):
    """
    Calculate the required frequency w to construct the generated T2 peaks in the T2 domain.

    Args:
        T2_location (array): an array of T2 locations (ms)

    Returns:
        ratio_min, frequency: the minimal ratio between adjacent components, required frequency.
    """    
    
    T2_location_sort = np.sort(T2_location)
    ratio_min = (T2_location_sort[1:]/T2_location_sort[:-1]).min()
    frequency = np.pi/np.log(ratio_min)   
    return ratio_min, frequency


def minimum_amplitude_calculator(SNR, frequency):
    """
    Calculate the minimum allowable amplitude for all T2 components at a given SNR.

    Args:
        SNR (float): signal to noise ratio
        frequency (float): required frequency of the T2 components

    Returns:
        minimum_amp_T2_peak: the minimal amplitude of T2 components
    """    
    
    amp_noise = 1/SNR
    minimum_amp_T2_peak = amp_noise * np.sqrt(frequency/np.pi * np.sinh(np.pi*frequency))    
    return minimum_amp_T2_peak


def T2_amplitude_generator_v3(num_T2, minimum_amplitude):
    """
    Randomly generate normalized T2 peak amplitude.
    Generated amplitude should have uniform distribution at range [minimum_amplitude, 1-(num_T2-1)*minimum_amplitude]

    Args:
        num_T2 (int): number of T2 components
        minimum_amplitude (float): the minimal amplitude

    Returns:
        T2_amplitude: an array of T2 amplitudes
    """    
    
    T2_amplitude = np.zeros(num_T2)
    remainder = 1
    for i in range(num_T2-1):
        T2_amplitude[i] = np.random.uniform(minimum_amplitude, remainder-(num_T2-i-1)*minimum_amplitude, 1)
        remainder = 1 - T2_amplitude.sum()
    T2_amplitude[-1] = remainder
    T2_amplitude = T2_amplitude/T2_amplitude.sum()    
    T2_amplitude = np.squeeze(T2_amplitude) 
    if T2_amplitude.shape[0] == 1:
        T2_amplitude = T2_amplitude.reshape(1,) 
    ### amplitude array has a descending trend so it needs to be shuffled 
    np.random.shuffle(T2_amplitude) 
    return T2_amplitude  


def metrics_extraction_v3(T2_location, T2_amplitude, MW_low_cutoff=0, MW_high_cutoff=40, IEW_low_cutoff=40, IEW_high_cutoff=200):
    """
    This function extracts five metrics: myelin water fraction (MWF), MW geometric mean T2 (MWGMT2), IEWF, IEWGMT2, GMT2.

    Args:
        T2_location (array): an array of T2 locations (ms)
        T2_amplitude (array): an array of T2 amplitudes
        MW_low_cutoff (float, optional): myelin water lower boundary (ms). Defaults to 0.
        MW_high_cutoff (float, optional): myelin water upper boundary (ms). Defaults to 40.
        IEW_low_cutoff (float, optional): IE water lower boundary (ms). Defaults to 40.
        IEW_high_cutoff (float, optional): IE water upper boundary (ms). Defaults to 200.

    Returns:
        MWF, MWGMT2, IEWF, IEWGMT2, GMT2: myelin water fraction (MWF), MW geometric mean T2 (MWGMT2), IEWF, IEWGMT2, GMT2
    """    
      
    ### get the location index of MW and IEW
    MW_loc = (T2_location>=MW_low_cutoff) & (T2_location<=MW_high_cutoff)
    IEW_loc = (T2_location>=IEW_low_cutoff) & (T2_location<=IEW_high_cutoff)    
    ### calculate MWF and IEWF
    MWF = T2_amplitude[MW_loc].sum()/T2_amplitude.sum()
    IEWF = T2_amplitude[IEW_loc].sum()/T2_amplitude.sum()    
    ### calculate weighted geometric mean using equation in https://en.wikipedia.org/wiki/Weighted_geometric_mean
    if MWF == 0:
        MWGMT2 = np.nan
    else:      
        MWGMT2 = np.exp(np.dot(T2_amplitude[MW_loc], np.log(T2_location[MW_loc])) / T2_amplitude[MW_loc].sum())
    if IEWF == 0:
        IEWGMT2 = np.nan
    else:
        IEWGMT2 = np.exp(np.dot(T2_amplitude[IEW_loc], np.log(T2_location[IEW_loc])) / T2_amplitude[IEW_loc].sum())
    ### calculate the overall weighted geometric mean
    GMT2 = np.exp(np.dot(T2_amplitude, np.log(T2_location)) / T2_amplitude.sum())
    return MWF, MWGMT2, IEWF, IEWGMT2, GMT2


## Produce decay data and spectral labels for model training
def load_decay_lib(file_path):
    """
    Load the calculated decay library from .mat file.

    Args:
        file_path (str): the file path to the .mat file

    Returns:
        decay_lib: the loaded decay library
    """    
    
    decay_lib = sio.loadmat(file_path)
    decay_lib = decay_lib['decay']
    return decay_lib


def produce_decay_from_lib(decay_lib, T2_location, T2_amplitude, FA):
    """
    Generate the decay curve generation from the decay library: weighted sum of the selected T2 components.

    Args:
        decay_lib (array): the loaded decay library
        T2_location (int): an array of T2 location (ms)
        T2_amplitude (int): an array of T2 amplitudes
        FA (int): selected refocusing flip angle (degree)

    Returns:
        decay_curve: the produced decay curve
    """   
    
    decay_curve = np.sum(decay_lib[T2_location-1,FA-1,:] * T2_amplitude.reshape(T2_amplitude.shape[0],1), axis=0)
    return decay_curve


def signal_with_noise_generation_phase_rotation(signal, SNR):
    """
    1. Project pure signal to real and imaginary axis according to a randomly generated phase factor.
    2. Generate noise (normal distribution on real and imaginary axis)
    3. Noise variance is determined by SNR (Rayleigh noise floor).

    Args:
        signal (array): the decay signal
        SNR (float): signal to noise ratio

    Returns:
        signal_with_noise: the signal with added noise on both real and imaginary axis
    """    
    
    phase = math.pi/2 * np.random.rand()
    signal_real = signal * math.cos(phase)
    signal_imaginary = signal * math.sin(phase)   
    Rayleigh_noise_variance = 1/(SNR * math.sqrt(math.pi/2))   
    noise_real = np.random.normal(0, Rayleigh_noise_variance, signal.shape[0])
    noise_imaginary = np.random.normal(0, Rayleigh_noise_variance, signal.shape[0])    
    signal_with_noise = ((signal_real + noise_real)**2 + (signal_imaginary + noise_imaginary)**2) ** 0.5   
    return signal_with_noise


def train_label_generator(T2_location, T2_amplitude, t2_basis):
    """
    This function takes randomly generated t2 peak locations and amplitudes as inputs, and uses basis t2s to represent these T2 peaks.
    Each peak is embedded by two nearest basis T2s.

    Args:
        T2_location (array): an array of T2 locations (ms)
        T2_amplitude (array): an array of T2 amplitudes
        t2_basis (array): the basis t2s (ms)

    Returns:
        train_label: the T2 spectrum depicted by the basis t2s
    """   
    
    ### create multi-dimensional placeholder (each dimension for each peak)
    train_label=np.zeros([T2_location.shape[0],t2_basis.shape[0]])  
    ### iterate through each peak and find the nearest couple of t2 basis and assign weighting factors
    for i in range(T2_location.shape[0]):
        for j in range(t2_basis.shape[0]):         
            if abs(t2_basis[j]-T2_location[i])<0.000000001:
                train_label[i,j] = T2_amplitude[i]            
            elif t2_basis[j-1]<T2_location[i] and t2_basis[j]>T2_location[i]:
                train_label[i,j-1] = (t2_basis[j]-T2_location[i])/(t2_basis[j]-t2_basis[j-1])*T2_amplitude[i]
                train_label[i,j] = (T2_location[i]-t2_basis[j-1])/(t2_basis[j]-t2_basis[j-1])*T2_amplitude[i]        
    ### return one dimensional train label
    return train_label.sum(axis=0)


def train_label_generator_gaussian_embedding(T2_location, T2_amplitude, T2_min, T2_max, t2_basis, sigma=1):
    """
    This function takes randomly generated t2 peak locations and amplitudes as inputs, and uses basis t2s to represent these T2 peaks.
    Each peak generates a gaussian function centered at its peak location (in log space), and then embedded by all basis T2s

    Args:
        T2_location (array): an array of T2 locations (ms)
        T2_amplitude (array): an array of T2 amplitude
        T2_min (float): the T2 lower boundary (ms)
        T2_max (float): the T2 upper boundary (ms)
        t2_basis (array): the basis t2s (ms)
        sigma (float, optional): variance of the Gaussian peaks. Defaults to 1.

    Returns:
        train_label: the T2 spectrum depicted by basis t2s
    """    
    
    ### create multi-dimensional placeholder (each dimension for each peak)
    train_label=np.zeros([T2_location.shape[0],t2_basis.shape[0]])
    ### iterate through each peak and assign weighting factors to t2_basis according to normal distribution
    for i in range(T2_location.shape[0]):
        train_label[i,:] = gaussian_embedding_log_scale(T2_location[i], T2_min=T2_min, T2_max=T2_max, t2_basis=t2_basis, sigma=sigma) * T2_amplitude[i]  
    ### return one dimensional train label
    return train_label.sum(axis=0)


def gaussian_embedding_log_scale(peak, T2_min, T2_max, t2_basis, sigma):
    """
    This function takes one t2 delta peak as inputs, and returns a normalized gaussian weighted t2_basis labels on log scale.

    Args:
        peak (float): one T2 location
        T2_min (float): T2 lower boundary (ms)
        T2_max (float): T2 upper boundary (ms)
        t2_basis (array): basis t2s (ms)
        sigma (float): variance of the Gaussian peak.

    Returns:
        t2_basis_weights_scaled: the normalized Gaussian peaks
    """    
    
    base = (T2_max/T2_min)**(1/t2_basis.shape[0])
    t2_basis_index = np.arange(t2_basis.shape[0])
    peak_index = np.log(peak/T2_min)/np.log(base)
    t2_basis_weights = 1/(sigma*np.sqrt(2*math.pi)) * np.exp(-(t2_basis_index - peak_index)**2/(2*sigma**2))
    t2_basis_weights[t2_basis_weights<1e-7] = 0
    t2_basis_weights_scaled = t2_basis_weights/t2_basis_weights.sum()  
    return t2_basis_weights_scaled


def produce_training_data(decay_lib,
                          realizations = 10000,
                          SNR = None,
                          SNR_boundary_low = 50,
                          SNR_boundary_high = 500,
                          echo_3 = 30,
                          echo_last = 320,
                          echo_train_num = 32,
                          num_t2_basis = 40,
                          FA_min = 50,
                          peak_width = 1,
                          T2_min_universal = None,
                          T2_max_universal = None,
                          exclude_M_max = True, 
                          N_weights = 0.2):
    """
    Produce training data via SAME-ECOS simulation pipeline (use a single cpu core).

    Args:
        decay_lib (array): the decay library
        realizations (int, optional): the number of simulation realizations. Defaults to 10000.
        SNR (float, optional): pick a fixed SNR instead of random generation. Defaults to None.
        SNR_boundary_low (float, optional): lower boundary of SNR. Defaults to 50.
        SNR_boundary_high (float, optional): upper boundary of SNR. Defaults to 500.
        echo_3 (float, optional): the 3rd echo time in ms. Defaults to 30.
        echo_last (float, optional): the last echo time in ms. Defaults to 320.
        echo_train_num (int, optional): the number of echoes in the echo train. Defaults to 32.
        num_t2_basis (int, optional): the number of basis t2s. Defaults to 40.
        FA_min (float, optional): the minimal refocusing flip angle (degree) for simulation. Defaults to 50.
        peak_width (float, optional): the variance of the gaussian peak. Defaults to 1.
        T2_min_universal (float, optional): the overall minimal T2 (ms) of the analysis. Defaults to calculate on the fly if None is given.
        T2_max_universal (float, optional): the overall maximal T2 (ms) of the analysis. Defaults to to 2000ms if None is given.
        exclude_M_max (bool, optional): exclude the M_max if True. Defaults to True.
        N_weights (float, optional): the weighting factor for N peaks (weight = N_choice ** N_weights).

    Returns:
        data: dictionary collection of the produced training data
    """    
    
    ### Define T2 range, maximum number (M_max) of T2 peaks at the highest SNR, allowable number (N) of T2 peaks for simulation 
    if T2_min_universal == None:
        T2_min_universal,_ = T2_boundary(SNR_boundary_high, echo_3, echo_last) ## Lower boundary is determined by the highest SNR
    if T2_max_universal == None:
        T2_max_universal = 2000 ## empirically determined
    t2_basis = t2_basis_generator(T2_min_universal, T2_max_universal, num_t2_basis)
    M_max = int(np.floor(T2_components_resolution_finite_domain(SNR_boundary_high, T2_min_universal, T2_max_universal))) ## M at highest SNR
    #resolution_max = resloution_limit(T2_min_universal, T2_max_universal, M_max) ## resolution at highest SNR
    if exclude_M_max == True:
        N = M_max - 1 ## for simulation M_max is excluded
    else:
        N = M_max
    ### Create placeholders for memory efficiency
    T2_location_all = np.zeros([realizations, N])
    T2_amplitude_all = np.zeros([realizations,N])
    decay_curve_all = np.zeros([realizations, echo_train_num])
    decay_curve_with_noise_all = np.zeros([realizations, echo_train_num])
    train_label_all = np.zeros([realizations,num_t2_basis])
    train_label_gaussian_all = np.zeros([realizations,num_t2_basis])
    num_T2_SNR_FA_all = np.zeros([realizations,3])
    ### For each realization
    for i in range(realizations):        
        ### Randomly determine the SNR, the minimum T2, the number of T2s (must < M), and the flip angle FA.
        #SNR = 100 ## for fixed SNR
        if SNR == None:
            SNR = np.random.randint(SNR_boundary_low, SNR_boundary_high)
        T2_min, _ = T2_boundary(SNR, echo_3, echo_last)
        T2_max = T2_max_universal
        M = np.floor(T2_components_resolution_finite_domain(SNR, T2_min, T2_max))
        N_choice = np.arange(1, M+1)
        weight = N_choice ** N_weights ## weighting factor for each choice (may change in the future)
        num_T2 = int(np.random.choice(N_choice, p=weight/weight.sum()))
        FA = np.random.randint(FA_min, 180+1)
        ### Calculate the resolution limit
        resolution = resloution_limit(T2_min, T2_max, M)
        ### Randomly generate T2 peak location with respect to resolution limit.
        #T2_location = T2_location_generator_v3(T2_min, T2_max, num_T2, num_t2_basis, resolution, log_cutoff=10, smooth=False)
        T2_location = T2_location_generator_v3(T2_min, T2_max, num_T2, M_max, scale='log')
        while T2_resolution_examiner(T2_location, resolution) == False:
            T2_location = T2_location_generator_v3(T2_min, T2_max, num_T2, M_max, scale='log')    
        ### Randomly generate T2 peak amplitude. When two or more peaks, minimal detectable amplitude is calculated
        if num_T2==1:
            T2_amplitude = np.array([1.0])
        else: 
            _ , frequency = required_frequency(T2_location)
            minimum_amplitude = minimum_amplitude_calculator(SNR, frequency)
            T2_amplitude = T2_amplitude_generator_v3(num_T2, minimum_amplitude)
        ### Decay curve generation (weighted sum of each T2 component)
        decay_curve = produce_decay_from_lib(decay_lib, T2_location, T2_amplitude, FA)
        ### Add noise to decay curve
        decay_curve_with_noise = signal_with_noise_generation_phase_rotation(signal=decay_curve, SNR=SNR)
        ### T2 basis set embedding (nearest t2_basis neighbors)
        train_label = train_label_generator(T2_location, T2_amplitude, t2_basis)
        ### T2 basis set embedding (gaussian peaks)
        train_label_gaussian = train_label_generator_gaussian_embedding(T2_location, T2_amplitude, T2_min_universal, T2_max_universal, t2_basis, peak_width)
        ### Extract metrics (use t2_basis and train label instead of T2_location and T2_amplitude to prevent basis set embedding error)
        #MWF, MWGMT2, IEWF, IEWGMT2, GMT2 = metrics_extraction_v3(t2_basis, train_label, MW_low_cutoff, MW_high_cutoff, IEW_low_cutoff, IEW_high_cutoff)
        ### Ground truth metrics (use tT2_location and T2_amplitude)
        #MWF_GT, MWGMT2_GT, IEWF_GT, IEWGMT2_GT, GMT2_GT = metrics_extraction_v3(T2_location, T2_amplitude, MW_low_cutoff, MW_high_cutoff, IEW_low_cutoff, IEW_high_cutoff)
        ### Pad T2_location and T2_amplitude to have uniform size
        T2_location = np.pad(T2_location,[(0, N-int(num_T2))], mode='constant', constant_values=0)
        T2_amplitude = np.pad(T2_amplitude,[(0, N-int(num_T2))], mode='constant', constant_values=0)   
        ### Store generated parameters in placeholders       
        T2_location_all[i,:] = T2_location
        T2_amplitude_all[i,:] = T2_amplitude 
        decay_curve_all[i,:] = decay_curve
        decay_curve_with_noise_all[i,:] = decay_curve_with_noise
        train_label_all[i,:] = train_label
        train_label_gaussian_all[i,:] = train_label_gaussian
        num_T2_SNR_FA_all[i,:] = num_T2, SNR, FA
    ### return a data dict
    data = {'T2_location': T2_location_all, 
            'T2_amplitude': T2_amplitude_all,
            'decay_curve': decay_curve_all,
            'decay_curve_with_noise': decay_curve_with_noise_all,
            'train_label': train_label_all,
            'train_label_gaussian': train_label_gaussian_all,
            'num_T2_SNR_FA': num_T2_SNR_FA_all,
            }      
    return data


def mp_yield_training_data(func_produce_training_data,
                           decay_lib,
                           realizations,
                           ncores,
                           SNR_boundary_low=50,
                           SNR_boundary_high=800,
                           echo_3=30,
                           echo_last=320,
                           echo_train_num=32,
                           num_t2_basis=40,
                           FA_min=50,
                           peak_width=1,
                           T2_min_universal=None,
                           T2_max_universal=None,
                           exclude_M_max=False):
    """
    Use multiple cpu cores to accelerate training data production using multiprocessing package.

    Args:
        func_produce_training_data (function): the function to produce training data using a single cpu core.
        decay_lib (array): the decay library.
        realizations (int): the number of simulation realizations.
        ncores (int): number of cpu cores to use.
        SNR_boundary_low (float, optional): lower boundary of SNR. Defaults to 50.
        SNR_boundary_high (float, optional): upper boundary of SNR. Defaults to 800.
        echo_3 (float, optional): the 3rd echo time (ms). Defaults to 30.
        echo_last (float, optional): the last echo time (ms). Defaults to 320.
        echo_train_num (int, optional): the number of echoes in the echo train. Defaults to 32.
        num_t2_basis (int, optional): the number of basis t2s. Defaults to 40.
        FA_min (float, optional): the minimal refocusing flip angle for simulation. Defaults to 50.
        peak_width (float, optional): the variance of the gaussian peak. Defaults to 1.
        T2_min_universal (float, optional): the overall minimal T2 (ms) of the analysis. Defaults to calculate on the fly if None is given.
        T2_max_universal (float, optional): the overall maximal T2 (ms) of the analysis. Defaults to to 2000ms if None is given.
        exclude_M_max (bool, optional): exclude the M_max if True. Defaults to False.

    Returns:
        data_all: a data dictionary concatenated from all cpu cores
    """

    import multiprocessing as mp
    pool = mp.Pool(processes=ncores)
    ### distribute job to each cpu core
    realizations_pool_list = [realizations//ncores]*ncores 
    if realizations%ncores != 0:
        realizations_pool_list.append(realizations%ncores)
    data = pool.starmap(func_produce_training_data, [(decay_lib, realizations, 
                                                      SNR_boundary_low, SNR_boundary_high,
                                                      echo_3, echo_last, echo_train_num,
                                                      num_t2_basis, FA_min, peak_width, 
                                                      T2_min_universal, T2_max_universal,
                                                      exclude_M_max) 
                                                      for realizations in realizations_pool_list])
    pool.close()
    pool.join()
    ### concatenate data calculated from each cpu core
    keys = data[0].keys()
    data_all =  {key: None for key in keys}
    for key in keys:
        data_all[key] = np.concatenate([data[x][key] for x in range(len(data))])    
    return data_all


def plot_all_echoes(img, slice_num, rows, columns, fig_size=None, tight=True):
    """Plot all echoes (axis=-1) of the 4D image data at a given slice (axis=-2). 

    Args:
        img (4D array): the 4D image data.
        slice_num (int): the slice number.
        rows (int): the number of subfigures to plot in each row.
        columns (int): the number of subfigures to plot in each column.
        fig_size (tuple, optional): the figure size. Defaults to None.
        tight (bool, optional): tight layout when plot. Defaults to True.
    """    
    
    if fig_size != None:
        fig = plt.figure(figsize=fig_size)
    else:
        fig = plt.figure()
    for i in range(img.shape[-1]):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img[:,:,slice_num,i])
        plt.title('echo {}'.format(i+1))
        plt.axis('off')
    if tight:
        plt.tight_layout()
    plt.show()


def NN_predict_4D_decay(decay_data, NN_model):
    """
    Use trained neural network to make predictions on 4D image data.

    Args:
        decay_data (4D array): 4D image data with the last dimension of the echo train.
        NN_model (model): the pre-trained neural network model.

    Returns:
        4D array: the predicted spectrum of each voxel
    """
    ### flat voxels, and normalize to the first echo
    decay_flat = decay_data.reshape(
        decay_data.shape[0] * decay_data.shape[1] * decay_data.shape[2],
        decay_data.shape[3])
    decay_flat_norm = decay_flat / (decay_flat[:, 0].reshape(
        decay_data.shape[0] * decay_data.shape[1] * decay_data.shape[2], 1))
    ### use trained model to predict the spectrum
    NN_predict_spectrum_flat = NN_model.predict(decay_flat_norm)
    ### reshape the flat spectrum back to 4D array
    NN_predict_spectrum = NN_predict_spectrum_flat.reshape(
        decay_data.shape[0], decay_data.shape[1], decay_data.shape[2],
        NN_predict_spectrum_flat.shape[1])

    return NN_predict_spectrum


def quantitative_map_production(t2_basis,
                                spectrum,
                                MW_low_cutoff=0,
                                MW_high_cutoff=40,
                                IEW_low_cutoff=40,
                                IEW_high_cutoff=200):
    """
    This function produce 5 quantitative maps from predicted spectra: MWF, MWGMT2, IEWF, IEWGMT2, GMT2.
    Spectrum in a data shape such as (240, 240, 40, 40) with the last dimension indicating the number of basis t2s. 
    This function is calling another function 'metric_extraction_v3'.

    Args:
        t2_basis (array): basis t2s (ms).
        spectrum (4D array): the predicted spectrum of each image voxel.
        MW_low_cutoff (float, optional): the lower boundary of myelin water (ms). Defaults to 0.
        MW_high_cutoff (float, optional): the upper boundary of myelin water (ms). Defaults to 40.
        IEW_low_cutoff (float, optional): the lower boundary of IE water (ms). Defaults to 40.
        IEW_high_cutoff (float, optional): the upper boundary of IE water (ms). Defaults to 200.

    Returns:
        4D array: the last dimension in order: MWF, MWGMT2, IEWF, IEWGMT2, GMT2
    """    

    spectrum_flat = spectrum.reshape(
        spectrum.shape[0] * spectrum.shape[1] * spectrum.shape[2],
        spectrum.shape[3])
    NN_predict_metrics_flat = np.zeros((spectrum_flat.shape[0], 5))
    for item in range(spectrum_flat.shape[0]):
        NN_predict_metrics_flat[item, :] = metrics_extraction_v3(
            t2_basis, spectrum_flat[item, :], MW_low_cutoff, MW_high_cutoff,
            IEW_low_cutoff, IEW_high_cutoff)
    NN_predict_metrics = NN_predict_metrics_flat.reshape(
        spectrum.shape[0], spectrum.shape[1], spectrum.shape[2],
        NN_predict_metrics_flat.shape[1])
    return NN_predict_metrics


def plot_all_slice(maps, nrow, ncol, vmin=None, vmax=None, figsize=(30, 12), fontsize=20, cfontsize=20, cshrink=0.3, cpad=0.05, clocation='bottom'):
    """
    Plot all slices of a 3D image.

    Args:
        maps (3D array): 3D image data.
        nrow (int): number of rows
        ncol (int): number of columns
        vmin (float, optional): minimal intensity. Defaults to None.
        vmax (float, optional): maximal intensity. Defaults to None.
        figsize (tuple, optional): figure size. Defaults to (30, 12).
        fontsize (float, optional): the font size. Defaults to 20.
        cfontsize (float, optional): colorbar font size. Defaults to 20.
        cshrink (float, optional): colorbar shrink factor. Defaults to 0.3.
        cpad (float, optional): padding between figure and colorbar. Defaults to 0.05.
        clocation (str, optional): colorbar location. Defaults to 'bottom'.
    """    
    
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
    for i, ax in enumerate(axs.flat):
        if (vmin and vmax) is not None:
            img = ax.imshow(maps[:, :, i], vmin=vmin, vmax=vmax)
        else:
            img = ax.imshow(maps[:, :, i])
        ax.set_title('slice {}'.format(i+1), fontsize=fontsize)
        ax.axis('off')
    plt.tight_layout()
    cbar = fig.colorbar(img, ax=axs, shrink=cshrink,
                        location=clocation, pad=cpad)
    cbar.ax.tick_params(labelsize=cfontsize)
    plt.show()