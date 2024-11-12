# Radar Performance Analysis of the Frequency-Hopped Code Selection Scheme
The Frequency-Hopped Code Selection (FHCS) scheme is a novel signalling strategy developed for dual-function radar communications (DFRC) systems. In this project, we utilise the Python environment to implement the FHCS scheme and to study the impact of information embedding on the transmitted radar waveforms through an assessment of the average ambiguity function (AF).

We begin by importing the following libraries.
```python
# Math and Plotting libraries
import math
import numpy
import itertools
import matplotlib
import matplotlib.pyplot as plt

# Personal libraries
import DFRCsubs

# Miscellaneous initialisations
matplotlib.use('TkAgg')
```
We include a pre-built module ```DFRCsubs``` that supports this main module ```AvgAFmain```, whose library of functions we will reference throughout this project. Having imported the relevant libraries, we initialise the following parameters that model our virtual DFRC system. As we are implementing a small-scale scenario that can be practically realised on non-HPC workstations, we choose the number of transmitting elements ```M``` to be 2, which entails the number of hopping frequencies ```K``` to be 4. We also set the number of subpulses (chips) ```Q``` comprising the radar fast-time as 5.
```python
## INITIALISE DFRC PARAMETERS
Fc = 8e9                                    # Carrier frequency.
BW = 100e6                                  # Bandwidth of MIMO radar system.
Fs = 2*BW                                   # Sampling frequency.
PRF = 100e3                                 # Pulse repetition frequency.
PRI = 1/PRF                                 # Pulse repetition interval.
M = 2                                       # No. of transmit antennas.
K = 2*M                                     # No. of hopping frequencies. Note that Q<=K. The maximum number of subpulses is when Q=K.
Kv = numpy.arange(0,K)                      # FH index vector (initialised as row vector).
Kv = Kv[:,numpy.newaxis]                    # Transpose to column vector.
Delta_f = BW/K                              # FH interval (frequency step).
Delta_t = 1/Delta_f                         # FH duration.
Ns = 2*K                                    # No. of samples per FH duration.
ns = numpy.arange(0,Ns)                     # Discrete-time index (fast-time).
Fn = 1/(2*K)                                # Normalised frequency.
Q = 5                                       # No. of subpulses in one radar pulse.
DuCy = (Q*Delta_t)/PRI                      # Duty cycle (pulse width).
Tw = DuCy * PRI                             # Pulsewidth.
TBWP = round(Tw*BW)                         # Time-bandwidth product.
sqrt05 = 1/math.sqrt(2)                     # Convenience.
FH_set = 1/math.sqrt(Ns) * numpy.exp(1j * 2 * numpy.pi * Fn * Kv * ns)  # Generate (K x ns) matrix of hopping frequencies.
tau = numpy.arange(-Q*Ns+1,Q*Ns)/Fs         # Vector of time-delays.
```
Following this initialisation procedure, we check that the time-bandwidth product of the DFRC system, and thus its ensuing radar operation, is properly satisfied. This is achieved when the product of ```Q``` and ```K``` is less than or equal to the ```TBWP``` variable (which is the product of the pulsewidth and bandwidth of the transmit signal).
```python
## CHECK TIME-BANDWIDTH PRODUCT
if (Q*K <= TBWP):
    print('Q * K = %d, TBWP = %d >>> GOOD' % (Q*K,TBWP))
else:
    print('Q * K = %d, TBWP = %d >>> BAD' % (Q*K,TBWP))
```
```python
NUMBER OF REALISATIONS - COMPLETE CODEBOOK: 7776
NUMBER OF REALISATIONS - TRUNCATED CODEBOOK: 1024
Q * K = 20, TBWP = 20 >>> GOOD
```
Having verified the time-bandwidth product of our system, we proceed to build the accompanying communication symbol dictionaries. To do this, we first delineate the ```BuildDict()``` function within ```DFRCsubs```, which takes the number of hopping frequencies ```K``` and number of chips ```Q``` as inputs, and returns the complete symbol dictionary ```SymbDict```, directly truncated symbol dictionary ```SymbDict_T_DIR```, complementarily truncated symbol dictionary ```SymbDict_T_CMP```, the cardinality of the complete symbol dictionary ```L``` and truncated symbol dictionaries ```L_tilde```, and the maximum number of bits necessary to encode the truncated dictionaries ```N_bits```.
```python
## INPUTS:
### K: number of hopping frequencies.
### M: number of transmitting elements.
## OUTPUTS:
### L: maximum no. of symbols in full dictionary.
### L_tilde: maximum no. of symbols in useable (truncated) dictionaries.
### N_bits: no. of bits composing a FHCS symbol (assuming dictionary is truncated).
### D: full dictionary.
### D_T_DIR: directly truncated dictionary.
### D_T_CMP: complementarily truncated dictionary.
def BuildDict(K, M):
    # Initialise symbol dictionary parameters:
    L = math.comb(K, M)
    N_bits = int(np.floor(np.log2(L)))
    L_tilde = pow(2, N_bits)
    # Build full dictionary:
    SymbDict = np.array(list(combinations(range(0,K),M)))
    SymbDict = SymbDict.transpose()
    # Build directly truncated dictionary:
    SymbDict_T_DIR = SymbDict[:,0:L_tilde]
    # Build complementarily truncated dictionary:
    SymbDict_T_CMP = np.zeros((M,L_tilde), dtype = int)
    for i in range(0,int(L_tilde/2)):
        SymbDict_T_CMP[:,2*i] = SymbDict[:,i]
        SymbDict_T_CMP[:,2*i+1] = SymbDict[:,L-i-1]

    return L, L_tilde, N_bits, SymbDict, SymbDict_T_DIR, SymbDict_T_CMP
```
Now, calling this function within ```AvgAFmain```:
```python
## BUILD SYMBOL DICTIONARIES
### Refer to BuildDict function in DFRCsubs.py for detailed listing of output variables.
L, L_tilde, N_bits, D, D_T_DIR, D_T_CMP = DFRCsubs.BuildDict(K, M)
```
To aid the viewer's comprehension, we construct the function ```PrintDictionary()``` within ```DFRCsubs```, which prints the sequence of FH codes comprising the input symbol dictionary.
```python
## PRINTS VERBOSE SYMBOL DICTIONARY
## INPUTS:
### D: symbol dictionary
def PrintDictionary(D):
    # Iterate over rows of input symbol dictionary:
    for i in np.arange(D.shape[0]):
        print('FH codes assigned to transmit element m = %d: ' % i, end = ' ')
        # Iterate over columns of input symbol dictionary and print FH codes:
        for j in np.arange(D.shape[1]):
            if j != D.shape[1] - 1:
                print('%d  ' % D[i, j], end = ' ')
            else:
                print('%d  ' % D[i, j])
```
We parse the symbol dictionaries ```D```, ```D_T_DIR``` and ```D_T_CMP``` to ```PrintDictionary()```, which returns the following console output:
```python
{D} -> Complete symbol dictionary (cardinality L = 6)
-----------------------------------------------------
FH codes assigned to transmit element m = 0:  0   0   0   1   1   2  
FH codes assigned to transmit element m = 1:  1   2   3   2   3   3  

{D_T_DIR} -> Directly truncated symbol dictionary (cardinality L_tilde = 4)
---------------------------------------------------------------------------
FH codes assigned to transmit element m = 0:  0   0   0   1  
FH codes assigned to transmit element m = 1:  1   2   3   2  

{D_T_CMP} -> Complementarily truncated symbol dictionary (cardinality L_tilde = 4)
----------------------------------------------------------------------------------
FH codes assigned to transmit element m = 0:  0   2   0   1  
FH codes assigned to transmit element m = 1:  1   3   2   3  
```
For the matrix of FH codes comprising a symbol dictionary, the respective mapping between its rows and columns to the transmitting elements and symbols are indicated by their ordinal indices. For example, with our simulation employing two transmitting elements, we have row 0 mapping to transmit element 0, and row 1 mapping to transmit element 1. Meanwhile, column 0 maps to symbol 0, column 1 maps to symbol 1, etc. Hence, with the complete dictionary ```D``` possessing a cardinality of 6, it has six columns, while ```D_T_DIR``` and ```D_T_CMP``` have four. Also, despite our truncated dictionaries having the same cardinality, notice the difference in their construction from the complete dictionary ```D```. Specifically, the directly truncated dictionary ```D_T_DIR``` is formed from columns (symbols) 0, 1, 2 and 3 of ```D```, while the complementarily truncated dictionary ```D_T_CMP``` is formed from columns (symbols) 0, 1, 5 and 6 of ```D```.

As the symbol dictionary employed by the DFRC system must be truncated to ensure proper communications performance, we select the complementarily truncated dictionary ```D_T_CMP``` with which to assess the average AF performance. For convenience, we extract those indices from ```D``` which build ```D_T_CMP``` for use later on.
```python
# Store symbol indices from full dictionary (D) used to construct complementarily truncated dictionary (D_T_CMP):
L_delta = L - L_tilde
SymInd = numpy.concatenate((numpy.arange(int(L_tilde-L_tilde/2)), numpy.arange(int(L_tilde - L_tilde/2 + L_delta),L)))
```

We initialise those variables used to store the results of the Monte Carlo simulation, and generate the complete set of permuted symbol indices ```WPI``` forming all possible radar waveform realisations:
```python
# Initialise Monte Carlo variables
W_FHCS = numpy.zeros((M,Q*Ns), dtype = 'complex')   # Initialise FHCS waveform realisation.
afs = numpy.zeros((len(tau),M,M))                   # Initialise AF results matrix.
AFs = numpy.zeros((len(tau),M,M))                   # Initialise AF results matrix.

# Build matrix of symbol index permutations constructing each of the NW_trunc realisations:
WPI = np.array(list(itertools.product(SymInd, repeat = Q)))
NW_trunc = WPI.shape[0]
print('NUMBER OF WAVEFORM REALISATIONS FOR TRUNCATED CODEBOOK: %d' % NW_trunc)
```

- Begin Monte Carlo simulation.
```python
## SYNTHESISE AF FOR EACH WAVEFORM REALISATION:
print('Iterating over NW_trunc = %d realisations...\n' % NW_trunc)
for i in range(NW_trunc):
    # Obtain waveform realisation index:
    WPi = WPI[i]
    # Build matrix of FH codes forming i-th DFRC waveform realisation:
    C = D[:,WPi]
    # Build FHCS waveform subpulse-by-subpulse:
    for q in range(Q):
        W_FHCS[:,q*Ns:(q+1)*Ns] = FH_set[C[:,q],:]
    # Iterate over pair-wise combinations of M available DFRC waveforms:
    for m1 in range(M):
        for m2 in range(M):
            # Compute auto-AF:
            if (m1 == m2):
                AF_C = numpy.squeeze(DFRC.AAF(W_FHCS[m1,:]))
                afs[:,m1,m2] = AF_C
            # Compute cross-AF:
            else:
                AF_C = numpy.squeeze(DFRC.CAF(W_FHCS[m1,:], W_FHCS[m2,:]))
                afs[:,m1,m2] = AF_C
            # Iteratively sum auto- and cross-AF to give total AF
            AFs = AFs + afs

    if (i % 1000 == 0):
        print('Completed %d realisations...\n' % i)     # Print status.
```

- Data in AFs is cleaned to extract and store the auto-AFs (AAF) and cross-AFs (CAF) in their own variables.
- AAF and CAF are summed and then averaged across the number of possible transmitted waveforms.
```python
## DATA CLEANING AND EXTRACTION
# Reshape AFs_trunc matrix to extract auto-AF and cross-AF results:
n3, n1, n2 = AFs.shape
FHCS_AF = numpy.transpose(numpy.reshape(AFs, (n3, n1*n2)))

# Build logical indexes for extraction:
v = numpy.reshape(numpy.diag(numpy.ones(M, dtype = 'int'),0),(4,1))
AAF_ind = numpy.nonzero(v)[0]
CAF_ind = numpy.nonzero(1 - v)[0]

# Extract auto-AF and cross-AF results:
FHCS_AAF = numpy.sum(FHCS_AF[AAF_ind,:], axis = 0)
FHCS_CAF = numpy.sum(FHCS_AF[CAF_ind,:], axis = 0)

# Sum auto-AF and cross-AF and compute average AF across M*Q*NW_trunc total waveforms:
fhcs_avgAF = numpy.divide(FHCS_AAF + FHCS_CAF, M*Q*NW_trunc)
FHCS_avgAF = numpy.divide(fhcs_avgAF, numpy.max(fhcs_avgAF))
```

- Generate plots.
```python
## GENERATE PLOTS
# Build time indices normalised by subpulse-width Delta_t:
int_del = range(-Q,Q+1)
tau_chip = numpy.divide(tau, Delta_t)
scale = Ns/(Delta_t * Fs)

# Generate zero-Doppler plot:
fig = px.line(x = numpy.divide(tau_chip,scale), y = 10 * numpy.log10(FHCS_avgAF))
fig.update_layout(xaxis_title = "Chip delay", yaxis_title = "Amplitude (dB)")
fig.show()
```

Results:
<p align="center">
  <img src="https://github.com/WilliamBaxter417/Portfolio/blob/main/Dynamic%20Programming%20%26%20Algorithm%20Design/Radar%20Performance%20Analysis/images/avgAF.png" />
</p>








