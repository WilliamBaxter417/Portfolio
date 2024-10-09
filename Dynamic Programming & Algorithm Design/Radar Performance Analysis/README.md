- FHCS is a novel signalling strategy developed for dual-function radar communications systems.
- We study the impact of information embedding on the transmitted radar waveforms through an assessment of the average ambiguity function (AF), obtained via Monte Carlo methods.
- Provide a link to one of the publications for further reading and clarification of the model and its results.
- The script from which the following document is based is located [here](https://github.com/WilliamBaxter417/Portfolio/blob/main/Dynamic%20Programming%20%26%20Algorithm%20Design/Radar%20Performance%20Analysis/avgAF.py).
- The \* indicates a reference to my thesis. <br>

- Begin by importing the following libraries. We will use the Plotly Python Graphing library (insert reason). <br>
```python
import math
import numpy
import itertools
import plotly.express as px
import plotly.io as pio
import DFRC
```
- Initialise the following parameters to model the virtual DFRC system. <br>
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

- Build the symbol dictionaries.
```python
## BUILD SYMBOL DICTIONARIES
### Refer to BuildDict function in DFRC.py for detailed listing of output variables.
L, L_tilde, N_bits, D, D_T_DIR, D_T_CMP = DFRC.BuildDict(K,M)
L_delta = L - L_tilde
```

- Since we are compressing the transmitted radar pulse, the ensuing time-bandwidth product must be satisfied to ensure proper radar operation\*.
```python
## CHECK TIME-BANDWIDTH PRODUCT
NW = pow(L,Q)
NW_trunc = pow(L_tilde,Q)
print('NUMBER OF REALISATIONS - COMPLETE CODEBOOK: %d' % NW)
print('NUMBER OF REALISATIONS - TRUNCATED CODEBOOK: %d' % NW_trunc)
if (Q*K <= TBWP):
    print('Q * K = %d, TBWP = %d >>> GOOD' % (Q*K,TBWP))
else:
    print('Q * K = %d, TBWP = %d >>> BAD' % (Q*K,TBWP))
```

- Symbol dictionary must be truncated to ensure proper communications performance.
- Extract indices for those symbols from the full dictionary which will comprise the truncated dictionary\*.
```python
## CALCULATE AVERAGE AF ACROSS ALL REALISATIONS FOR TRUNCATED DICTIONARY
# Store symbol indices from full dictionary used to construct complementarily truncated dictionary:
SymInd = numpy.concatenate((numpy.arange(int(L_tilde-L_tilde/2)), numpy.arange(int(L_tilde - L_tilde/2 + L_delta),L)))
```

- Initialise variables used to store results of Monte Carlo simulation.
```python
W_FHCS = numpy.zeros((M,Q*Ns), dtype = 'complex')   # Initialise FHCS waveform realisation.
afs = numpy.zeros((len(tau),M,M))                   # Initialise AF results matrix.
AFs = numpy.zeros((len(tau),M,M))                   # Initialise AF results matrix.
```

- For the truncated dictionary, generate the complete set of permuted symbol indices possible for transmission by the DFRC system\*.
```python
# Build matrix of symbol index permutations constructing each of the NW_trunc realisations:
WPI = numpy.array(list(itertools.product(SymInd, repeat = Q)))
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
![avgAF](https://github.com/WilliamBaxter417/Portfolio/blob/main/Dynamic%20Programming%20%26%20Algorithm%20Design/Radar%20Performance%20Analysis/avgAF.png)








