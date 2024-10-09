- FHCS is a novel signalling strategy developed for dual-function radar communications systems.
- We study the impact of information embedding on the transmitted radar waveforms through an assessment of the average ambiguity function (AF), obtained via Monte Carlo methods.
- Provide a link to one of the publications for further reading and clarification of the model and its results.
- The script from which the following document is based is located [here](https://github.com/WilliamBaxter417/Portfolio/blob/main/Dynamic%20Programming%20%26%20Algorithm%20Design/Communications%20Performance%20Analysis/FHCS_KvsSER.py).
- The \* indicates a reference to my thesis. <br>

```python
import math
import numpy
import itertools
import pandas as pd
import plotly.express as px
import DFRC
```

```python
## INITIALISE DFRC SYSTEM
Fc = 8e9                                    # Carrier frequency.
BW = 100e6                                  # Bandwidth of MIMO radar system.
Fs = 2*BW                                   # Sampling frequency.
PRF = 100e3                                 # Pulse repetition frequency.
PRI = 1/PRF                                 # Pulse repetition interval.
Q = 10                                      # No. of subpulses in one radar pulse.
M_arr = [2, 4, 6, 8, 10]                    # Array of number of transmit antennas.
## INITIALISE COMMUNICATIONS PARAMETERS
sig_pwr = 1                                 # Signal power.
theta_c = 20                                # Direction of communications receiver.
SNR = numpy.arange(-10,11)                  # Signal-to-noise-ratio (decibels).
NSNR = len(SNR)                             # No. of SNR values.
SER = numpy.zeros((len(M_arr), NSNR))       # Initialise vector of SER results.
MC = 1000                                   # No. of Monte Carlo runs.
N_sym = Q*MC                                # Total number of transmitted communication symbols.
sqrt05 = 1/math.sqrt(2)                     # Convenience.
```

- Main simulation is wrapped around a Monte-Carlo block.
- For each iteration of the random process, a random sequence of symbols are extracted from the truncated dictionary and embedded within the transmitted radar waveform.
```python
# For the MC-th run:
## Generate communications channel coefficient:
beta_c = numpy.exp(1j * 2 * numpy.pi * numpy.random.rand(1))
## Randomly select L_tilde symbols from full dictionary D for transmission:
C_i = numpy.ceil(L_tilde * numpy.random.rand(Q)).astype('int') - 1  # Randomly generate L_tilde symbol indices.
C = D[:,C_i]            # Draw symbols from full dictionary D.
sym_tx[mc,:,:] = C      # Store selected symbols for SER calculation.

## Build FHCS waveform subpulse-by-subpulse:
for q in range(Q):
    W_FHCS[:,q*Ns:(q+1)*Ns] = FH_set[C[:,q],:]
### Beamform FHCS waveforms in direction of theta_c:
fhcs_tx = numpy.dot(sv,W_FHCS)
### Reshape into row-vector:
FHCS_TX = numpy.reshape(fhcs_tx, [1,fhcs_tx.shape[0]])
```
- Then, we iterate through each SNR value to embed noise in the transmit signal, followed by decoding the ensuing receive signal.
```python
## Simulate receive signal and perform symbol decoding:
for nsnr in range(len(SNR)):
    ### Calculate signal noise power based on current SNR value:
    noisePwr = numpy.divide(sig_pwr,numpy.pow(10,SNR[nsnr]/10))
    ### Generate AWGN complex noise scaled by noise power:
    n = numpy.sqrt(noisePwr) * sqrt05  * (numpy.random.randn(W_FHCS.shape[1],1) + 1j * numpy.random.randn(W_FHCS.shape[1],1))
    ### Embed noise into transmit signal (FHCS_TX) to obtain receive signal (FHCS_RX):
    FHCS_RX = beta_c * FHCS_TX + numpy.transpose(numpy.conjugate(n))
    ### Decode recieved signal subpulse-by-subpulse:
    for q in range(Q):
        sym_dec[mc,nsnr,:,q] = DFRC.decode_FHCS(FHCS_RX[:,q*Ns:(q+1)*Ns], FH_set, M)
```
- Once the received signals from all Monte Carlo runs have been decoded, we iterate through each SNR value and calculate the corresponding symbol error rate (SER).
```python
## SER CALCULATION BLOCK
for nsnr in range(len(SNR)):
    # For the NSNR-th slice:
    ## Generate (MC x Q) two-element Boolean matrix of symbol errors (se) by comparing transmitted symbol indices (sym_tx) with decoded symbol indices (sym_dec):
    ## NOTE: '1' entries indicate error
    se = 1 - (numpy.sum((sym_tx == sym_dec[:,nsnr,:,:]).astype(int), axis = 1)==M).astype(int)
    ## Vectorise (MC x Q) matrix of symbol errors (se) into a ((MC*Q) x 1) column vector:
    se_vec = numpy.transpose(numpy.reshape(se, (1,se.shape[0]*se.shape[1])))
    ## Obtain SER by computing average across number of transmitted symbols (N_sym):
    SER[m_cnt,nsnr] = numpy.sum(se_vec, axis = 0)/N_sym
```

- Plot results
```python
## GENERATE PLOTS
# Build data frame for SER:
df = pd.DataFrame(SER.T)
fig = px.line(df, x = SNR, y = df.columns)
# Set y-axis to log-scale for ease of interpretation and update axes titles and legend:
fig.update_layout(xaxis_title = "SNR", yaxis_title = "SER", yaxis_type = 'log', legend_title_text = 'Hopping frequencies')
fig.update_traces({'name': 'K = 4'}, selector = {'name': '0'})
fig.update_traces({'name': 'K = 8'}, selector = {'name': '1'})
fig.update_traces({'name': 'K = 12'}, selector = {'name': '2'})
fig.update_traces({'name': 'K = 16'}, selector = {'name': '3'})
fig.update_traces({'name': 'K = 20'}, selector = {'name': '4'})
fig.show()
```

Results:
![SERvsSNR](https://github.com/WilliamBaxter417/Portfolio/blob/main/Dynamic%20Programming%20%26%20Algorithm%20Design/Communications%20Performance%20Analysis/SERvsSNR.png)

