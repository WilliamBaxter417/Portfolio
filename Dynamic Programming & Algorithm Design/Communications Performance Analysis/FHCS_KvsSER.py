import math
import numpy
import itertools
import pandas as pd
import plotly.express as px
import DFRC

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

## MAIN SIMULATION BLOCK
for m_cnt in range(len(M_arr)):
    M = M_arr[m_cnt]                        # Number of transmit antennas for current iteration.
    K = 2 * M                               # Set number of hopping frequencies.
    print('Calculating frequency hop K = %d:\n' % K)

    ## INITIALISE FHCS WAVEFORM PARAMETERS FOR CURRENT K VALUE
    Kv = numpy.arange(0,K)                  # FH index vector (initialised as row vector).
    Kv = Kv[:,numpy.newaxis]                # Transpose Kv to column vector.
    Delta_f = BW / K                        # FH interval (frequency step).
    Delta_t = 1 / Delta_f                   # FH duration.
    Ns = 2 * K                              # No. of samples per FH duration.
    ns = numpy.arange(0, Ns)                # Discrete-time index (fast-time).
    Fn = 1 / (2 * K)                        # Normalised frequency.
    DuCy = (Q * Delta_t) / PRI              # Duty cycle (pulse width).
    Tw = DuCy * PRI                         # Pulsewidth.
    FH_set = 1 / math.sqrt(Ns) * numpy.exp(1j * 2 * numpy.pi * Fn * Kv * ns)  # Generate (K x ns) matrix of hopping frequencies.

    ## BUILD SYMBOL DICTIONARIES FOR CURRENT ITERATION
    ### Refer to BuildDict function in DFRC.py for detailed listing of output variables.
    L, L_tilde, N_bits, D, D_T_DIR, D_T_CMP = DFRC.BuildDict(K, M)

    ## INITIALISE TRANSMIT SIGNAL
    Mv = numpy.arange(0,M)
    sv = numpy.exp(-1j * 2 * numpy.pi * Mv * math.sin(theta_c * numpy.pi/180))  # Steering vector of the transmit array towards theta_c.
    W_FHCS = numpy.zeros((M, Q*Ns), dtype = 'complex')      # Initialise FHCS waveform realisation.
    sym_tx = numpy.zeros((MC, M, Q), dtype = 'int')         # Transmitted symbol stream.
    sym_dec = numpy.zeros((MC, NSNR, M, Q), dtype = 'int')  # Received symbol stream.

    ## MONTE-CARLO BLOCK
    for mc in range(MC):
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

        if ((mc+1) % 100 == 0):
            print('Number of MC runs done = %d...\n' % (mc+1))  # Print status.

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