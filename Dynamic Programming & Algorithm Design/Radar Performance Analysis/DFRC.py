from itertools import combinations
import math
import numpy

## BUILDS FULL, DIRECTLY TRUNCATED AND COMPLEMENTARILY TRUNACTED SYMBOL DICTIONARIES
## INPUTS:
### K - number of hopping frequencies.
### M - number of transmitting elements.
## OUTPUTS:
### L: maximum no. of symbols in full dictionary.
### L_tilde: maximum no. of symbols in useable (truncated) dictionaries.
### N_bits: no. of bits composing a FHCS symbol (assuming dictionary is truncated).
### D: full dictionary.
### D_T_DIR: directly truncated dictionary.
### D_T_CMP: complementarily truncated dictionary.
def BuildDict(K,M):
    # Initialise symbol dictionary parameters:
    L = math.comb(K, M)
    N_bits = int(numpy.floor(numpy.log2(L)))
    L_tilde = pow(2, N_bits)
    # Build full dictionary:
    SymbDict = numpy.array(list(combinations(range(0,K),M)))
    SymbDict = SymbDict.transpose()
    # Build directly truncated dictionary:
    SymbDict_T_DIR = SymbDict[:,0:L_tilde]
    # Build complementarily truncated dictionary:
    SymbDict_T_CMP = numpy.zeros((M,L_tilde), dtype = int)
    for i in range(0,int(L_tilde/2)):
        SymbDict_T_CMP[:,2*i] = SymbDict[:,i]
        SymbDict_T_CMP[:,2*i+1] = SymbDict[:,L-i-1]

    return L, L_tilde, N_bits, SymbDict, SymbDict_T_DIR,SymbDict_T_CMP

## BUILDS DICTIONARY OF BITMAPPINGS FOR TRUNCATED SYMBOL DICTIONARIES
## INPUTS:
### N_bits: no. of bits composing a FHCS symbol (assuming dictionary is truncated).
### L_tilde: maximum no. of symbols in useable (truncated) dictionaries.
## OUTPUT:
### BD: (N_bits x L_tilde) binary matrix of bitmappings, whose column index values correspond to the symbol index (for e.g., column 0 -> symbol 0, column 1 -> symbol 1, etc).
def BuildBitMap(N_bits, L_tilde):
    # Initialise bitmap d
    BD = numpy.zeros((N_bits, L_tilde), dtype = 'int')
    for i in range(L_tilde):
        b = numpy.array([int(d) for d in str(bin(i))[2:]])
        BD[:,i] = numpy.concatenate([numpy.zeros(N_bits-len(b)), b], axis = 0)

    return BD

## COMPUTES 1-D AUTO-AMBIGUITY FUNCTION (AAF)
## INPUTS:
### x: signal of interest (1-D complex vector).
## OUTPUTS:
### amb: ambiguity function (1-D complex vector).
def AAF(x):
    nx = x.shape[0]
    # Initialise 1-D auto-ambiguity function (AAF) vector:
    amb = numpy.zeros((2*nx-1,1))
    # Zero pad to support element-wise multiplication:
    Xext = numpy.concatenate([numpy.zeros((nx-1), dtype='complex'), x, numpy.zeros((nx-1), dtype = 'complex')], axis = 0)
    # Populate 1-D AAF vector by iterating over n (time delay):
    for n in range(2*nx-1):
        amb[n,:] = numpy.abs( numpy.sum( numpy.multiply(x, numpy.conjugate(Xext[n:n+nx])) ) )

    return amb

## COMPUTES 1-D CROSS-AMBIGUITY FUNCTION (CAF)
## INPUTS:
### x: 1st signal of interest (1-D complex vector).
### y: 2nd signal of interest (1-D complex vector).
## OUTPUTS:
### amb: ambiguity function (1-D complex vector).
def CAF(x, y):
    nx = x.shape[0]
    ny = y.shape[0]
    # If y is shorter than x, then zeropad y:
    if (ny < nx):
        X = x
        Y = [y, numpy.zeros((1, nx-ny), dtype = 'complex')]
    # If y is longer than x, then zeropad x:
    elif (ny > nx):
        X = [x, numpy.zeros((1, ny-nx), dtype = 'complex')]
    else:
        X = x
        Y = y

    NY = Y.shape[0]
    Xext = numpy.concatenate([numpy.zeros((NY-1), dtype = 'complex'), X, numpy.zeros((NY-1), dtype = 'complex')], axis = 0)
    amb = numpy.zeros((2*NY-1,1))
    # Populate 1-D AAF vector by iterating over n (time delay):
    for n in range(2*NY-1):
        amb[n,:] = numpy.abs( numpy.sum( numpy.multiply(Y, numpy.conjugate(Xext[n:n+NY])) ) )

    return amb

## DECODES FHCS WAVEFORMS
## INPUTS:
### r: FHCS waveform of interest.
### W: Set of transmitted FHCS waveforms.
### M: number of transmit antennas.
## OUTPUTS:
### FHCS_idx: decoded symbol index.
def decode_FHCS(r, W, M):
    # Compute Hermitian inner product:
    Yr = numpy.dot(r,numpy.transpose(numpy.conjugate(W)))
    # Sort complex valued Yr according to magnitude in descending order
    idx = (-numpy.abs(Yr)).argsort()
    # Construct decoded symbol by taking M largest entries of idx and sorting them in ascending order
    FHCS_idx = numpy.transpose(numpy.sort(numpy.transpose(idx)[0:M], axis = 0))

    return FHCS_idx