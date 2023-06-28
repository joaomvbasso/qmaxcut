'''
This runs the iteration from BFMVZ21 to solve quantum maxcut,
but using the qaoa state for maxcut. This is as in the paper,
except now we compute the expectation over ZZ, as well as YY and XX.
'''

import sys
import numpy as np
import math
import cmath
import scipy.optimize as opt
from itertools import product

np.set_printoptions(precision=3, suppress=True)

p = int(sys.argv[1])
bs_size = 2*p+1
bitstrings = list(product([-1,1], repeat=bs_size))

def eib_entries(bit1, bit2, beta):
	if bit1 == bit2:
		return math.cos(beta)
	return 1j * math.sin(beta)

def f_betas(bs, betas):
	all_betas = np.concatenate((betas, np.array([-beta for beta in betas[::-1]])))
	#print(all_betas)
	prod = 0.5
	for j in range(bs_size-1):
		prod *= eib_entries(bs[j], bs[j+1], all_betas[j])
	return prod

def g_betas(bs, betas):
	all_betas = np.concatenate((betas, np.array([-beta for beta in betas[::-1]])))
	prod = 0.5
	for j in range(bs_size-1):
		if j==p-1:
			prod *= eib_entries(bs[j], -bs[j+1], all_betas[j])
		else:
			prod *= eib_entries(bs[j], bs[j+1], all_betas[j])
	return prod

def gamma_dp(gammas, bs1, bs2):
	all_gammas = np.concatenate((gammas, np.array([0]), np.array([-gamma for gamma in gammas[::-1]])))
	#print(all_gammas)
	total = 0
	for j in range(bs_size):
		total += all_gammas[j] * bs1[j] * bs2[j]
	return total

# build the Hs
def get_Hp(gammas, betas):
	Hs = np.ones((p+1, 2**bs_size), dtype=complex)
	for m in range(1, p+1):
		for ind1, bs1 in enumerate(bitstrings):
			total = 0
			for ind2, bs2 in enumerate(bitstrings):
				total += f_betas(bs2, betas) * Hs[m-1][ind2] * (gamma_dp(gammas, bs1, bs2) ** 2)
			Hs[m][ind1] = cmath.exp(-0.5 * total)
		'''
		# constructs G matrix (for debugging)
		G = np.zeros((bs_size, bs_size), dtype=complex)
		for row in range(bs_size):
			for col in range(bs_size):
				entry = 0
				for ind, bs in enumerate(bitstrings):
					entry += f_betas(bs, betas) * Hs[m][ind] * bs[row] * bs[col]
				G[row][col] = entry
		print(f'matrix G {m}:')
		print(G)

		'''

	return Hs[p]

def get_expectation(params, verbose=False):
	gammas = params[:p]
	betas = params[p:]
	assert len(gammas) == p
	assert len(betas) == p

	Hp = get_Hp(gammas, betas)
	totalZZ = 0
	totalYY = 0
	totalXX = 0

	for ind1, bs1 in enumerate(bitstrings):
		for ind2, bs2 in enumerate(bitstrings):
			totalZZ += bs1[p] * bs2[p] * f_betas(bs1, betas) * f_betas(bs2, betas) * Hp[ind1] * Hp[ind2] * gamma_dp(gammas, bs1, bs2)
			totalYY += bs1[p] * bs2[p] * g_betas(bs1, betas) * g_betas(bs2, betas) * Hp[ind1] * Hp[ind2] * gamma_dp(gammas, bs1, bs2)
			totalXX += g_betas(bs1, betas) * g_betas(bs2, betas) * Hp[ind1] * Hp[ind2] * gamma_dp(gammas, bs1, bs2)

	totalZZ *= 0.5 * 1j
	totalYY *= -0.5 * 1j
	totalXX *= 0.5 * 1j

	if verbose:
		print(f'ZZ term:{totalZZ}')
		print(f'YY term:{totalYY}')
		print(f'XX term:{totalXX}')

	return 0.5 * (totalXX + totalYY + totalZZ)

obj = opt.minimize(get_expectation, np.random.rand(2*p), tol=1e-3, options={'maxiter':100, 'disp':True})
print(f'Optimization result: {obj.message}')
print(f'Minimum arguments: {obj.x}')
print(f'Minimum energy: {obj.fun}')





