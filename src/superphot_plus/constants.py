"""A collection of constants used by many modules in the package."""

import numpy as np

# Nested sampling priors
PRIOR_A = [-0.2, 1.2, 0., 0.5]
PRIOR_BETA = [0., 0.02, 0.0052, 1.5 * 0.000336]
PRIOR_GAMMA = [-2., 2.5, 1.1391, 1.5 * .1719]
PRIOR_T0 = [-100., 200., 0., 50.]
PRIOR_TAU_RISE = [-1.0, 3., 0.5990, 1.5 * 0.2073]
PRIOR_TAU_FALL = [0.5, 4., 1.4296, 1.5 * 0.1003]
PRIOR_EXTRA_SIGMA = [-5., -0.5, -1.5364, 0.2691]

PRIOR_A_g = [0., 5., 1.0607, 1.5 * 0.1544]
PRIOR_BETA_g = [1., 1.07, 1.0424, 0.0026]
PRIOR_GAMMA_g = [0.8, 1.2, 1.0075, 0.0139]
PRIOR_T0_g = [1. - 0.0006, 1.0006, 0.9999 + 8.9289e-5, 1.5 * 4.5055e-05]
PRIOR_TAU_RISE_g = [0.5, 2., 0.9663, 0.0128]
PRIOR_TAU_FALL_g = [0.1, 3., 0.5488, 0.0553]
PRIOR_EXTRA_SIGMA_g = [0.2, 2., 0.8606, 0.0388]

# Nested sampling parameters
MAX_ITER = 5000
DLOGZ = 0.5
NLIVE = 50

# Numpyro parameters
PAD_SIZE = 30

# Classifier parameters
INPUT_DROPOUT_FRAC = 0.2
HIDDEN_DROPOUT_FRAC = 0.5
SEED = 9876
EPOCHS = 500
VALID_RATIO = 0.9
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_FOLDS = 20

# New classifications parameters
MEANS_TRAINED_MODEL = np.array([5.18928404e-03, 1.27960653e+00, 7.16663998e-01, 1.59438422e+00,
                      -1.45215889e+00, 9.54225774e-01, 1.04234314e+00, 1.00733255e+00,
                      9.99977873e-01, 9.66049977e-01, 5.62890503e-01, 8.63290836e-01,
                      -6.27014662e+00])

STDDEVS_TRAINED_MODEL = np.array([5.00371052e-04, 3.17145223e-01, 3.93625733e-01, 2.58445333e-01,
                     3.72614525e-01, 2.15892037e-01, 2.44030784e-03, 1.27667116e-02,
                     5.11363683e-05, 1.18290613e-02, 6.24454144e-02, 4.99345471e-02,
                     8.98741848e-01])

# Plotting parameters
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
