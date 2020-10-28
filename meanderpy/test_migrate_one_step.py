import meanderpy.meanderpy as mp
import meanderpy.meanderpyp as mpp
import meanderpy.cases as cases

import numpy as np
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt

def test_sine_no_slope():
	self = cases.ChannelSineNoSlope()

	N = len(self.x)
	base_x, base_y = mp.migrate_one_step(self.x,self.y,self.z,self.W,self.kl,self.dt,1,self.Cf,self.D,0,0,-1.0,2.5)
	test_x, test_y = mpp.migrate_one_step(self.x,self.y,self.z,self.W*np.ones(N), self.D*np.ones(N), self.Cf,self.kl,self.dt)

	assert_almost_equal(base_x, test_x)
	assert_almost_equal(base_y, test_y)

def test_sine_constant_slope():
	self = cases.ChannelSineConstantSlope()

	N = len(self.x)
	base_x, base_y = mp.migrate_one_step(self.x,self.y,self.z,self.W,self.kl,self.dt,1,self.Cf,self.D,0,0,-1.0,2.5)
	test_x, test_y = mpp.migrate_one_step(self.x,self.y,self.z,self.W*np.ones(N), self.D*np.ones(N), self.Cf,self.kl,self.dt)

	assert_almost_equal(base_x, test_x)
	assert_almost_equal(base_y, test_y)

def test_scattered_sine_no_slope():
	self = cases.ChannelScatteredSineNoSlope()

	N = len(self.x)
	base_x, base_y = mp.migrate_one_step(self.x,self.y,self.z,self.W,self.kl,self.dt,1,self.Cf,self.D,0,0,-1.0,2.5)
	test_x, test_y = mpp.migrate_one_step(self.x,self.y,self.z,self.W*np.ones(N), self.D*np.ones(N), self.Cf,self.kl,self.dt)

	assert_almost_equal(base_x, test_x)
	assert_almost_equal(base_y, test_y)

def test_scattered_sine_constant_slope():
	self = cases.ChannelScatteredSineConstantSlope()

	N = len(self.x)
	base_x, base_y = mp.migrate_one_step(self.x,self.y,self.z,self.W,self.kl,self.dt,1,self.Cf,self.D,0,0,-1.0,2.5)
	test_x, test_y = mpp.migrate_one_step(self.x,self.y,self.z,self.W*np.ones(N), self.D*np.ones(N), self.Cf,self.kl,self.dt)

	assert_almost_equal(base_x, test_x)
	assert_almost_equal(base_y, test_y)