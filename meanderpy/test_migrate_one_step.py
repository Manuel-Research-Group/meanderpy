import meanderpy.meanderpy as mp
import meanderpy.meanderpyp as mpp
import meanderpy.cases as cases

import numpy as np
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt

def test_sine_no_slope():
	self = cases.ChannelSineNoSlope()
	ONES = np.ones(len(self.x))
	
	test = mpp.Channel(self.x,self.y,self.z,self.W*ONES, self.D*ONES)
	base_x, base_y = mp.migrate_one_step(self.x,self.y,self.z,self.W,self.kl,self.dt,1,self.Cf,self.D,0,0,-1.0,2.5)
	test.migrate(self.Cf,self.kl,self.dt)

	assert_almost_equal(base_x, test.x)
	assert_almost_equal(base_y, test.y)

def test_sine_constant_slope():
	self = cases.ChannelSineConstantSlope()
	ONES = np.ones(len(self.x))
	
	test = mpp.Channel(self.x,self.y,self.z,self.W*ONES, self.D*ONES)
	base_x, base_y = mp.migrate_one_step(self.x,self.y,self.z,self.W,self.kl,self.dt,1,self.Cf,self.D,0,0,-1.0,2.5)
	test.migrate(self.Cf,self.kl,self.dt)

	assert_almost_equal(base_x, test.x)
	assert_almost_equal(base_y, test.y)


def test_scattered_sine_no_slope():
	self = cases.ChannelScatteredSineNoSlope()
	ONES = np.ones(len(self.x))
	
	test = mpp.Channel(self.x,self.y,self.z,self.W*ONES, self.D*ONES)
	base_x, base_y = mp.migrate_one_step(self.x,self.y,self.z,self.W,self.kl,self.dt,1,self.Cf,self.D,0,0,-1.0,2.5)
	test.migrate(self.Cf,self.kl,self.dt)

	assert_almost_equal(base_x, test.x)
	assert_almost_equal(base_y, test.y)


def test_scattered_sine_constant_slope():
	self = cases.ChannelScatteredSineConstantSlope()
	ONES = np.ones(len(self.x))
	
	test = mpp.Channel(self.x,self.y,self.z,self.W*ONES, self.D*ONES)
	base_x, base_y = mp.migrate_one_step(self.x,self.y,self.z,self.W,self.kl,self.dt,1,self.Cf,self.D,0,0,-1.0,2.5)
	test.migrate(self.Cf,self.kl,self.dt)

	assert_almost_equal(base_x, test.x)
	assert_almost_equal(base_y, test.y)

