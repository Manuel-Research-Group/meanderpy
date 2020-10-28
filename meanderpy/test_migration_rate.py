import meanderpy.meanderpy as mp
import meanderpy.meanderpyp as mpp
import meanderpy.cases as cases

import numpy as np
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt

def test_sine_no_slope():
	self = cases.ChannelSineNoSlope()

	ns=len(self.x)
	curv = mp.compute_curvature(self.x, self.y)
	_, _, _, ds, s = mp.compute_derivatives(self.x,self.y,self.z)
	R0 = self.kl*self.W*curv # simple linear relationship between curvature and nominal migration rate
	alpha = 2*self.Cf/self.D # exponent for convolution function G
	
	base_R1 = mp.compute_migration_rate(0,ns,ds,alpha,-1.0,2.5,R0)
	test_R1 = mpp.compute_migration_rate(R0, self.Cf, self.D * np.ones(ns), ds, s[-1])

	assert_almost_equal(base_R1, test_R1)

def test_sine_constant_slope():
	self = cases.ChannelSineConstantSlope()

	ns=len(self.x)
	curv = mp.compute_curvature(self.x, self.y)
	_, _, _, ds, s = mp.compute_derivatives(self.x,self.y,self.z)
	R0 = self.kl*self.W*curv # simple linear relationship between curvature and nominal migration rate
	alpha = 2*self.Cf/self.D # exponent for convolution function G
	
	base_R1 = mp.compute_migration_rate(0,ns,ds,alpha,-1.0,2.5,R0)
	test_R1 = mpp.compute_migration_rate(R0, self.Cf, self.D * np.ones(ns), ds, s[-1])

	assert_almost_equal(base_R1, test_R1)

def test_scattered_sine_no_slope():
	self = cases.ChannelScatteredSineNoSlope()

	ns=len(self.x)
	curv = mp.compute_curvature(self.x, self.y)
	_, _, _, ds, s = mp.compute_derivatives(self.x,self.y,self.z)
	R0 = self.kl*self.W*curv # simple linear relationship between curvature and nominal migration rate
	alpha = 2*self.Cf/self.D # exponent for convolution function G
	
	base_R1 = mp.compute_migration_rate(0,ns,ds,alpha,-1.0,2.5,R0)
	test_R1 = mpp.compute_migration_rate(R0, self.Cf, self.D * np.ones(ns), ds, s[-1])

	assert_almost_equal(base_R1, test_R1)

def test_scattered_sine_constant_slope():
	self = cases.ChannelScatteredSineConstantSlope()

	ns=len(self.x)
	curv = mp.compute_curvature(self.x, self.y)
	_, _, _, ds, s = mp.compute_derivatives(self.x,self.y,self.z)
	R0 = self.kl*self.W*curv # simple linear relationship between curvature and nominal migration rate
	alpha = 2*self.Cf/self.D # exponent for convolution function G
	
	base_R1 = mp.compute_migration_rate(0,ns,ds,alpha,-1.0,2.5,R0)
	test_R1 = mpp.compute_migration_rate(R0, self.Cf, self.D * np.ones(ns), ds, s[-1])

	assert_almost_equal(base_R1, test_R1)
