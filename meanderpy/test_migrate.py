import meanderpy.meanderpy as mp
import meanderpy.meanderpyp as mpp
import meanderpy.cases as cases

import numpy as np
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt

def test_sine_no_slope():
	self = cases.ChannelSineNoSlope()

	ONES = np.ones(len(self.x))
	base_ch = mp.Channel(self.x.copy(), self.y.copy(), self.z.copy(), self.W, self.D)
	test_ch = mpp.Channel(self.x.copy(), self.y.copy(), self.z.copy(), self.W * ONES, self.D * ONES)

	base_ch_belt = mp.ChannelBelt(channels=[base_ch],cutoffs=[],cl_times=[0.0],cutoff_times=[])
	test_ch_belt = mpp.ChannelBelt(test_ch)

	base_ch_belt.migrate(self.nit,self.saved_ts,self.ds,self.pad,self.crdist,self.Cf,self.kl,self.kv,self.dt,self.density,self.t1,self.t2,self.t3,self.aggr_factor)
	test_ch_belt.migrate(self.nit,self.saved_ts,self.ds,self.pad,self.crdist,self.Cf,self.kl,self.kv,self.dt,self.density,self.t1,self.t2,self.t3,self.aggr_factor)

	assert_almost_equal(base_ch_belt.channels[-1].x, test_ch_belt.channels[-1].x)
	assert_almost_equal(base_ch_belt.channels[-1].y, test_ch_belt.channels[-1].y)
	assert_almost_equal(base_ch_belt.channels[-1].z, test_ch_belt.channels[-1].z)

def test_sine_constant_slope():
	self = cases.ChannelSineConstantSlope()

	ONES = np.ones(len(self.x))
	base_ch = mp.Channel(self.x.copy(), self.y.copy(), self.z.copy(), self.W, self.D)
	test_ch = mpp.Channel(self.x.copy(), self.y.copy(), self.z.copy(), self.W * ONES, self.D * ONES)

	base_ch_belt = mp.ChannelBelt(channels=[base_ch],cutoffs=[],cl_times=[0.0],cutoff_times=[])
	test_ch_belt = mpp.ChannelBelt(test_ch)

	base_ch_belt.migrate(self.nit,self.saved_ts,self.ds,self.pad,self.crdist,self.Cf,self.kl,self.kv,self.dt,self.density,self.t1,self.t2,self.t3,self.aggr_factor)
	test_ch_belt.migrate(self.nit,self.saved_ts,self.ds,self.pad,self.crdist,self.Cf,self.kl,self.kv,self.dt,self.density,self.t1,self.t2,self.t3,self.aggr_factor)

	assert_almost_equal(base_ch_belt.channels[-1].x, test_ch_belt.channels[-1].x)
	assert_almost_equal(base_ch_belt.channels[-1].y, test_ch_belt.channels[-1].y)
	assert_almost_equal(base_ch_belt.channels[-1].z, test_ch_belt.channels[-1].z)

def test_scattered_sine_no_slope():
	self = cases.ChannelScatteredSineNoSlope()

	ONES = np.ones(len(self.x))
	base_ch = mp.Channel(self.x.copy(), self.y.copy(), self.z.copy(), self.W, self.D)
	test_ch = mpp.Channel(self.x.copy(), self.y.copy(), self.z.copy(), self.W * ONES, self.D * ONES)

	base_ch_belt = mp.ChannelBelt(channels=[base_ch],cutoffs=[],cl_times=[0.0],cutoff_times=[])
	test_ch_belt = mpp.ChannelBelt(test_ch)

	base_ch_belt.migrate(self.nit,self.saved_ts,self.ds,self.pad,self.crdist,self.Cf,self.kl,self.kv,self.dt,self.density,self.t1,self.t2,self.t3,self.aggr_factor)
	test_ch_belt.migrate(self.nit,self.saved_ts,self.ds,self.pad,self.crdist,self.Cf,self.kl,self.kv,self.dt,self.density,self.t1,self.t2,self.t3,self.aggr_factor)

	assert_almost_equal(base_ch_belt.channels[-1].x, test_ch_belt.channels[-1].x)
	assert_almost_equal(base_ch_belt.channels[-1].y, test_ch_belt.channels[-1].y)
	assert_almost_equal(base_ch_belt.channels[-1].z, test_ch_belt.channels[-1].z)

def test_scattered_sine_constant_slope():
	self = cases.ChannelScatteredSineConstantSlope()

	ONES = np.ones(len(self.x))
	base_ch = mp.Channel(self.x.copy(), self.y.copy(), self.z.copy(), self.W, self.D)
	test_ch = mpp.Channel(self.x.copy(), self.y.copy(), self.z.copy(), self.W * ONES, self.D * ONES)

	base_ch_belt = mp.ChannelBelt(channels=[base_ch],cutoffs=[],cl_times=[0.0],cutoff_times=[])
	test_ch_belt = mpp.ChannelBelt(test_ch)

	base_ch_belt.migrate(self.nit,self.saved_ts,self.ds,self.pad,self.crdist,self.Cf,self.kl,self.kv,self.dt,self.density,self.t1,self.t2,self.t3,self.aggr_factor)
	test_ch_belt.migrate(self.nit,self.saved_ts,self.ds,self.pad,self.crdist,self.Cf,self.kl,self.kv,self.dt,self.density,self.t1,self.t2,self.t3,self.aggr_factor)

	assert_almost_equal(base_ch_belt.channels[-1].x, test_ch_belt.channels[-1].x)
	assert_almost_equal(base_ch_belt.channels[-1].y, test_ch_belt.channels[-1].y)
	assert_almost_equal(base_ch_belt.channels[-1].z, test_ch_belt.channels[-1].z)
