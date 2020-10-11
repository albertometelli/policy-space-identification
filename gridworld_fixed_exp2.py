import numpy as np
import sys
import os
from gym.envs.toy_text import gridworld
from util.learner import *
from util.optimizer import *
from util.policy_boltzmann import *
from util.util_gridworld import *

from prettytable import PrettyTable
import random

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

def set_all_seeds(seed, env):
	random.seed(seed)
	np.random.seed(seed)
	env.seed(seed)


NUM_EXPERIMENTS = 1
type1err_tot = 0
type2err_tot = 0

MDP_HORIZON = 50
LEARNING_STEPS = 200
LEARNING_EPISODES = 250

ML_MAX_STEPS = 1000
ML_LEARNING_RATE = 0.03

N = int(sys.argv[1]) # number of episodes collected for the LR test and the configuration

seed = int(sys.argv[2])

N_test = 1000

print("N = %s" % N)

table = PrettyTable()
table.field_names = ["Method", "Norm Theta Diff", "KL Div", "Feature Ex Diff"]

for experiment_i in range(NUM_EXPERIMENTS):

	table.clear_rows()

	print("\nExperiment",experiment_i,flush=True)
	w_row = np.ones(5,dtype=np.float32)
	w_row[np.random.choice(5)] *= 5
	w_col = np.ones(5,dtype=np.float32)
	w_col[np.random.choice(5)] *= 5
	w_grow = np.ones(5,dtype=np.float32)
	w_grow[np.random.choice(5)] *= 5
	w_gcol = np.ones(5,dtype=np.float32)
	w_gcol[np.random.choice(5)] *= 5
	initialModel = np.array([w_row,w_col,w_grow,w_gcol],dtype=np.float32)

	sfMask = np.random.choice(a=[False, True], size=(16), p=[0.5, 0.5])
	sfMask = np.concatenate([sfMask,[True]]) # constant term

	sfTestMask = np.zeros(shape=16,dtype=np.bool) # State features rejected (we think the agent have)
	sfTestTrials = np.zeros(shape=16,dtype=np.int32) # Num of trials for each state feature

	#
	# Initial model - First test
	#
	print("Using initial MDP =\n",initialModel,flush=True)

	mdp = gridworld.GridworldEnv(initialModel)
	mdp.horizon = MDP_HORIZON

	set_all_seeds(seed, mdp)
	
	#agent_policy_initial_model = GaussianPolicy(nStateFeatures=np.count_nonzero(sfMask),actionDim=2)
	#agent_learner_initial_model = GpomdpLearner(mdp,agent_policy_initial_model,gamma=0.98)
	agent_policy_initial_model = BoltzmannPolicy(np.count_nonzero(sfMask),4)
	agent_learner_initial_model = GpomdpLearner(mdp,agent_policy_initial_model,gamma=0.98)

	learn(
		learner=agent_learner_initial_model,
		steps=LEARNING_STEPS,
		nEpisodes=LEARNING_EPISODES,
		sfmask=sfMask,
		loadFile=None,
		saveFile=None,
		autosave=True,
		plotGradient=False,
		printInfo=False
	)

	gamma = 0.99

	super_policy_initial_model = BoltzmannPolicy(nStateFeatures=17,nActions=4)
	super_learner_initial_model = GpomdpLearner(mdp,super_policy_initial_model,gamma=0.98)

	eps_initial_model = collect_gridworld_episodes(mdp,agent_policy_initial_model,N,mdp.horizon,stateFeaturesMask=sfMask,showProgress=True,exportAllStateFeatures=True)

	eps_test_model = collect_gridworld_episodes(mdp, agent_policy_initial_model, N_test, mdp.horizon,
												   stateFeaturesMask=sfMask, showProgress=True,
												   exportAllStateFeatures=True)

	stateFeatures = [eps_test_model["s"][ep_n][0:ep_len] for ep_n, ep_len in enumerate(eps_test_model["len"])]
	stateFeatures = np.vstack(stateFeatures).T

	feature_expectations = [np.sum(eps_test_model["s"][ep_n][0:ep_len] * gamma ** np.arange(ep_len)[:, None], axis=0) for ep_n, ep_len in enumerate(eps_test_model["len"])]
	feature_expectations = np.mean(feature_expectations, axis=0)
	# Test parameters



	lr_lambda = lrTest(eps_initial_model,sfTestMask,lr=ML_LEARNING_RATE,maxSteps=ML_MAX_STEPS)

	#print("REAL AGENT MASK\n",sfMask,flush=True)
	#print("ESTIMATED AGENT MASK\n",sfTestMask,flush=True)
	#print("LR_LAMBDA\n",lr_lambda,flush=True)

	'''
	x = np.array(sfTestMask,dtype=np.int32)-np.array(sfMask[0:16],dtype=np.int32)
	type1err = np.count_nonzero(x == 1) # Rejected features the agent doesn't have
	type2err = np.count_nonzero(x == -1) # Not rejected features the agent has
	type1err_tot += type1err
	type2err_tot += type2err
	print("REAL AGENT MASK\n",sfMask[0:16],flush=True)
	print("ESTIMATED AGENT MASK\n",sfTestMask,flush=True)
	print("Type 1 error frequency (last experiment):",np.float32(type1err)/16.0)
	print("Type 2 error frequency (last experiment):",np.float32(type2err)/16.0)
	print("Type 1 error frequency [",experiment_i+1,"]:",np.float32(type1err_tot)/16.0/np.float32(experiment_i+1))
	print("Type 2 error frequency [",experiment_i+1,"]:",np.float32(type2err_tot)/16.0/np.float32(experiment_i+1))
	'''

	norms, kls, fes = [], [], []

	j = 0
	true_params = np.zeros((4, 17))
	for i in range(16):
		if sfMask[i]:
			true_params[:, i] = agent_policy_initial_model.params[:, j]
			j += 1
	true_params[:, -1] = agent_policy_initial_model.params[:, -1]
	agent_policy_true = BoltzmannPolicy(17, 4)
	agent_policy_true.params = true_params

	# ML with TRUE mask
	setToZeroMask = np.zeros(17, dtype=np.bool)
	for i in range(16):
		if not sfMask[i]:
			setToZeroMask[i] = True

	super_policy = BoltzmannPolicy(17, 4)
	optimizer = AdamOptimizer(super_policy.paramsShape, learning_rate=ML_LEARNING_RATE)
	test_params = super_policy.estimate_params(eps_initial_model, optimizer, setToZero=setToZeroMask, epsilon=0.001,
											   minSteps=100,
											   maxSteps=1000, printInfo=False)

	test_norm_diff = np.linalg.norm(test_params - true_params)
	norms.append(test_norm_diff)
	test_kl = agent_policy_true.kl_divergence(stateFeatures, super_policy)
	kls.append(test_kl)

	eps_estMask_model = collect_gridworld_episodes(mdp, super_policy, N_test, mdp.horizon,
												   stateFeaturesMask=None, showProgress=False,
												   exportAllStateFeatures=True)

	feature_expectations_estMask = [
		np.sum(eps_estMask_model["s"][ep_n][0:ep_len] * gamma ** np.arange(ep_len)[:, None], axis=0)
		for ep_n, ep_len in enumerate(eps_estMask_model["len"])]
	feature_expectations_estMask = np.mean(feature_expectations_estMask, axis=0)

	fe_diff_mse = np.linalg.norm(feature_expectations - feature_expectations_estMask)
	fes.append(fe_diff_mse)
	table.add_row(['ML True Mask', test_norm_diff, test_kl, fe_diff_mse])

	#ML with estimated mask
	setToZeroMask = np.zeros(17, dtype=np.bool)
	for i in range(16):
		if not sfTestMask[i]:
			setToZeroMask[i] = True

	super_policy = BoltzmannPolicy(17, 4)
	optimizer = AdamOptimizer(super_policy.paramsShape, learning_rate=ML_LEARNING_RATE)
	test_params = super_policy.estimate_params(eps_initial_model, optimizer, setToZero=setToZeroMask, epsilon=0.001, minSteps=100,
											 maxSteps=1000, printInfo=False)

	test_norm_diff = np.linalg.norm(test_params - true_params)
	norms.append(test_norm_diff)
	test_kl = agent_policy_true.kl_divergence(stateFeatures, super_policy)
	kls.append(test_kl)

	eps_estMask_model = collect_gridworld_episodes(mdp, super_policy, N_test, mdp.horizon,
											   stateFeaturesMask=None, showProgress=False,
											   exportAllStateFeatures=True)

	feature_expectations_estMask = [np.sum(eps_estMask_model["s"][ep_n][0:ep_len] * gamma ** np.arange(ep_len)[:, None], axis=0)
								for ep_n, ep_len in enumerate(eps_estMask_model["len"])]
	feature_expectations_estMask = np.mean(feature_expectations_estMask, axis=0)

	fe_diff_mse = np.linalg.norm(feature_expectations - feature_expectations_estMask)
	fes.append(fe_diff_mse)
	table.add_row(['ML Est Mask', test_norm_diff, test_kl, fe_diff_mse])

	# ML
	super_policy = BoltzmannPolicy(17, 4)
	optimizer = AdamOptimizer(super_policy.paramsShape, learning_rate=ML_LEARNING_RATE)
	mse_params = super_policy.estimate_params(eps_initial_model, optimizer, setToZero=None, epsilon=0.001,
												minSteps=100, maxSteps=1000, printInfo=False, lambda_ridge=0.)

	mse_norm_diff = np.linalg.norm(mse_params - true_params)
	norms.append(mse_norm_diff)
	mse_kl = agent_policy_true.kl_divergence(stateFeatures, super_policy)
	kls.append(mse_kl)

	eps_mse_model = collect_gridworld_episodes(mdp, super_policy,N_test, mdp.horizon,
												   stateFeaturesMask=None, showProgress=False,
												   exportAllStateFeatures=True)

	feature_expectations_mse = [np.sum(eps_mse_model["s"][ep_n][0:ep_len] * gamma ** np.arange(ep_len)[:, None], axis=0)
							for ep_n, ep_len in enumerate(eps_mse_model["len"])]
	feature_expectations_mse = np.mean(feature_expectations_mse, axis=0)

	fe_diff_mse = np.linalg.norm(feature_expectations - feature_expectations_mse)
	fes.append(fe_diff_mse)
	table.add_row(['ML', mse_norm_diff, mse_kl, fe_diff_mse])


	# Ridge
	for lambda_ridge in [0.001, 0.01]: #, 0.1, 1]:
		super_policy = BoltzmannPolicy(17, 4)
		optimizer = AdamOptimizer(super_policy.paramsShape, learning_rate=ML_LEARNING_RATE)
		ridge_params = super_policy.estimate_params(eps_initial_model, optimizer, setToZero=None, epsilon=0.001,
												 minSteps=100, maxSteps=1000, printInfo=False, lambda_ridge=lambda_ridge)

		ridge_norm_diff = np.linalg.norm(ridge_params - true_params)
		norms.append(ridge_norm_diff)
		ridge_kl = agent_policy_true.kl_divergence(stateFeatures, super_policy)
		kls.append(ridge_kl)

		eps_model_ridge = collect_gridworld_episodes(mdp, super_policy, N_test, mdp.horizon,
												   stateFeaturesMask=None, showProgress=False,
												   exportAllStateFeatures=True)

		feature_expectations_ridge = [
			np.sum(eps_model_ridge["s"][ep_n][0:ep_len] * gamma ** np.arange(ep_len)[:, None], axis=0)
			for ep_n, ep_len in enumerate(eps_model_ridge["len"])]
		feature_expectations_ridge = np.mean(feature_expectations_ridge, axis=0)

		fe_diff_ridge = np.linalg.norm(feature_expectations - feature_expectations_ridge)
		fes.append(fe_diff_ridge)
		table.add_row(['ML Ridge(%s)' % lambda_ridge, ridge_norm_diff, ridge_kl, fe_diff_ridge])

	# Lasso
	for lambda_lasso in [0.001, 0.01]: #, 0.1, 1]:
		super_policy = BoltzmannPolicy(17, 4)
		optimizer = AdamOptimizer(super_policy.paramsShape, learning_rate=ML_LEARNING_RATE)
		lasso_params = super_policy.estimate_params(eps_initial_model, optimizer, setToZero=None, epsilon=0.001,
													minSteps=100, maxSteps=1000, printInfo=False,
													lambda_lasso=lambda_lasso)

		lasso_norm_diff = np.linalg.norm(lasso_params - true_params)
		norms.append(lasso_norm_diff)
		lasso_kl = agent_policy_true.kl_divergence(stateFeatures, super_policy)
		kls.append(lasso_kl)

		eps_model_lasso = collect_gridworld_episodes(mdp, super_policy, N_test, mdp.horizon,
													 stateFeaturesMask=None, showProgress=False,
													 exportAllStateFeatures=True)

		feature_expectations_lasso = [
			np.sum(eps_model_lasso["s"][ep_n][0:ep_len] * gamma ** np.arange(ep_len)[:, None], axis=0)
			for ep_n, ep_len in enumerate(eps_model_lasso["len"])]
		feature_expectations_lasso = np.mean(feature_expectations_lasso, axis=0)

		fe_diff_lasso = np.linalg.norm(feature_expectations - feature_expectations_lasso)
		fes.append(fe_diff_lasso)
		table.add_row(['ML Lasso(%s)' % lambda_lasso, lasso_norm_diff, lasso_kl, fe_diff_lasso])

	# Shannon
	for lambda_shannon in [0.1, 1, 10, 100]:
		super_policy = BoltzmannPolicy(17, 4)
		optimizer = AdamOptimizer(super_policy.paramsShape, learning_rate=ML_LEARNING_RATE)
		shannon_params = super_policy.estimate_params(eps_initial_model, optimizer, setToZero=None, epsilon=0.001,
													minSteps=100, maxSteps=1000, printInfo=False,
													lambda_shannon=lambda_shannon)

		shannon_norm_diff = np.linalg.norm(shannon_params - true_params)
		norms.append(shannon_norm_diff)
		shannon_kl = agent_policy_true.kl_divergence(stateFeatures, super_policy)
		kls.append(shannon_kl)

		eps_model_shannon = collect_gridworld_episodes(mdp, super_policy, N_test, mdp.horizon,
													 stateFeaturesMask=None, showProgress=False,
													 exportAllStateFeatures=True)

		feature_expectations_shannon = [
			np.sum(eps_model_shannon["s"][ep_n][0:ep_len] * gamma ** np.arange(ep_len)[:, None], axis=0)
			for ep_n, ep_len in enumerate(eps_model_shannon["len"])]
		feature_expectations_shannon = np.mean(feature_expectations_shannon, axis=0)

		fe_diff_shannon = np.linalg.norm(feature_expectations - feature_expectations_shannon)
		fes.append(fe_diff_shannon)
		table.add_row(['ML Shannon(%s)' % lambda_shannon, shannon_norm_diff, shannon_kl, fe_diff_shannon])

	# Tsallis
	for lambda_tsallis in [0.1, 1, 10, 100]:
		super_policy = BoltzmannPolicy(17, 4)
		optimizer = AdamOptimizer(super_policy.paramsShape, learning_rate=ML_LEARNING_RATE)
		tsallis_params = super_policy.estimate_params(eps_initial_model, optimizer, setToZero=None, epsilon=0.001,
													minSteps=100, maxSteps=1000, printInfo=False,
													lambda_tsallis=lambda_tsallis)

		tsallis_norm_diff = np.linalg.norm(tsallis_params - true_params)
		norms.append(tsallis_norm_diff)
		tsallis_kl = agent_policy_true.kl_divergence(stateFeatures, super_policy)
		kls.append(tsallis_kl)

		eps_model_tsallis = collect_gridworld_episodes(mdp, super_policy, N_test, mdp.horizon,
													 stateFeaturesMask=None, showProgress=False,
													 exportAllStateFeatures=True)

		feature_expectations_tsallis = [
			np.sum(eps_model_tsallis["s"][ep_n][0:ep_len] * gamma ** np.arange(ep_len)[:, None], axis=0)
			for ep_n, ep_len in enumerate(eps_model_tsallis["len"])]
		feature_expectations_tsallis = np.mean(feature_expectations_tsallis, axis=0)

		fe_diff_tsallis = np.linalg.norm(feature_expectations - feature_expectations_tsallis)
		fes.append(fe_diff_tsallis)
		table.add_row(['ML Tsallis(%s)' % lambda_tsallis, tsallis_norm_diff, tsallis_kl, fe_diff_tsallis])

	# FE
	super_policy = BoltzmannPolicy(17, 4)
	optimizer = AdamOptimizer(super_policy.paramsShape, learning_rate=ML_LEARNING_RATE)
	fe_params = super_policy.estimate_params_mceirl(eps_initial_model, optimizer, setToZero=None, epsilon=0.001,
											  minSteps=100, maxSteps=100, printInfo=False, params=mse_params)

	fe_norm_diff = np.linalg.norm(fe_params - true_params)
	norms.append(fe_norm_diff)
	fe_kl = agent_policy_true.kl_divergence(stateFeatures, super_policy)
	kls.append(fe_kl)

	eps_model_fe = collect_gridworld_episodes(mdp, super_policy, N_test, mdp.horizon,
												 stateFeaturesMask=None, showProgress=False,
												 exportAllStateFeatures=True)

	feature_expectations_fe = [
		np.sum(eps_model_fe["s"][ep_n][0:ep_len] * gamma ** np.arange(ep_len)[:, None], axis=0)
		for ep_n, ep_len in enumerate(eps_model_fe["len"])]
	feature_expectations_fe = np.mean(feature_expectations_fe, axis=0)

	fe_diff_fe = np.linalg.norm(feature_expectations - feature_expectations_fe)
	fes.append(fe_diff_fe)
	table.add_row(['FE', fe_norm_diff, fe_kl, fe_diff_fe])


	# FE Shannon
	for alpha_shannon in [0.001, 0.01, 0.1, 1]:
		super_policy = BoltzmannPolicy(17, 4)
		optimizer = AdamOptimizer(super_policy.paramsShape, learning_rate=ML_LEARNING_RATE)
		fe_shannon_params = super_policy.estimate_params_mceirl(eps_initial_model, optimizer, setToZero=None, epsilon=0.001,
														minSteps=100, maxSteps=200, printInfo=False, alpha_shannon=alpha_shannon, params=mse_params)

		fe_shannon_norm_diff = np.linalg.norm(fe_shannon_params - true_params)
		norms.append(fe_shannon_norm_diff)
		fe_shannon_kl = agent_policy_true.kl_divergence(stateFeatures, super_policy)
		kls.append(fe_shannon_kl)

		eps_model_fe = collect_gridworld_episodes(mdp, super_policy, N_test, mdp.horizon,
												  stateFeaturesMask=None, showProgress=False,
												  exportAllStateFeatures=True)

		feature_expectations_fe = [
			np.sum(eps_model_fe["s"][ep_n][0:ep_len] * gamma ** np.arange(ep_len)[:, None], axis=0)
			for ep_n, ep_len in enumerate(eps_model_fe["len"])]
		feature_expectations_fe = np.mean(feature_expectations_fe, axis=0)

		fe_diff_fe = np.linalg.norm(feature_expectations - feature_expectations_fe)
		fes.append(fe_diff_fe)
		table.add_row(['FE Shannon(%s)' % alpha_shannon, fe_shannon_norm_diff, fe_shannon_kl, fe_diff_fe])

	# FE Tsallis
	for alpha_tsallis in [0.001, 0.01, 0.1, 1]:
		super_policy = BoltzmannPolicy(17, 4)
		optimizer = AdamOptimizer(super_policy.paramsShape, learning_rate=ML_LEARNING_RATE)
		fe_tsallis_params = super_policy.estimate_params_mceirl(eps_initial_model, optimizer, setToZero=None, epsilon=0.001,
														minSteps=100, maxSteps=200, printInfo=False, alpha_tsallis=alpha_tsallis, params=mse_params)

		fe_tsallis_norm_diff = np.linalg.norm(fe_tsallis_params - true_params)
		norms.append(fe_tsallis_norm_diff)
		fe_tsallis_kl = agent_policy_true.kl_divergence(stateFeatures, super_policy)
		kls.append(fe_tsallis_kl)

		eps_model_fe = collect_gridworld_episodes(mdp, super_policy, N_test, mdp.horizon,
												  stateFeaturesMask=None, showProgress=False,
												  exportAllStateFeatures=True)

		feature_expectations_fe = [
			np.sum(eps_model_fe["s"][ep_n][0:ep_len] * gamma ** np.arange(ep_len)[:, None], axis=0)
			for ep_n, ep_len in enumerate(eps_model_fe["len"])]
		feature_expectations_fe = np.mean(feature_expectations_fe, axis=0)

		fe_diff_fe = np.linalg.norm(feature_expectations - feature_expectations_fe)
		fes.append(fe_diff_fe)
		table.add_row(['FE Tsallis(%s)' % alpha_tsallis, fe_tsallis_norm_diff, fe_tsallis_kl, fe_diff_fe])

	print(table)

	res = np.array([norms, kls, fes])
	np.save('res_%s_%s_%s' % (N, time.time(), os.getpid()), res)



#print("[OUTPUT] N = ",N)
#print("[OUTPUT] Type 1 error frequency:",np.float32(type1err_tot)/16.0/np.float32(NUM_EXPERIMENTS))
#print("[OUTPUT] Type 2 error frequency:",np.float32(type2err_tot)/16.0/np.float32(NUM_EXPERIMENTS))