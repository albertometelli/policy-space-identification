import numpy as np
import random
import os
import sys
from gym.envs.toy_text import gridworld
from util.learner import *
from util.optimizer import *
from util.policy_boltzmann import *
from util.util_gridworld import *

from prettytable import PrettyTable

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)
SAVE_STATE_IMAGES = False

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

CONFIGURATION_STEPS = 150

MAX_NUM_TRIALS = 3
N = int(sys.argv[1])  # number of episodes collected for the LR test and the configuration

seed = int(sys.argv[2])

N_test = 1000

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

	gamma = 0.99
	if SAVE_STATE_IMAGES:
		saveStateImage("stateImage_"+str(experiment_i)+"_0A.png",mdp,sfTestMask)

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

	super_policy_initial_model = BoltzmannPolicy(nStateFeatures=17,nActions=4)
	super_learner_initial_model = GpomdpLearner(mdp,super_policy_initial_model,gamma=0.98)

	eps_initial_model = collect_gridworld_episodes(mdp,agent_policy_initial_model,N,mdp.horizon,stateFeaturesMask=sfMask,showProgress=True,exportAllStateFeatures=True)

	# Test parameters
	lr_lambda = lrTest(eps_initial_model,sfTestMask,lr=ML_LEARNING_RATE,maxSteps=ML_MAX_STEPS)
	print("REAL AGENT MASK\n",sfMask,flush=True)
	print("ESTIMATED AGENT MASK\n",sfTestMask,flush=True)
	print("LR_LAMBDA\n",lr_lambda,flush=True)
	if SAVE_STATE_IMAGES:
		saveStateImage("stateImage_"+str(experiment_i)+"_0B.png",mdp,sfTestMask)


	#
	# Cycle ENVIRONMENT CONFIGURATION
	#
	model = None
	eps = None
	super_learner = None
	for conf_index in range(1,10000):
		# Choose next parameter for model optimization
		sfTestMaskIndices = np.where(sfTestMask == False)
		nextIndex = -1
		if sfTestMaskIndices[0].size == 0:
			print("Rejected every feature. End of the experiment.",flush=True)
			break
		for i in sfTestMaskIndices[0]:
			if sfTestTrials[i] < MAX_NUM_TRIALS:
				nextIndex = i
				if(sfTestTrials[i]==0):
					model = initialModel.copy()
					eps = eps_initial_model
					super_learner = super_learner_initial_model
				sfTestTrials[i] += 1
				break
		if nextIndex == -1:
			print("Tested every not rejected feature",MAX_NUM_TRIALS,"times. End of the experiment.",flush=True)
			break
		sfGradientMask = np.zeros(shape=17,dtype=np.bool)
		sfGradientMask[nextIndex] = True
		print("Iteration",conf_index,"\nConfiguring model to test parameter",nextIndex,flush=True)

		model2 = model.copy()
		model2[0,0] += 0.01
		model2[1,1] -= 0.01
		model2[2,2] += 0.01
		model2[3,3] -= 0.01

		modelOptimizer = AdamOptimizer(shape=(4,5),learning_rate=0.01)
		for _i in range(CONFIGURATION_STEPS):
			modelGradient = getModelGradient(super_learner,eps,sfGradientMask,model,model2)
			model2 += modelOptimizer.step(modelGradient)

		model = model2.copy()
		print("Using MDP =\n",model,flush=True)

		mdp = gridworld.GridworldEnv(model)
		mdp.horizon = MDP_HORIZON
		if SAVE_STATE_IMAGES:
			saveStateImage("stateImage_"+str(experiment_i)+"_"+str(conf_index)+"A.png",mdp,sfTestMask)
		
		agent_policy = BoltzmannPolicy(np.count_nonzero(sfMask),4)
		agent_learner = GpomdpLearner(mdp,agent_policy,gamma=0.98)

		learn(
			learner=agent_learner,
			steps=LEARNING_STEPS,
			nEpisodes=LEARNING_EPISODES,
			sfmask=sfMask,
			loadFile=None,
			saveFile=None,
			autosave=True,
			plotGradient=False,
			printInfo=False
		)

		super_policy = BoltzmannPolicy(nStateFeatures=17,nActions=4)
		super_learner = GpomdpLearner(mdp,super_policy,gamma=0.98)

		eps = collect_gridworld_episodes(mdp,agent_policy,N,mdp.horizon,stateFeaturesMask=sfMask,showProgress=True,exportAllStateFeatures=True)

		# Test parameters
		sfTestMask_single = np.ones(shape=sfTestMask.size,dtype=np.bool)
		sfTestMask_single[nextIndex] = False
		lr_lambda = lrTest(eps,sfTestMask_single,lr=ML_LEARNING_RATE,maxSteps=ML_MAX_STEPS)
		sfTestMask[nextIndex] = sfTestMask_single[nextIndex]
		print("Agent feature",nextIndex,"present:",sfMask[nextIndex],flush=True)
		print("Estimated:",sfTestMask[nextIndex],flush=True)
		print("Lr lambda =",lr_lambda[nextIndex],flush=True)
		if SAVE_STATE_IMAGES:
			saveStateImage("stateImage_"+str(experiment_i)+"_"+str(conf_index)+"B.png",mdp,sfTestMask)

	eps_test_model = collect_gridworld_episodes(mdp, agent_policy_initial_model, N_test, mdp.horizon,
												   stateFeaturesMask=sfMask, showProgress=True,
												   exportAllStateFeatures=True)

	stateFeatures = [eps_test_model["s"][ep_n][0:ep_len] for ep_n, ep_len in enumerate(eps_test_model["len"])]
	stateFeatures = np.vstack(stateFeatures).T

	feature_expectations = [np.sum(eps_test_model["s"][ep_n][0:ep_len] * gamma ** np.arange(ep_len)[:, None], axis=0) for ep_n, ep_len in enumerate(eps_test_model["len"])]
	feature_expectations = np.mean(feature_expectations, axis=0)

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

	# ML with estimated mask
	setToZeroMask = np.zeros(17, dtype=np.bool)
	for i in range(16):
		if not sfTestMask[i]:
			setToZeroMask[i] = True

	super_policy = BoltzmannPolicy(17, 4)
	optimizer = AdamOptimizer(super_policy.paramsShape, learning_rate=ML_LEARNING_RATE)
	test_params = super_policy.estimate_params(eps_initial_model, optimizer, setToZero=setToZeroMask, epsilon=0.001,
											   minSteps=100,
											   maxSteps=1000, printInfo=False)

	test_norm_diff = np.linalg.norm(test_params - true_params)
	norms.append(test_norm_diff)
	print('test_conf_norm_diff \t %s' % test_norm_diff)
	test_kl = agent_policy_true.kl_divergence(stateFeatures, super_policy)
	kls.append(test_kl)
	print('test_conf_kl \t %s' % test_kl)

	eps_estMask_model = collect_gridworld_episodes(mdp, super_policy, N_test, mdp.horizon,
												   stateFeaturesMask=None, showProgress=False,
												   exportAllStateFeatures=True)

	feature_expectations_estMask = [
		np.sum(eps_estMask_model["s"][ep_n][0:ep_len] * gamma ** np.arange(ep_len)[:, None], axis=0)
		for ep_n, ep_len in enumerate(eps_estMask_model["len"])]
	feature_expectations_estMask = np.mean(feature_expectations_estMask, axis=0)

	fe_diff_mse = np.linalg.norm(feature_expectations - feature_expectations_estMask)
	print('ML EstConf Mask', test_norm_diff, test_kl, fe_diff_mse)

	fes.append(fe_diff_mse)
	table.add_row(['ML EstConf Mask', test_norm_diff, test_kl, fe_diff_mse])

	res = np.array([norms, kls, fes])
	np.save('res_conf_%s_%s_%s' % (N, time.time(), os.getpid()), res)
