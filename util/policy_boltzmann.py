import numpy as np
from util.policy import Policy


class BoltzmannPolicy(Policy):

	"""
	Boltzmann policy

	Parameters are in a matrix (actions) x (state_features)
	"""

	def __init__(self, nStateFeatures, nActions, paramFillValue=0.1):

		super().__init__()

		self.nActions = nActions
		self.nStateFeatures = nStateFeatures

		self.nParams = nActions * nStateFeatures

		self.paramsShape = (self.nActions,self.nStateFeatures)
		self.params = np.zeros(self.paramsShape)
	

	def compute_policy(self, stateFeatures):

		assert(len(stateFeatures)==self.nStateFeatures)

		terms = np.exp(np.dot(self.params,stateFeatures))
		sum_terms = np.sum(terms)
		
		return terms/sum_terms
	

	def draw_action(self, stateFeatures):
		policy = self.compute_policy(stateFeatures)
		#print(policy.max())
		return np.random.choice(self.nActions, p=policy)


	def compute_log_gradient(self, stateFeatures, action):

		"""
		Compute the gradient of the log of the policy function wrt to the policy params
		"""

		prob = np.exp(np.dot(stateFeatures, self.params.T))
		prob = prob / np.sum(prob, axis=1)[:, None]
		mean = prob[:, :, None] * stateFeatures[:, None, :]
		log_gradient = -mean
		row_index = np.arange(stateFeatures.shape[0], dtype=np.int)
		log_gradient[row_index, action] = log_gradient[row_index, action] + stateFeatures

		return log_gradient
	

	def compute_log(self, stateFeatures, action):
		terms = np.dot(stateFeatures,self.params.T) # shape=(T,nA)
		log = terms[np.arange(terms.shape[0]),action] # phi.T * theta
		terms = np.exp(terms)
		a_sum_terms = np.sum(terms,axis=1)
		log -= np.log(a_sum_terms)
		return np.sum(log)

	def compute_log_prob(self, stateFeatures, action):
		all_prob = np.exp(np.dot(stateFeatures, self.params.T))
		all_prob = all_prob / np.sum(all_prob, axis=1)[:, None]
		row_index = np.arange(stateFeatures.shape[0], dtype=np.int)
		prob = all_prob[row_index, action]
		return np.log(prob)

	def kl_divergence(self, stateFeatures, other):
		this_prob = self.compute_policy(stateFeatures)
		other_prob = other.compute_policy(stateFeatures)

		inner = this_prob * np.log(1e-24 + this_prob / (other_prob + 1e-24))
		return np.sum(inner) / stateFeatures.shape[1]


	def estimate_params(self, data, optimizer, params=None, setToZero=None, epsilon=0.01, minSteps=50, maxSteps=0,
						printInfo=True, lambda_ridge=0., lambda_lasso=0., lambda_shannon=0., lambda_tsallis=0.):

		"""
		Estimate the parameters of the policy with Maximum Likelihood given a set
		of trajectories.

		Return when the values stops improving, i.e. ||new_params-params||<epsilon
		"""

		if params is not None:
			self.params = params
		else:
			self.params = np.zeros(shape=self.paramsShape)

		if setToZero is not None:
			self.params[:,setToZero] = 0
		
		flag = True
		steps = 0

		n_episodes = len(data["len"])

		while flag:
		
			grad = np.zeros(shape=self.paramsShape, dtype=np.float32)

			grad_log_probs = []
			log_probs = []
			for ep_n,ep_len in enumerate(data["len"]):
				log_prob = self.compute_log_prob(data["s"][ep_n][0:ep_len],data["a"][ep_n][0:ep_len])
				log_probs.append(log_prob)
				grad_log_prob = self.compute_log_gradient(data["s"][ep_n][0:ep_len],data["a"][ep_n][0:ep_len])
				grad_log_probs.append(grad_log_prob)

			log_probs = np.concatenate(log_probs, axis=0)
			grad_log_probs = np.concatenate(grad_log_probs, axis=0)

			iw = np.exp(log_probs) / np.sum(np.exp(log_probs)) * len(log_probs)
			grad = np.sum(grad_log_probs, axis=0)
			
			if setToZero is not None:
				grad[:,setToZero] = 0

			if lambda_ridge > 0:
				grad = grad - n_episodes * lambda_ridge * self.params
			if lambda_lasso > 0:
				grad = grad - n_episodes * lambda_lasso * np.sign(self.params)
			if lambda_shannon > 0:
				grad_reg = np.sum(iw[:, None, None] * (log_probs[:, None, None] + 1) * grad_log_probs, axis=0)
				grad = grad - lambda_shannon * grad_reg
			if lambda_tsallis > 0:
				grad_reg = np.sum(iw[:, None, None] * np.exp(log_probs[:, None, None]) * grad_log_probs, axis=0)
				grad = grad - lambda_tsallis * grad_reg

			update_step = optimizer.step(grad)
			self.params = self.params + update_step
			
			update_size = np.linalg.norm(np.ravel(np.asarray(update_step)),2)
			if printInfo:
				print(steps," - Update size :",update_size)
			steps += 1
			if update_size<epsilon or steps>maxSteps:
				flag = False
			if steps<minSteps:
				flag = True
		
		if setToZero is not None:
			self.params[:,setToZero] = 0

		return self.params

	def estimate_params_mceirl(self, data, optimizer, gamma=1, params=None, setToZero=None, epsilon=0.01, minSteps=50, maxSteps=0,
						alpha_shannon=0., alpha_tsallis=0.1, printInfo=True):

		"""
		Estimate the parameters of the policy with Maximum Likelihood given a set
		of trajectories.

		Return when the values stops improving, i.e. ||new_params-params||<epsilon
		"""

		if params is not None:
			self.params = params
		else:
			self.params = np.zeros(shape=self.paramsShape)

		if setToZero is not None:
			self.params[:, setToZero] = 0

		flag = True
		steps = 0

		nEpisodes = len(data["len"])
		eps_s = data["s"]
		eps_a = data["a"]
		eps_len = data["len"]

		# Compute all the log-gradients
		nEpisodes = len(eps_len)
		maxEpLength = max(eps_len)
		nFeatures = self.nStateFeatures * self.nActions

		feats = np.zeros(shape=(nEpisodes, maxEpLength, nFeatures), dtype=np.float32)
		for n, T in enumerate(eps_len):
			eps_f = np.zeros((T, self.nActions, self.nStateFeatures))
			eps_f[:, eps_a[n, :T], :] = eps_s[n, :T]
			feats[n, :T, :] = np.reshape(eps_f, (T, nFeatures))

		i = 0
		while flag:

			#print(i)
			i += 1

			grads_log_pi = np.zeros(shape=(nEpisodes, maxEpLength, self.nParams), dtype=np.float32)
			cum_log_probs = np.zeros(shape=(nEpisodes, maxEpLength), dtype=np.float32)
			shannon_e = np.zeros(shape=(nEpisodes, maxEpLength), dtype=np.float32)
			tsallis_e = np.zeros(shape=(nEpisodes, maxEpLength), dtype=np.float32)
			pi = np.zeros(shape=(nEpisodes, maxEpLength), dtype=np.float32)

			for n, T in enumerate(eps_len):
				log_prob = self.compute_log_prob(eps_s[n, :T], eps_a[n, :T])
				prob = np.exp(log_prob)
				cum_log_probs[n, :T] = np.cumsum(log_prob)

				g = self.compute_log_gradient(eps_s[n, :T], eps_a[n, :T])
				flat_g = np.reshape(g, (T, -1))
				grads_log_pi[n, :T, :] = flat_g

				shannon_e[n, :T] = log_prob
				tsallis_e[n, :T] = .5 * (1 - prob)
				pi[n, :T] = prob

			# Importance Sampling
			iw = np.exp(cum_log_probs) / np.sum(np.exp(cum_log_probs), axis=0)[None, :]		# (nEpisodes, maxEpLength)
			#print('iw', iw.shape)

			# Features
			disc_feat_target = feats * gamma ** np.arange(maxEpLength)[None, :, None] * iw[:, :, None] * nEpisodes # (nEpisodes, maxEpLength, nFeatures)
			disc_feat_behavioral = feats * gamma ** np.arange(maxEpLength)[None, :, None]  # (nEpisodes, maxEpLength, nFeatures)
			#print(np.mean(iw * nEpisodes), np.max(iw * nEpisodes), np.min(iw * nEpisodes))
			#print('disc_feat_target', disc_feat_target.shape)
			#print('disc_feat_behavioral', disc_feat_behavioral.shape)

			disc_feat_diff = np.mean(np.sum(disc_feat_target - disc_feat_behavioral, axis=1), axis=0) # (nFeatures)
			#print(np.linalg.norm(disc_feat_diff))

			grads_log_pi_cum_dim = np.cumsum(grads_log_pi, axis=1)[:, :, :, None]
			disc_feat_behavioral_dim = disc_feat_behavioral[:, :, None, :]
			iw_dim = iw[:, :, None, None]

			# Compute the baseline
			num = np.sum(iw_dim * grads_log_pi_cum_dim * grads_log_pi_cum_dim * disc_feat_behavioral_dim, axis=0)	# (maxEpLength, nParams, nFeatures)
			den = np.sum(iw_dim * grads_log_pi_cum_dim * grads_log_pi_cum_dim, axis=0) + 1e-24 	# (maxEpLength, nParams, nFeatures)
			b = num / den 	# (maxEpLength, nParams, nFeatures)
			b_dim = b[None]


			#b_dim = 0.



			# Compute the gradient of feature diff
			grads_linear = iw_dim * grads_log_pi_cum_dim * (disc_feat_behavioral_dim - b_dim) 	# (nEpisodes, maxEpLength, nParams, nFeatures)
			gradient_ep = np.sum(grads_linear, axis=1)	# (nEpisodes, nParams, nFeatures)
			gradient_fe = np.mean(gradient_ep, axis=0)	# (nParams, nFeatures)

			gradient_diff_fe = np.dot(gradient_fe, disc_feat_diff) # (nParams)
			gradient_diff_fe = np.reshape(gradient_diff_fe, newshape=self.paramsShape)

			total_gradient = -gradient_diff_fe
			#print(np.linalg.norm(disc_feat_diff))
			#print('total-gradient', total_gradient.shape)

			if alpha_shannon > 0.:
				disc_shannon_e = shannon_e * gamma ** np.arange(maxEpLength)[None, :]	# (nEpisodes, maxEpLength)
				disc_shannon_e_dim = disc_shannon_e[:, :, None, None]  # (nEpisodes, maxEpLength, 1, 1)
				#print('disc_shannon_e_dim', disc_shannon_e_dim.shape)

				disc_grad = grads_log_pi * gamma ** np.arange(maxEpLength)[None, :, None]	# (nEpisodes, maxEpLength, nParams)
				disc_grad_dim = disc_grad[:, :, :, None] # (nEpisodes, maxEpLength, nParams, 1)
				#print('disc_grad_dim', disc_grad_dim.shape)

				# Compute the baseline
				num = np.sum(iw_dim * grads_log_pi_cum_dim * grads_log_pi_cum_dim * disc_shannon_e_dim, axis=0)  # (maxEpLength, nParams, 1)
				den = np.sum(iw_dim * grads_log_pi_cum_dim * grads_log_pi_cum_dim, axis=0) + 1e-24  # (maxEpLength, nParams, 1)
				b = num / den  # (maxEpLength, nParams, 1)
				#print('b', b.shape)

				# Compute the gradient of feature diff
				grads_linear = iw_dim * grads_log_pi_cum_dim * (disc_shannon_e_dim - b[None]) + iw_dim * disc_grad_dim # (nEpisodes, maxEpLength, nParams, 1)
				gradient_ep = np.sum(grads_linear, axis=1)  # (nEpisodes, nParams, 1)
				gradient_shannon = np.mean(gradient_ep, axis=0)  # (nParams, 1)
				gradient_shannon = np.reshape(gradient_shannon, newshape=self.paramsShape)

				total_gradient += alpha_shannon * gradient_shannon

			if alpha_tsallis > 0.:
				disc_tsallis_e = tsallis_e * gamma ** np.arange(maxEpLength)[None, :]  # (nEpisodes, maxEpLength)
				disc_tsallis_e_dim = disc_tsallis_e[:, :, None, None]  # (nEpisodes, maxEpLength, 1, 1)

				disc_pi_grad = pi[:, :, None] * grads_log_pi * gamma ** np.arange(maxEpLength)[None, :, None]  # (nEpisodes, maxEpLength, nParams)
				disc_pi_grad_dim = disc_pi_grad[:, :, :, None]  # (nEpisodes, maxEpLength, nParams, 1)

				# Compute the baseline
				num = np.sum(iw_dim * grads_log_pi_cum_dim * grads_log_pi_cum_dim * disc_tsallis_e_dim, axis=0)  # (maxEpLength, nParams, 1)
				den = np.sum(iw_dim * grads_log_pi_cum_dim * grads_log_pi_cum_dim, axis=0) + 1e-24  # (maxEpLength, nParams, 1)
				b = num / den  # (maxEpLength, nParams, 1)

				# Compute the gradient of feature diff
				grads_linear = iw_dim * grads_log_pi_cum_dim * (disc_tsallis_e_dim - b[None]) + iw_dim * disc_pi_grad_dim  # (nEpisodes, maxEpLength, nParams, 1)
				gradient_ep = np.sum(grads_linear, axis=1)  # (nEpisodes, nParams, 1)
				gradient_tsallis = np.mean(gradient_ep, axis=0)  # (nParams, 1)
				gradient_tsallis = np.reshape(gradient_tsallis, newshape=self.paramsShape)

				total_gradient += alpha_tsallis * gradient_tsallis

			update_step = optimizer.step(total_gradient)
			self.params = self.params + update_step

			update_size = np.linalg.norm(np.ravel(np.asarray(update_step)), 2)
			if printInfo:
				print(steps, " - Update size :", update_size)
			steps += 1
			if update_size < epsilon or steps > maxSteps:
				flag = False
			if steps < minSteps:
				flag = True

		if setToZero is not None:
			self.params[:, setToZero] = 0

		return self.params
	

	def getLogLikelihood(self, data, params=None):

		if params is not None:
			self.params = params
		
		eps_s = data["s"]
		eps_a = data["a"]
		eps_len = data["len"]

		log_likelihood = 0

		for n,T in enumerate(eps_len):
			sf_episode = eps_s[n]
			a_episode = eps_a[n]

			log_likelihood += self.compute_log(sf_episode,a_episode)
		
		return log_likelihood


	def getAnalyticalFisherInformation(self, data):
		
		eps_s = data["s"]
		eps_len = data["len"]

		fisherInformation = np.zeros(shape=(self.nParams,self.nParams),dtype=np.float32)

		for n,T in enumerate(eps_len):
			for t in range(T):
				sf = eps_s[n,t]
				policy = self.compute_policy(sf)

				x1 = np.zeros(shape=(self.nParams,self.nParams))
				for a in range(self.nActions):
					sa = np.zeros(shape=self.nParams)
					sa[a*self.nStateFeatures:(a+1)*self.nStateFeatures] = sf
					x1 += policy[a]*np.outer(sa,sa)

				x2_vec = np.multiply(np.tile(sf,(self.nActions,1)).T,policy).T
				x2_vec = np.ravel(x2_vec)
				x2 = np.outer(x2_vec,x2_vec)

				fisherInformation += x1-x2

		fisherInformation /= np.sum(eps_len)

		return fisherInformation