n_samples = 10000
n_features = 200
n_groups = 3
prop_diff=0.4
set.seed(333)
#samples
group = rep(seq(n_groups), length.out = n_samples)
dat= (rexp(n_samples*n_features, rate = 0.1) +
        rnorm(n_samples*n_features, mean =1000, sd = 10)) *
  (rexp(n_samples*n_features, rate = 0.5) + 1)
X <- matrix(dat, nrow=n_features, ncol=n_samples,
            dimnames=list(paste0("feature_", seq_len(n_features)),
                          paste0("cell_", seq_len(n_samples))))
diff_feat1 = rbinom(n_features, 1, prop_diff)
diff_feat2 = rbinom(n_features, 1, prop_diff)

for(i in seq(n_features)){
  if(diff_feat1[i] != 0) X[i, which(group == 1)] = X[i, which(group == 1)] * (rexp(1, rate = 0.5) + 0.7)
  if(diff_feat2[i] != 0) X[i, which(group == 3)] = X[i, which(group == 3)] * (rexp(1, rate = 0.5) + 0.7)
}
zero_inflated_locs = sample(1:length(X), size = floor(0.3 * length(X)), replace = FALSE)
X[zero_inflated_locs] = 0

# setup_py_env(method = "conda")
# start_FastPCA_env()
