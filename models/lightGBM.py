import lightgbm as lgb

class lightgbm():
    def __init__(self, num_leaves = 31, num_trees = 100, max_depth = 12, metric = 'binary_logloss', feature_fraction = 0.8, bagging_fraction = 0.9, lgb_num_iterations=15):
        #Define Parameters
        self.params = {'num_leaves': num_leaves,
                          'num_trees': num_trees,
                          'boosting_type': 'gbdt',
                          'objective': 'binary', #  I changed this
                          'max_depth': max_depth,
                          'verbosity': 0,
                          'task': 'train',
                          'metric': metric, # And this as well
                          "learning_rate" : 1e-2,
                          "bagging_fraction" : bagging_fraction,  # subsample
                          "bagging_freq" : 5,        # subsample_freq
                          "bagging_seed" : 341,
                          "feature_fraction" : feature_fraction,  # colsample_bytree
                          "feature_fraction_seed":341,}

        self.lgb_num_iterations = lgb_num_iterations

    def train(self,embeddings, drivers):
        #Put data into the lgb Dataset
        lgb_train = lgb.Dataset(embeddings, drivers) #Check how the data is being given
        #Train the model
        self.lgbm = lgb.train(self.params, lgb_train, self.lgb_num_iterations)
        #Plot the loss
        lgb.plot_metric(self.lgbm)

    def predict(self,embeddings):
        output = self.lgbm.predict(embeddings)
        return output