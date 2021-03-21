import inspect

import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from tf2_dist_utils import build_loss

class GenFactMachine():

    def __init__(self, dist, feature_cards, factor_dim):
        self.dist = dist
        self.feature_cards = feature_cards
        self.factor_dim = factor_dim
        self.n_params = len(inspect.signature(dist).parameters)

    def fit(self, X, y):
        self.loss = build_loss("GFMloss", self.dist)

        self.model = self._build_model(
            X,
            build_loss("GFMloss", self.dist),
            self.n_params, 
            self.feature_cards, 
            self.factor_dim)

        hist = self.model.fit(
            X, y,
            validation_split=0.15, 
            batch_size=16,
            epochs=100,
            callbacks=[
              tfk.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ])
        
        return hist

    def predict(self, X, nsamples=None):
        params = tf.constant(self.model.predict(X))
        
        shape = params.shape

        # model.predict returns tensor of parameters
        # dist objects are assumed to take the individual
        # columns of the tensor as input
        if len(shape) > 1:
          lst = tf.split(
            params, 
            num_or_size_splits=shape[-1], 
            axis=(shape.ndims - 1))
        else:
          lst = [params]
        
        if nsamples:
          preds = self.dist(*lst).sample(nsamples)
        else:
          preds = self.dist(*lst).mean()   

        return preds

    def _build_model(self, X, loss, n_params, feature_cards, factor_dim):

        inputs_ = tfk.Input(shape=X.shape[1], dtype=X.dtype)

        fms = [
            FactorizationMachine(
              feature_cards=feature_cards, 
              factor_dim=factor_dim,
              name="sub_mdl_" + str(i))
            for i in range(n_params)        
        ]

        params = [
            fm(inputs_) for fm in fms          
        ]
        params =  tfkl.concatenate(params)

        model = tfk.Model(inputs=inputs_, outputs=params)
        model.compile(
            loss=loss(),
            optimizer=tfk.optimizers.RMSprop())
        
        return model