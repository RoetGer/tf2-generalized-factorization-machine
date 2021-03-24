import inspect

from functools import partial

import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from fm_zoo import AutomaticFeatureInteraction
from fm_zoo import AttentionalFactorizationMachine 
from fm_zoo import DeepFM
from fm_zoo import FactorizationMachine
from fm_zoo import FieldAwareNeuralFactorizationMachine
from fm_zoo import FMNeuralNetwork
from fm_zoo import NeuralFactorizationMachine
from fm_zoo import CompressedInteractionNetwork, ExtremeDeepFactorizationMachine
from fm_zoo import FieldAwareFactorizationMachine
from tf2_dist_utils.losses import build_loss

mdl_dic = {
    "fm": (
        FactorizationMachine,
        {}
    ),
    "afi": (
        AutomaticFeatureInteraction,
        {
            "n_heads": 5, 
            "n_attentions": 5, 
            "hidden_sizes": [5, 3, 1]
        }
    ),
    "afm": (
        AttentionalFactorizationMachine,
        {
            "attention_size": 2
        }
    ),
    "dfm": (
        DeepFM,
        {
            "hidden_sizes": [5, 3, 1]
        }
    ),
    "ffm": (
        FieldAwareFactorizationMachine,
        {}
    ),
    "fnfm": (
        FieldAwareNeuralFactorizationMachine,
        {
            "hidden_sizes": [5, 3, 1]
        }
    ),
    "fnn": (
        FMNeuralNetwork,
        {
            "hidden_sizes": [5, 3, 1]
        }
    ),
    "nfm": (
        NeuralFactorizationMachine,
        {
            "hidden_sizes": [5, 3, 1]           
        }
    ),
    "cin": (
        CompressedInteractionNetwork,
        {
            "cin_hidden_sizes": [100, 50, 10]        
        }
    ), 
    "xdfm": (
        ExtremeDeepFactorizationMachine,
        {
            "fnn_hidden_sizes": [20, 10, 1],
            "cin_hidden_sizes": [10, 5]
        }
    )
}


class GenFactMachine():
    '''Generalized Factorization Machine
    
    The basic idea of the Generalized Factorization Machine (GFM) is to 
    associate with each parameter of a distribution a separate Factorization
    Machine.
    
    E.g. we can assume the data comes from a Gaussian distribution, for each
    observation i, we have: y_i ~ N(mu_i, sigma_i). The GFM assume now that 
    mu_i = f(x_i) and sigma_i = g(x_i). The GFM approximates the two functions
    with Factorization Machines.

    Parameters
    ----------
    target_dist : tfp.distribution.Distribution
        Type of distribution from which targets have been sampled.
    feature_cards : list[int]
        Number of unique factors for each feature.
    factor_dim : int
        Number of dimensions of the latent factorization space.
    '''
    
    def __init__(self, target_dist, feature_cards, factor_dim, **kwargs):    
        self.target_dist = target_dist
        self.feature_cards = feature_cards
        self.factor_dim = factor_dim
        self.n_params = len(inspect.signature(target_dist).parameters)
        
        kwargs.update({
              "feature_cards": feature_cards, 
              "factor_dim": factor_dim
        })
        default_mdl_param.update(kwargs)
        
        self.base_mdl = partial(FactorizationMachine, **default_mdl_param)

    def fit(self, X, y, **kwargs):
        '''Fit GFM to the training data.

        Parameters
        ----------
        X : Array-like of shape (n_samples, n_features)
            The training input samples.

        y : Array-like of shape (n_samples,) or (n_samples, n_target_dimension)
            The target values.
        
        Returns
        -------
        self : GenFactMachine
            Fitted GFM model.
        '''
        
        self.loss = build_loss("GFMloss", self.target_dist)

        self.model = self._build_model(
            self.base_mdl,
            build_loss("GFMloss", self.target_dist),
            X.shape[1],
            X.dtype,
            self.n_params)

        default_fit_param = {
            "validation_split": 0.15, 
            "batch_size": 16,
            "epochs": 100,
            "callbacks": [
              tfk.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        }
        default_fit_param.update(kwargs)

        self.hist = self.model.fit(
            X, y,
            **default_fit_param)
        
        return self

    def predict(self, X, nsamples=None):
        '''Predict mean of or creates samples from target distribution.

        Parameters
        ----------
        X : Array-like of shape (n_samples, n_features)
            The input samples.
        nsamples : int, default=None
            Number of samples to be sampled from the target distribution

        Returns
        -------
        preds : tf.tensor
            Mean prediction for each observation or nsamples for
            each observation from target_dist.
        '''
        params = tf.constant(self.model.predict(X))
        
        shape = params.shape

        # model.predict returns tensor of parameters
        # target_dist objects are assumed to take the individual
        # columns of the tensor as input
        if len(shape) > 1:
          lst = tf.split(
            params, 
            num_or_size_splits=shape[-1], 
            axis=(shape.ndims - 1))
        else:
          lst = [params]
        
        if nsamples:
          preds = self.target_dist(*lst).sample(nsamples)
        else:
          preds = self.target_dist(*lst).mean()   

        return preds

    def _build_model(self, 
                     mdl, 
                     loss, 
                     input_shape, 
                     input_dtype,
                     n_params, 
                     **kwargs):

        inputs_ = tfk.Input(shape=input_shape, dtype=X.dtype)

        fms = [
            mdl(name="mdl" + str(i))
                for i in range(n_params)        
        ]

        params = [
            fm(inputs_) for fm in fms          
        ]
        params =  tfkl.concatenate(params)

        default_compile_param = {
            "optimizer": tfk.optimizers.RMSprop()
        }
        default_compile_param.update(kwargs)

        model = tfk.Model(inputs=inputs_, outputs=params)
        model.compile(
            loss=loss(),
            **default_compile_param)
        
        return model