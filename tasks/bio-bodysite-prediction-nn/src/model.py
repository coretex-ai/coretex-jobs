from typing import Callable
from pathlib import Path

import logging
import pickle

from scipy import sparse
from numpy.typing import ArrayLike
from tensorflow import Tensor

import numpy as np
import tensorflow as tf

from coretex import TaskRun

from utils import convertFromOneHot


class GatingLayer(tf.keras.layers.Layer):  # type: ignore[misc]

    def __init__(
        self,
        prev_node: int, # The number of nodes in the previous layer
        num_of_nodes: int,
        stddev_input_gates: float,
        activation_gating: Callable,
        output_layer: bool,
        sigma: float,
        a: int,
        layerId: int
    ) -> None:
        super().__init__()

        self.activation_gating = activation_gating
        self.output_layer = output_layer
        self.sigma = sigma
        self.a = a

        self.w = tf.Variable(
            tf.keras.initializers.TruncatedNormal(stddev = stddev_input_gates)([prev_node, num_of_nodes]),
            name = 'gating_weights' + str(layerId)
        )

        self.b = tf.Variable(
            tf.constant_initializer(0.0)([num_of_nodes]),
            name = 'gating_biases' + str(layerId)
        )

    @tf.function
    def call(self, alpha: Tensor, original_X: Tensor) -> tuple[Tensor, Tensor]:
        gates_layer_out = self.activation_gating(tf.matmul(alpha, self.w) + self.b)

        if self.output_layer:
            stochastic_gates = self.get_stochastic_gate_train(original_X, gates_layer_out)
            original_X = original_X * stochastic_gates

        return gates_layer_out, original_X

    def get_stochastic_gate_train(self, X: Tensor, alpha: Tensor) -> Tensor:
        # gaussian reparametrization
        base_noise = tf.random.normal(shape = tf.shape(X), mean = 0., stddev = 1.)
        z = alpha + self.sigma * base_noise
        stochastic_gate = self.hard_sigmoid(z, self.a)

        return stochastic_gate

    def hard_sigmoid(self, x: Tensor, a: Tensor) -> Tensor:
        """
            Segment-wise linear approximation of sigmoid.
            Faster than sigmoid.
            Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
            In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.
            # Arguments
                x: A tensor or variable.
            # Returns
                A tensor.
        """

        x = a * x + 0.5
        zero = tf.convert_to_tensor(0., x.dtype.base_dtype)
        one = tf.convert_to_tensor(1., x.dtype.base_dtype)
        x = tf.clip_by_value(x, zero, one)
        return x


class Model(tf.keras.Model):  # type: ignore[misc]

    def __init__(
        self,
        input_node: int,
        hidden_layers_node: list[int],
        output_node: int,
        gating_net_hidden_layers_node: list[int],
        display_step: int,
        activation_gating: str,
        activation_pred: str,
        feature_selection: bool = True,
        batch_normalization: bool = True,
        a: int = 1,
        sigma: float = 0.5,
        lam: float = 0.5,
        gamma1: int = 0,
        gamma2: int = 0,
        stddev_input_gates: float = 0.1,
        seed: int = -1
    ) -> None:

        """ LSPIN Model
        # Arguments:
            input_node: integer, input dimension of the prediction network
            hidden_layers_node: list, number of nodes for each hidden layer for the prediction net, example: [200,200]
            output_node: integer, number of nodes for the output layer of the prediction net, 1 (regression) or 2 (classification)
            gating_net_hidden_layers_node: list, number of nodes for each hidden layer of the gating net, example: [200,200]
            display_step: integer, number of epochs to output info
            activation_gating: string, activation function of the gating net: 'relu', 'l_relu', 'sigmoid', 'tanh', or 'none'
            activation_pred: string, activation function of the prediction net: 'relu', 'l_relu', 'sigmoid', 'tanh', or 'none'
            feature_selection: bool, if using the gating net
            a: float,
            sigma: float, std of the gaussion reparameterization noise
            lam: float, regularization parameter (lambda_1) of the L0 penalty
            gamma1: float, regularization parameter (lambda_2) to encourage similar samples select similar features
            gamma2: float, variant of lambda2, to encourage disimilar samples select disimilar features
            num_meta_label: integer, the number of group labels when computing the second regularization term
            stddev_input: float, std of the normal initializer for the prediction network weights
            stddev_input_gates: float, std of the normal initializer for the gating network weights
            seed: integer, random seed
        """

        super().__init__()

        if not self.checkActivationFunction(activation_pred):
            raise RuntimeError(">> [MicrobiomeForensics] Unrecognized activation function. Accetable options are \"relu\", \"tanh\", \"sigmoid\" and \"linear\"")

        tf.random.set_seed(seed)

        self.input_node = input_node
        self.hidden_layers_node = hidden_layers_node
        self.output_node = output_node
        self.gating_net_hidden_layers_node = gating_net_hidden_layers_node
        self.gating_activation_func = activation_gating
        self.batch_normalization = batch_normalization
        self.stddev_input_gates = stddev_input_gates
        self.seed = seed

        # Register hyperparameters of LSPIN
        self.a = a
        self.sigma = sigma
        self.lam = lam

        self._activation_gating = activation_gating
        self.activation_gating = activation_gating  # type: ignore[assignment]

        self.activation_pred = activation_pred

        # Register hyperparameters for training

        #self._batch_size = batch_size
        self.display_step = display_step

        # 2nd regularization parameter for the similarity penalty
        self.gamma1 = gamma1
        self.gamma2 = gamma2

        gating_network: list[GatingLayer] = []
        prediction_network: list[tf.keras.layers.Dense] = []

        if feature_selection:
            last_layer_nodes = input_node
            for i, layer_nodes in enumerate(gating_net_hidden_layers_node):
                gating_layer = GatingLayer(
                    last_layer_nodes,
                    layer_nodes, stddev_input_gates,
                    self.activation_gating,
                    False,
                    sigma,
                    a,
                    i
                )
                gating_network.append(gating_layer)
                last_layer_nodes = layer_nodes

            # Output layer of the gating network
            gating_layer = GatingLayer(
                last_layer_nodes,
                input_node,
                stddev_input_gates,
                self.activation_gating,
                True,
                sigma,
                a,
                i + 1
            )
            gating_network.append(gating_layer)

        last_layer_nodes = input_node
        for i, layer_nodes in enumerate(hidden_layers_node):
            if batch_normalization:
                batch_norm_layer = tf.keras.layers.BatchNormalization()
                prediction_network.append(batch_norm_layer)

            prediction_layer = tf.keras.layers.Dense(layer_nodes, activation = activation_pred)
            prediction_network.append(prediction_layer)
            last_layer_nodes = layer_nodes

        # Output layer of the prediction network
        if batch_normalization:
            batch_norm_layer = tf.keras.layers.BatchNormalization()
            prediction_network.append(batch_norm_layer)

        prediction_layer = tf.keras.layers.Dense(output_node, "linear")
        prediction_network.append(prediction_layer)

        self.optimizer = tf.optimizers.Adam()

        self.gating_network = gating_network
        self.prediction_network = prediction_network
        self.feature_selection = feature_selection


    def loss_fn(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_pred))

        if self.feature_selection:
            reg = 0.5 - 0.5 * tf.math.erf((-1 / (2 * self.a) - self.alpha) / (self.sigma * np.sqrt(2)))
            reg_gates = self.lam*tf.reduce_mean(tf.reduce_mean(reg, axis = -1))
            loss = loss + reg_gates

        return loss


    def call(self, X: Tensor) -> Tensor:
        self.alpha = X
        for gating_layer in self.gating_network:
            self.alpha, X = gating_layer(self.alpha, X)

        for prediction_layer in self.prediction_network:
            X = prediction_layer(X)

        return X


    def train(
        self,
        taskRun: TaskRun,
        train_data: tf.data.Dataset,
        test_data: tf.data.Dataset,
        train_batches: int,
        test_batches: int,
        epochs: int = 100,
        learningRate: float = 0.1,
    ) -> float:

        self.optimizer.learning_rate = learningRate

        # Create metrics
        self.valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        self.valid_accuracy = tf.keras.metrics.CategoricalAccuracy(name='valid_accuracy')

        for epoch in range(epochs):

            # Training
            for i, data in enumerate(train_data):
                if i == train_batches:
                    break

                self._train_step(data["features"], data["labels"])

            # Validation on train data for metrics
            for i, data in enumerate(train_data):
                if i == train_batches:
                    break

                self._valid_step(data["features"], data["labels"])

            train_loss = self.valid_loss.result()
            train_acc = self.valid_accuracy.result()
            self.valid_loss.reset_states()
            self.valid_accuracy.reset_states()

            # Validation on validation data
            for i, data in enumerate(test_data):
                if i == test_batches:
                    break

                self._valid_step(data["features"], data["labels"])

            valid_loss = self.valid_loss.result()
            valid_acc = self.valid_accuracy.result()
            self.valid_loss.reset_states()
            self.valid_accuracy.reset_states()

            taskRun.submitMetrics({
                "train_loss": (epoch + 1, float(train_loss)),
                "train_accuracy": (epoch + 1, float(train_acc)),
                "valid_loss": (epoch + 1, float(valid_loss)),
                "valid_accuracy": (epoch + 1, float(valid_acc))
            })

            if (epoch + 1) % self.display_step == 0:
                logging.info(f">> [LSPIN] Epoch: {epoch + 1}| Train [loss: {train_loss:.4f}, acc: {train_acc * 100:.2f}%] - Test [loss: {valid_loss:.4f}, acc: {valid_acc * 100:.2f}%]")

        return float(self.valid_accuracy.result())


    @tf.function
    def _train_step(self, X: Tensor, y: Tensor) -> None:
        with tf.GradientTape() as tape:
            X = self(X, training = True)
            loss = self.loss_fn(y, X)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


    @tf.function
    def _valid_step(self, X: Tensor, y: Tensor) -> Tensor:
        X = self(X)
        y_pred = tf.nn.softmax(X)
        loss = self.loss_fn(y, X)

        self.valid_loss(loss)
        y_pred_hot = self.soft_to_hot(y_pred)
        self.valid_accuracy(y, y_pred_hot)

        return y_pred_hot


    def predict(self, data: tf.data.Dataset, batches: int) -> np.ndarray:
        y_pred: list[list[int]] = []

        for i, batch in enumerate(data):
            if i == batches:
                break

            y_pred.extend(list(self._test_step(batch["features"])))

        return convertFromOneHot(np.array(y_pred))


    def test(self, data: tf.data.Dataset, batches: int) -> tuple[np.ndarray, np.ndarray]:

        y_pred: list[list[int]] = [] # List of one hot vectors
        y_true: list[list[int]] = []

        for i, batch in enumerate(data):
            if i == batches:
                break

            y_pred.extend(list(self._test_step(batch["features"])))
            y_true.extend(list(batch["labels"].numpy()))

        return convertFromOneHot(np.array(y_pred)), convertFromOneHot(np.array(y_true))


    @tf.function
    def _test_step(self, X: Tensor) -> Tensor:
        X = self(X)
        y_pred = tf.nn.softmax(X)
        return self.soft_to_hot(y_pred)


    def test_from_array(self, X: ArrayLike) -> np.ndarray:
        if type(X) == sparse.csr_matrix:
            X = X.toarray().astype(np.float32)

        return self.soft_to_hot(self._predict_from_array(X)).numpy()  # type: ignore[no-any-return]


    @tf.function
    def _predict_from_array(self, X: ArrayLike) -> Tensor:
        X = self(X)
        return tf.nn.softmax(X)


    @property
    def activation_gating(self) -> Callable:
        return self._activation_gating  # type: ignore[return-value]


    @activation_gating.setter
    def activation_gating(self, value: str) -> Callable:  # type: ignore[return]
        if value == 'relu':
            self._activation_gating = tf.nn.relu
        elif value == 'l_relu':
            self._activation_gating = tf.nn.leaky_relu
        elif value == 'sigmoid':
            self._activation_gating = tf.nn.sigmoid
        elif value == 'tanh':
            self._activation_gating = tf.nn.tanh
        elif value == 'none':
            self._activation_gating = lambda x: x  # type: ignore[assignment]
        else:
            raise NotImplementedError('activation for the gating network not recognized')


    @tf.function
    def soft_to_hot(self, y: Tensor) -> Tensor:
        indices = tf.math.argmax(y, 1)
        return tf.one_hot(indices, depth = y.shape[1])


    def get_config(self) -> dict:
        return {
            "input_node": self.input_node,
            "hidden_layers_node": self.hidden_layers_node,
            "output_node": self.output_node,
            "gating_net_hidden_layers_node": self.gating_net_hidden_layers_node,
            "display_step": self.display_step,
            "activation_gating": self.gating_activation_func,
            "activation_pred": self.activation_pred,
            "feature_selection": self.feature_selection,
            "batch_normalization": self.batch_normalization,
            "a": self.a,
            "sigma": self.sigma,
            "lam": self.lam,
            "gamma1": self.gamma1,
            "gamma2": self.gamma2,
            "stddev_input_gates": self.stddev_input_gates,
            "seed": self.seed
        }


    def save(self, path: Path) -> None:
        path.mkdir(parents = True, exist_ok = True)
        with path.joinpath("config.pkl").open("wb") as file:
            pickle.dump(self.get_config(), file)

        model_weights: list[Tensor] = []
        for layer in self.layers:
            model_weights.append(layer.get_weights())

        with path.joinpath("weights.pkl").open("wb") as file:
            pickle.dump(model_weights, file)


    @classmethod
    def load(cls, path: Path) -> tf.keras.Model:
        with path.joinpath("config.pkl").open("rb") as file:
            config = pickle.load(file)

        model = cls(**config)
        model(np.random.rand(1, config["input_node"]))

        with path.joinpath("weights.pkl").open("rb") as file:
            model_weights = pickle.load(file)

        layers = model.layers
        for i, layer_weights in enumerate(model_weights):
            layers[i].set_weights(layer_weights)

        return model


    def checkActivationFunction(self, function: str) -> bool:
        acceptableFunctions = ["relu", "tanh", "sigmoid", "linear"]
        return function in acceptableFunctions
