import pickle
import os
import time
import tensorflow as tf
from tensorflow.keras.regularizers import l2

from model import Model
from utils.parser_utils import minibatches, load_and_preprocess_data

"""
10 epochs, hidden size (200, cube) -> 88.35 UAS 
"""

class Config(object):
    n_features = 36
    n_classes = 3
    dropout = 0.5  # (p_drop in the handout)
    embed_size = 50
    hidden_size = 200
    batch_size = 1024
    n_epochs = 10
    lr = 0.0005
    # Both of these improve performance significantly, especially the cube activation function
    lambda_ = 1e-2 # 0
    cube = True # False


class ParserModel(Model):
    """
    Implements a feedforward neural network with an embedding layer and single hidden layer.
    This network will predict which transition should be applied to a given partial parse
    configuration.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, n_classes), type tf.float32
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32"""

        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.n_features))
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.n_classes))
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=0):
        """Creates the feed_dict for the dependency parser.
        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = None
        feed_dict = {self.input_placeholder: inputs_batch, self.dropout_placeholder: dropout}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
            - Creates a tf.Variable and initializes it with self.pretrained_embeddings.
            - Uses the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, n_features, embedding_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, n_features * embedding_size).
        Returns:
            embeddings: tf.Tensor of shape (None, n_features*embed_size)
        """
        embeddings = tf.Variable(self.pretrained_embeddings)
        embeddings = tf.nn.embedding_lookup(embeddings, self.input_placeholder)
        embeddings = tf.reshape(embeddings, (-1, self.config.n_features * self.config.embed_size))
        return embeddings

    def add_prediction_op(self):
        """Adds the 1-hidden-layer NN:
            h = Relu(xW1 + b1)
            h_drop = Dropout(h, dropout_rate)
            pred = h_dropW2 + b2

        Note that we are not applying a softmax to pred. The softmax will instead be done in
        the add_loss_op function, which improves efficiency because we can use
        tf.nn.softmax_cross_entropy_with_logits

        Use the initializer from q2_initialization.py to initialize W and W2 (you can initialize b1
        and b2 with zeros)

        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
        """
        D = self.config.n_features * self.config.embed_size
        H = self.config.hidden_size
        C = self.config.n_classes

        x = self.add_embedding()
        h = tf.keras.layers.Dense(H, input_shape=(D,), kernel_regularizer=l2(self.config.lambda_)) (x)
        h = tf.pow(h, 3 * tf.ones_like(h))
        h_drop = tf.layers.dropout(h, rate = 1 - self.dropout_placeholder)
        pred = tf.keras.layers.Dense(C, input_shape=(D, ), kernel_regularizer=l2(self.config.lambda_)) (h_drop)
        return pred

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        In this case we are using cross entropy loss.
        The loss should be averaged over all examples in the current minibatch.
        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(self.labels_placeholder, pred))
        return loss

    def add_training_op(self, loss):
        """
        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess, parser, train_examples, dev_set):
        n_minibatches = 1 + len(train_examples) / self.config.batch_size
        prog = tf.keras.utils.Progbar(target=n_minibatches)
        for i, (train_x, train_y) in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, train_x, train_y)
            prog.update(i + 1, [("train loss", loss)])
        print(" Evaluating on dev set", end=' ')
        dev_UAS, _ = parser.parse(dev_set)
        print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
        return dev_UAS

    def fit(self, sess, saver, parser, train_examples, dev_set):
        best_dev_UAS = 0
        for epoch in range(self.config.n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_UAS = self.run_epoch(sess, parser, train_examples, dev_set)
            if dev_UAS > best_dev_UAS:
                best_dev_UAS = dev_UAS
                if saver:
                    print("New best dev UAS! Saving model in ./data/weights/parser.weights")
                    saver.save(sess, './data/weights/parser.weights')
            print()

    def __init__(self, config, pretrained_embeddings):
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.build()


def main(debug=False):
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    config = Config()
    parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(debug)
    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    with tf.Graph().as_default() as graph:
        print("Building model...", end=' ')
        start = time.time()
        model = ParserModel(config, embeddings)
        parser.model = model
        init_op = tf.global_variables_initializer()
        saver = None if debug else tf.train.Saver()
        print("took {:.2f} seconds\n".format(time.time() - start))
    graph.finalize()

    with tf.Session(graph=graph) as session:
        parser.session = session
        session.run(init_op)

        print(80 * "=")
        print("TRAINING")
        print(80 * "=")
        model.fit(session, saver, parser, train_examples, dev_set)

        if not debug:
            print(80 * "=")
            print("TESTING")
            print(80 * "=")
            print("Restoring the best model weights found on the dev set")
            saver.restore(session, './data/weights/parser.weights')
            print("Final evaluation on test set", end=' ')
            UAS, dependencies = parser.parse(test_set)
            print("- test UAS: {:.2f}".format(UAS * 100.0))
            print("Writing predictions")
            with open('q2_test.predicted.pkl', 'wb') as f:
                pickle.dump(dependencies, f, -1)
            print("Done!")


if __name__ == '__main__':
    main()

