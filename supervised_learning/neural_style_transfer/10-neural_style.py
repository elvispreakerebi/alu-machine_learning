#!/usr/bin/env python3
"""
Neural Style Transfer (NST).
"""

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()


class NST:
    """
    Performs tasks for neural style transfer.
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1, var=10):
        """
        Class constructor.

        Args:
            style_image (np.ndarray): Style reference image (h, w, 3).
            content_image (np.ndarray): Content reference image (h, w, 3).
            alpha (float): Weight for content cost.
            beta (float): Weight for style cost.
            var (float): Weight for variational (TV) cost.

        After init, sets style_image, content_image, alpha, beta, var, model,
        gram_style_features, and content_feature.
        """
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if (isinstance(alpha, bool) or
                not isinstance(alpha, (int, float)) or alpha < 0):
            raise TypeError("alpha must be a non-negative number")
        if (isinstance(beta, bool) or
                not isinstance(beta, (int, float)) or beta < 0):
            raise TypeError("beta must be a non-negative number")
        if (isinstance(var, bool) or
                not isinstance(var, (int, float)) or var < 0):
            raise TypeError("var must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.var = var
        self.load_model()
        self.generate_features()

    def load_model(self):
        """
        Creates the model used to calculate cost.

        Builds a Keras model from VGG19 (no top) with the same input as VGG19
        and outputs the style layer activations followed by the content layer.
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet',
        )
        vgg.trainable = False
        outputs = (
            [vgg.get_layer(name).output for name in self.style_layers] +
            [vgg.get_layer(self.content_layer).output]
        )
        self.model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

    def generate_features(self):
        """
        Extracts style Gram matrices and content layer activations.

        Sets gram_style_features from the style image's style-layer outputs
        and content_feature from the content image's content-layer output.
        """
        n_style = len(self.style_layers)
        style_outputs = self.model(self.style_image)
        self.gram_style_features = [
            self.gram_matrix(style_outputs[i]) for i in range(n_style)
        ]
        content_outputs = self.model(self.content_image)
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Style cost for a single layer.

        E_l = (1 / C_l^2) * sum_ij (G_ij - A_ij)^2

        Args:
            style_output (tf.Tensor or tf.Variable): Layer activations (1,h,w,c).
            gram_target (tf.Tensor or tf.Variable): Target Gram (1, c, c).

        Returns:
            tf.Tensor: Scalar style cost for the layer.
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)):
            raise TypeError("style_output must be a tensor of rank 4")
        rank_s = style_output.shape.ndims
        if rank_s is None or rank_s != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        c_dim = style_output.shape[3]
        if c_dim is None:
            c_dim = int(tf.shape(style_output)[3].numpy())
        else:
            c_dim = int(c_dim)

        def _gram_target_typeerror():
            return TypeError(
                "gram_target must be a tensor of shape [1, {c}, {c}] where "
                "{c} is the number of channels in style_output".format(c=c_dim)
            )

        if not isinstance(gram_target, (tf.Tensor, tf.Variable)):
            raise _gram_target_typeerror()

        rank_g = gram_target.shape.ndims
        if rank_g is None or rank_g != 3:
            raise _gram_target_typeerror()

        g0 = gram_target.shape[0]
        g1 = gram_target.shape[1]
        g2 = gram_target.shape[2]
        if g0 is not None and g1 is not None and g2 is not None:
            if (int(g0) != 1 or int(g1) != int(g2) or
                    int(g1) != c_dim):
                raise _gram_target_typeerror()
        else:
            gt = tf.shape(gram_target)
            if (int(gt[0].numpy()) != 1 or
                    int(gt[1].numpy()) != int(gt[2].numpy()) or
                    int(gt[1].numpy()) != c_dim):
                raise _gram_target_typeerror()

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """
        Combined style cost: L_style = sum_l w_l * E_l.

        Weights w_l are equal and sum to 1 (w_l = 1 / L).

        Args:
            style_outputs (list): Style-layer outputs for the generated image;
                length must match self.style_layers.

        Returns:
            tf.Tensor: Scalar style cost.
        """
        l_len = len(self.style_layers)
        if (not isinstance(style_outputs, list) or
                len(style_outputs) != l_len):
            raise TypeError(
                "style_outputs must be a list with a length of {l} where "
                "{l} is the length of self.style_layers".format(l=l_len)
            )
        weight = 1.0 / float(l_len)
        terms = []
        for i in range(l_len):
            e_l = self.layer_style_cost(
                style_outputs[i], self.gram_style_features[i])
            terms.append(weight * e_l)
        return tf.add_n(terms)

    def content_cost(self, content_output):
        """
        Content cost for the generated image.

        L_content = (1 / (H W C)) * sum_ijk (F_ijk - P_ijk)^2

        Args:
            content_output (tf.Tensor or tf.Variable): Content-layer output
                for the generated image; shape must match self.content_feature.

        Returns:
            tf.Tensor: Scalar content cost.
        """
        cf = self.content_feature
        s_str = str(tuple(
            int(x) for x in tf.shape(cf).numpy().tolist()
        ))

        def _raise_typeerror():
            raise TypeError(
                "content_output must be a tensor of shape {s} where {s} is "
                "the shape of self.content_feature".format(s=s_str)
            )

        if not isinstance(content_output, (tf.Tensor, tf.Variable)):
            _raise_typeerror()
        same_shape = tf.reduce_all(
            tf.equal(tf.shape(content_output), tf.shape(cf)))
        if not bool(same_shape.numpy()):
            _raise_typeerror()
        return tf.reduce_mean(tf.square(content_output - cf))

    def _forward_total_cost(self, generated_image):
        """
        Forward pass: VGG features, style/content costs, total J (no validation).

        Args:
            generated_image: Batch (1, nh, nw, 3) in [0, 1].

        Returns:
            tuple: (J_total, J_content, J_style, J_var).
        """
        vgg19 = tf.keras.applications.vgg19
        preprocessed = vgg19.preprocess_input(generated_image * 255.0)
        outputs = self.model(preprocessed)
        J_style = self.style_cost(outputs[:-1])
        J_content = self.content_cost(outputs[-1])
        J_var = NST.variational_cost(generated_image)
        J_total = (
            tf.cast(self.alpha, tf.float32) * J_content +
            tf.cast(self.beta, tf.float32) * J_style +
            tf.cast(self.var, tf.float32) * J_var
        )
        return J_total, J_content, J_style, J_var

    def total_cost(self, generated_image):
        """
        Total cost: J = alpha * L_content + beta * L_style + var * L_var.

        Args:
            generated_image (tf.Tensor or tf.Variable): Shape (1, nh, nw, 3);
                must match self.content_image.

        Returns:
            tuple: (J, J_content, J_style, J_var).
        """
        ci = self.content_image
        s_str = str(tuple(
            int(x) for x in tf.shape(ci).numpy().tolist()
        ))

        def _raise_typeerror():
            raise TypeError(
                "generated_image must be a tensor of shape {s} where {s} is "
                "the shape of self.content_image".format(s=s_str)
            )

        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            _raise_typeerror()
        same_shape = tf.reduce_all(
            tf.equal(tf.shape(generated_image), tf.shape(ci)))
        if not bool(same_shape.numpy()):
            _raise_typeerror()

        return self._forward_total_cost(generated_image)

    def compute_grads(self, generated_image):
        """
        Gradients of total cost with respect to generated_image.

        Args:
            generated_image (tf.Tensor or tf.Variable): Same shape as
                self.content_image.

        Returns:
            tuple: (gradients, J_total, J_content, J_style, J_var).
        """
        ci = self.content_image
        s_str = str(tuple(
            int(x) for x in tf.shape(ci).numpy().tolist()
        ))

        def _raise_typeerror():
            raise TypeError(
                "generated_image must be a tensor of shape {s} where {s} is "
                "the shape of self.content_image".format(s=s_str)
            )

        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            _raise_typeerror()
        same_shape = tf.reduce_all(
            tf.equal(tf.shape(generated_image), tf.shape(ci)))
        if not bool(same_shape.numpy()):
            _raise_typeerror()

        tape_cls = getattr(tf, 'GradientTape', None)
        if tape_cls is None:
            tape_cls = tf.contrib.eager.GradientTape

        with tape_cls() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style, J_var = self._forward_total_cost(
                generated_image)

        gradients = tape.gradient(J_total, generated_image)
        return gradients, J_total, J_content, J_style, J_var

    def generate_image(
            self, iterations=1000, step=None, lr=0.01,
            beta1=0.9, beta2=0.99):
        """
        Optimize pixels with Adam to minimize content, style, and TV cost.

        Args:
            iterations (int): Number of gradient steps (loop runs iterations+1
                evaluations; updates occur for i < iterations).
            step (int or None): Print every step iterations and at ends.
            lr (float or int): Adam learning rate.
            beta1 (float): Adam beta1.
            beta2 (float): Adam beta2.

        Returns:
            tuple: (best_image ndarray, best_cost float).
        """
        if (isinstance(iterations, bool) or
                not isinstance(iterations, (int, np.integer))):
            raise TypeError("iterations must be an integer")
        if int(iterations) <= 0:
            raise ValueError("iterations must be positive")
        iterations = int(iterations)
        if step is not None:
            if isinstance(step, bool) or not isinstance(
                    step, (int, np.integer)):
                raise TypeError("step must be an integer")
            step = int(step)
            if step <= 0 or step >= iterations:
                raise ValueError(
                    "step must be positive and less than iterations")
        if isinstance(lr, bool) or not isinstance(lr, (int, float)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0.0 or beta1 > 1.0:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0.0 or beta2 > 1.0:
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = tf.Variable(
            tf.identity(self.content_image),
            trainable=True,
            dtype=tf.float32,
        )

        try:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=float(lr),
                beta_1=float(beta1),
                beta_2=float(beta2),
            )
        except TypeError:
            optimizer = tf.keras.optimizers.Adam(
                lr=float(lr),
                beta_1=float(beta1),
                beta_2=float(beta2),
            )

        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            grads, J_total, J_content, J_style, J_var = self.compute_grads(
                generated_image)
            jt = float(J_total.numpy())
            jc = float(J_content.numpy())
            js = float(J_style.numpy())
            jv = float(J_var.numpy())

            if jt < best_cost:
                best_cost = jt
                best_image = generated_image.numpy().copy()

            if step is None:
                do_print = (i == 0 or i == iterations)
            else:
                do_print = (
                    i == 0 or i == iterations or (i % step == 0))

            if do_print:
                print(
                    "Cost at iteration {i}: {J_total}, content {J_content}, "
                    "style {J_style}, var {J_var}".format(
                        i=i, J_total=jt, J_content=jc, J_style=js,
                        J_var=jv))

            if i < iterations:
                optimizer.apply_gradients([(grads, generated_image)])
                generated_image.assign(
                    tf.clip_by_value(generated_image, 0.0, 1.0))

        return best_image, best_cost

    @staticmethod
    def variational_cost(generated_image):
        """
        Variational (total-variation squared) cost on the generated image.

        Penalizes squared differences between adjacent pixels horizontally and
        vertically.

        Args:
            generated_image (tf.Tensor): Batch of shape (1, nh, nw, 3).

        Returns:
            tf.Tensor: Scalar variational cost.
        """
        dh = generated_image[:, :, 1:, :] - generated_image[:, :, :-1, :]
        dv = generated_image[:, 1:, :, :] - generated_image[:, :-1, :, :]
        return (
            tf.reduce_sum(tf.square(dh)) +
            tf.reduce_sum(tf.square(dv))
        )

    @staticmethod
    def gram_matrix(input_layer):
        """
        Computes the Gram matrix of a convolutional layer output.

        G_ij = (1 / (H * W)) * sum_{h,w} F_hwi * F_hwj

        Args:
            input_layer (tf.Tensor or tf.Variable): Shape (1, h, w, c).

        Returns:
            tf.Tensor: Gram matrix of shape (1, c, c).
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        rank = input_layer.shape.ndims
        if rank is None or rank != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = tf.unstack(tf.shape(input_layer))
        feats = tf.reshape(input_layer, (1, h * w, c))
        gram = tf.matmul(feats, feats, transpose_a=True)
        gram = gram / tf.cast(h * w, tf.float32)
        return gram

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its largest side is 512 pixels, then
        rescales pixel values from [0, 255] to [0, 1] after bicubic resize.

        Args:
            image (np.ndarray): Image array of shape (h, w, 3).

        Returns:
            tf.Tensor: Scaled image tensor of shape (1, h_new, w_new, 3).
        """
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        if h >= w:
            h_new = 512
            w_new = int(np.round(w * 512.0 / h))
        else:
            w_new = 512
            h_new = int(np.round(h * 512.0 / w))

        img = tf.convert_to_tensor(image, dtype=tf.float32)
        img = tf.expand_dims(img, axis=0)
        img = tf.image.resize_bicubic(img, size=(h_new, w_new))
        img = img / 255.0
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img
