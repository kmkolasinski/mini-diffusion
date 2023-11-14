import math
import tensorflow as tf
from tensorflow import keras
from keras_cv.backend import ops

from mini_diffusion_tf.ops import get_dtype

layers = tf.keras.layers


class TimeDependentUNetModel(keras.Model):
    def __init__(
        self,
        img_height,
        img_width,
        hdim: int = 320,
        num_downsampling_blocks: int = 2,
        num_upsampling_blocks: int = 3,
        num_heads: int = 2,
        name=None,
    ):
        time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
        latent = keras.layers.Input((img_height, img_width, 3), name="images")

        t_emb = TimeEmbedding(dim=hdim)(time_input)
        t_emb = keras.layers.Dense(hdim)(t_emb)
        t_emb = keras.layers.Activation("swish")(t_emb)
        t_emb = keras.layers.Dense(hdim)(t_emb)

        print("time_input:", time_input)
        print("latent:", latent)
        print("t_emb:", t_emb)
        groups = hdim // 8
        print("groups:", groups)

        # Downsampling flow
        outputs = []
        x = PaddedConv2D(hdim // 4, kernel_size=3, padding=1)(latent)
        outputs.append(x)

        context = None

        for _ in range(num_downsampling_blocks):
            x = ResBlock(hdim // 4)([x, t_emb])
            # x = SpatialTransformer(
            #     num_heads, hdim // 4 // num_heads, fully_connected=True
            # )([x, context])
            outputs.append(x)
        x = PaddedConv2D(hdim // 4, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(num_downsampling_blocks):
            x = ResBlock(hdim // 2)([x, t_emb])
            # x = SpatialTransformer(
            #     2 * num_heads, hdim // 4 // num_heads, fully_connected=True
            # )([x, context])
            outputs.append(x)
        x = PaddedConv2D(hdim // 2, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(num_downsampling_blocks):
            x = ResBlock(hdim)([x, t_emb])
            x = SpatialTransformer(
                4 * num_heads, hdim // 4 // num_heads, fully_connected=True
            )([x, context])
            outputs.append(x)
        x = PaddedConv2D(hdim, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(num_downsampling_blocks):
            x = ResBlock(hdim)([x, t_emb])
            outputs.append(x)

        # Middle flow

        x = ResBlock(hdim)([x, t_emb])
        x = SpatialTransformer(
            4 * num_heads, hdim // 4 // num_heads, fully_connected=True
        )([x, context])
        x = ResBlock(hdim)([x, t_emb])

        # Upsampling flow

        for _ in range(num_upsampling_blocks):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(hdim)([x, t_emb])
        x = Upsample(hdim)(x)

        for _ in range(num_upsampling_blocks):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(hdim)([x, t_emb])
            x = SpatialTransformer(
                4 * num_heads, hdim // 4 // num_heads, fully_connected=True
            )([x, context])
        x = Upsample(hdim)(x)

        for _ in range(num_upsampling_blocks):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(hdim // 2)([x, t_emb])
            # x = SpatialTransformer(
            #     2 * num_heads, hdim // 4 // num_heads, fully_connected=True
            # )([x, context])
        x = Upsample(hdim // 2)(x)

        for _ in range(num_upsampling_blocks):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(hdim // 4)([x, t_emb])
            # x = SpatialTransformer(
            #     num_heads, hdim // 4 // num_heads, fully_connected=True
            # )([x, context])

        # Exit flow

        x = keras.layers.GroupNormalization(groups=groups, epsilon=1e-5)(x)
        x = keras.layers.Activation("swish")(x)
        output = PaddedConv2D(3, kernel_size=3, padding=1)(x)

        super().__init__([latent, time_input], output, name=name)


class TimeEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)
        self.emb = tf.cast(self.emb, get_dtype())

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=get_dtype())
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


class PaddedConv2D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding=0, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.padding2d = keras.layers.ZeroPadding2D(padding)
        self.conv2d = keras.layers.Conv2D(filters, kernel_size, strides=strides)

    def call(self, inputs):
        x = self.padding2d(inputs)
        return self.conv2d(x)


class ResBlock(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        groups = output_dim // 8
        print("groups:", output_dim, groups)
        self.entry_flow = [
            keras.layers.GroupNormalization(groups=groups, epsilon=1e-5),
            keras.layers.Activation("swish"),
            PaddedConv2D(output_dim, 3, padding=1),
        ]
        self.embedding_flow = [
            keras.layers.Activation("swish"),
            keras.layers.Dense(output_dim),
        ]
        self.exit_flow = [
            keras.layers.GroupNormalization(groups=groups, epsilon=1e-5),
            keras.layers.Activation("swish"),
            PaddedConv2D(output_dim, 3, padding=1),
        ]

    def build(self, input_shape):
        if input_shape[0][-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1)
        else:
            self.residual_projection = lambda x: x

    def call(self, inputs):
        inputs, embeddings = inputs
        x = inputs
        for layer in self.entry_flow:
            x = layer(x)
        for layer in self.embedding_flow:
            embeddings = layer(embeddings)
        x = x + embeddings[:, None, None]
        for layer in self.exit_flow:
            x = layer(x)
        return x + self.residual_projection(inputs)


class SpatialTransformer(keras.layers.Layer):
    def __init__(self, num_heads, head_size, fully_connected=False, **kwargs):
        super().__init__(**kwargs)
        self.norm = keras.layers.GroupNormalization(epsilon=1e-5)
        channels = num_heads * head_size
        if fully_connected:
            self.proj1 = keras.layers.Dense(num_heads * head_size)
        else:
            self.proj1 = PaddedConv2D(num_heads * head_size, 1)
        self.transformer_block = BasicTransformerBlock(channels, num_heads, head_size)
        if fully_connected:
            self.proj2 = keras.layers.Dense(channels)
        else:
            self.proj2 = PaddedConv2D(channels, 1)

    def call(self, inputs):
        inputs, context = inputs
        _, h, w, c = inputs.shape
        x = self.norm(inputs)
        x = self.proj1(x)
        x = ops.reshape(x, (-1, h * w, c))
        x = self.transformer_block([x, context])
        x = ops.reshape(x, (-1, h, w, c))
        return self.proj2(x) + inputs


class BasicTransformerBlock(keras.layers.Layer):
    def __init__(self, dim, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn1 = CrossAttention(num_heads, head_size)
        # self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        # self.attn2 = CrossAttention(num_heads, head_size)
        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-5)
        # self.geglu = GEGLU(dim * 4)
        self.activation = tf.keras.activations.swish
        self.dense = keras.layers.Dense(dim)

    def call(self, inputs):
        inputs, context = inputs
        x = self.attn1(self.norm1(inputs), context=None) + inputs
        # x = self.attn2(self.norm2(x), context=context) + x
        return self.dense(self.activation(self.norm3(x))) + x


class CrossAttention(keras.layers.Layer):
    def __init__(self, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        self.to_q = keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.to_k = keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.to_v = keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.scale = head_size**-0.5
        self.num_heads = num_heads
        self.head_size = head_size
        self.out_proj = keras.layers.Dense(num_heads * head_size)

    def call(self, inputs, context=None):
        if context is None:
            context = inputs
        q, k, v = self.to_q(inputs), self.to_k(context), self.to_v(context)
        q = ops.reshape(q, (-1, inputs.shape[1], self.num_heads, self.head_size))
        k = ops.reshape(k, (-1, context.shape[1], self.num_heads, self.head_size))
        v = ops.reshape(v, (-1, context.shape[1], self.num_heads, self.head_size))

        q = ops.transpose(q, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)
        k = ops.transpose(k, (0, 2, 3, 1))  # (bs, num_heads, head_size, time)
        v = ops.transpose(v, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)

        score = td_dot(q, k) * self.scale
        weights = keras.activations.softmax(score)  # (bs, num_heads, time, time)
        attn = td_dot(weights, v)
        attn = ops.transpose(attn, (0, 2, 1, 3))  # (bs, time, num_heads, head_size)
        out = ops.reshape(attn, (-1, inputs.shape[1], self.num_heads * self.head_size))
        return self.out_proj(out)


class Upsample(keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.ups = keras.layers.UpSampling2D(2, interpolation="bilinear")
        self.conv = PaddedConv2D(channels, 3, padding=1)

    def call(self, inputs):
        return self.conv(self.ups(inputs))


class GEGLU(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dense = keras.layers.Dense(output_dim * 2)

    def call(self, inputs):
        x = self.dense(inputs)
        x, gate = x[..., : self.output_dim], x[..., self.output_dim :]
        tanh_res = keras.activations.tanh(
            gate * 0.7978845608 * (1 + 0.044715 * (gate**2))
        )
        return x * 0.5 * gate * (1 + tanh_res)


def td_dot(a, b):
    aa = ops.reshape(a, (-1, a.shape[2], a.shape[3]))
    bb = ops.reshape(b, (-1, b.shape[2], b.shape[3]))
    cc = keras.layers.Dot(axes=(2, 1))([aa, bb])
    return ops.reshape(cc, (-1, a.shape[1], cc.shape[1], cc.shape[2]))
