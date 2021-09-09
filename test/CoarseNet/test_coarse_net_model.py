import unittest

from tensorflow.keras import layers

from CoarseNet import coarse_net_model

# from my_sum import sum

# def get_mock_conv_block():
#     top = layers.Conv2D(w_size[0], (w_size[1], w_size[2]),
#                         kernel_regularizer=regularizers.l2(5e-5),
#                         padding='same',
#                         strides=strides,
#                         dilation_rate=dilation_rate,
#                         name=conv_type + name)(bottom)
#     top = layers.BatchNormalization(name='bn-' + name)(top)
#     top = layers.PReLU(alpha_initializer='zero', shared_axes=[
#         1, 2], name='prelu-' + name)(top)


class TestSum(unittest.TestCase):
    # def test_list_int(self):
    #     """
    #     Test that it can sum a list of integers
    #     """
    #     data = [1, 2, 3]
    #     result = sum(data)
    #     self.assertEqual(result, 6)

    # def test_list_fraction(self):
    #     """
    #     Test that it can sum a list of fractions
    #     """
    #     data = [Fraction(1, 4), Fraction(1, 4), Fraction(2, 5)]
    #     result = sum(data)
    #     self.assertEqual(result, 1)

    def test_conv_bn_prelu(self):
        input_layer = layers.Input((100, 100, 1))

        # mock_conv_block =

        result = coarse_net_model.conv_bn_prelu(input_layer, (100, 5, 5), "some_name")

        self.assertEqual(result, None)


if __name__ == '__main__':
    unittest.main()
