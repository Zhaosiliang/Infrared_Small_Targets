from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn


class AsymBiChaFuse(HybridBlock):
    def __init__(self, channels=64, r=4):
        super(AsymBiChaFuse, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)

        with self.name_scope():
            self.topdown = nn.HybridSequential(prefix='topdown')
            self.topdown.add(nn.GlobalAvgPool2D())
            self.topdown.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                       padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('relu'))
            self.topdown.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                       padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('sigmoid'))

            self.bottomup = nn.HybridSequential(prefix='bottomup')
            self.bottomup.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                        padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('relu'))
            self.bottomup.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                        padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('sigmoid'))

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):
        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * F.broadcast_mul(xl, topdown_wei) + 2 * F.broadcast_mul(xh, bottomup_wei)
        xs = self.post(xs)

        return xs
