import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

CONV_LAYERS = {3: nn.Conv3d, 2: nn.Conv2d, 1: nn.Conv1d,
               '3d': nn.Conv3d, '2d': nn.Conv2d, '1d': nn.Conv1d}

TRANSPOSE_LAYERS = {3: nn.ConvTranspose3d, 2: nn.ConvTranspose2d, 1: nn.ConvTranspose1d,
               '3d': nn.ConvTranspose3d, '2d': nn.ConvTranspose2d, '1d': nn.ConvTranspose1d}

def vq_loss(inputs, embedded, commitment=0.25):
    '''
    Compute the codebook and commitment losses for an
    input-output pair from a VQ layer.
    '''
    return (torch.mean(torch.pow(inputs.detach() - embedded, 2)) +
            commitment * torch.mean(torch.pow(inputs - embedded.detach(), 2)))


class VQ(nn.Module):
    '''
    A vector quantization layer.

    This layer takes continuous inputs and produces a few
    different types of outputs, including a discretized
    output, a commitment loss, a codebook loss, etc.

    Args:
        num_channels: the depth of the input Tensors.
        num_latents: the number of latent values in the
          dictionary to choose from.
        dead_rate: the number of forward passes after
          which a dictionary entry is considered dead if
          it has not been used.
    '''
    def __init__(self, num_channels, num_latents, dead_rate=100, **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.num_latents = num_latents
        self.dead_rate = dead_rate

        self.dictionary = nn.Parameter(torch.randn(num_latents, num_channels))
        self.usage_count = nn.Parameter(dead_rate * torch.ones(num_latents).long(),
                                        requires_grad=False)
        self._last_batch = None


    def embed(self, idxs):
        '''
        Convert encoded indices into embeddings.

        Args:
            idxs: an [N x H x W] or [N] Tensor.

        Returns:
            An [N x H x W x C] or [N x C] Tensor.
        '''
        embedded = F.embedding(idxs, self.dictionary)
        if len(embedded.shape) == 4:
            # NHWC to NCHW
            embedded = embedded.permute(0, 3, 1, 2).contiguous()
        return embedded


    def forward(self, inputs):
        '''
        Apply vector quantization.

        If the module is in training mode, this will also
        update the usage tracker and re-initialize dead
        dictionary entries.

        Args:
            inputs: the input Tensor. Either [N x C] or
              [N x C x H x W].

        Returns:
            A tuple (embedded, embedded_pt, idxs):
              embedded: the new [N x C x H x W] Tensor
                which passes gradients to the dictionary.
              embedded_pt: like embedded, but with a
                passthrough gradient estimator. Gradients
                through this pass directly to the inputs.
              idxs: a [N x H x W] Tensor of Longs
                indicating the chosen dictionary entries.
        '''
        channels_last = inputs
        if len(inputs.shape) == 4:
            # NCHW to NHWC
            channels_last = inputs.permute(0, 2, 3, 1).contiguous()

        diffs = embedding_distances(self.dictionary, channels_last)
        idxs = torch.argmin(diffs, dim=-1)
        embedded = self.embed(idxs)
        embedded_pt = embedded.detach() + (inputs - inputs.detach())

        if self.training:
            self._update_tracker(idxs)
            self._last_batch = channels_last.detach()

        return embedded, embedded_pt, idxs


    def revive_dead_entries(self, inputs=None):
        '''
        Use the dictionary usage tracker to re-initialize
        entries that aren't being used often.

        Args:
          inputs: a batch of inputs from which random
            values are sampled for new entries. If None,
            the previous input to forward() is used.
        '''
        if inputs is None:
            assert self._last_batch is not None, ('cannot revive dead entries until a batch has ' +
                                                  'been run')
            inputs = self._last_batch
        counts = self.usage_count.detach().cpu().numpy()
        new_dictionary = None
        inputs_numpy = None
        for i, count in enumerate(counts):
            if count:
                continue
            if new_dictionary is None:
                new_dictionary = self.dictionary.detach().cpu().numpy()
            if inputs_numpy is None:
                inputs_numpy = inputs.detach().cpu().numpy().reshape([-1, inputs.shape[-1]])
            new_dictionary[i] = random.choice(inputs_numpy)
            counts[i] = self.dead_rate
        if new_dictionary is not None:
            dict_tensor = torch.from_numpy(new_dictionary).to(self.dictionary.device)
            counts_tensor = torch.from_numpy(counts).to(self.usage_count.device)
            self.dictionary.data.copy_(dict_tensor)
            self.usage_count.data.copy_(counts_tensor)


    def _update_tracker(self, idxs):
        raw_idxs = set(idxs.detach().cpu().numpy().flatten())
        update = -np.ones([self.num_latents], dtype=np.int)
        for idx in raw_idxs:
            update[idx] = self.dead_rate
        self.usage_count.data.add_(torch.from_numpy(update).to(self.usage_count.device).long())
        self.usage_count.data.clamp_(0, self.dead_rate)


def embedding_distances(dictionary, tensor):
    '''
    Compute distances between every embedding in a
    dictionary and every vector in a Tensor.

    This will not generate a huge intermediate Tensor,
    unlike the naive implementation.

    Args:
        dictionary: a [D x C] Tensor.
        tensor: a [... x C] Tensor.

    Returns:
        A [... x D] Tensor of distances.
    '''
    dict_norms = torch.sum(torch.pow(dictionary, 2), dim=-1)
    tensor_norms = torch.sum(torch.pow(tensor, 2), dim=-1)

    # Work-around for https://github.com/pytorch/pytorch/issues/18862.
    exp_tensor = tensor[..., None].view(-1, tensor.shape[-1], 1)
    exp_dict = dictionary[None].expand(exp_tensor.shape[0], *dictionary.shape)
    dots = torch.bmm(exp_dict, exp_tensor)[..., 0]
    dots = dots.view(*tensor.shape[:-1], dots.shape[-1])

    return -2 * dots + dict_norms + tensor_norms[..., None]


#################################################################################################################
class Encoder(nn.Module):
    '''
    An abstract VQ-VAE encoder, which takes input Tensors,
    shrinks them, and quantizes the result.

    Sub-classes should overload the encode() method.

    Args:
        num_channels: the number of channels in the latent
          codebook.
        num_latents: the number of entries in the latent
          codebook.
        kwargs: arguments to pass to the VQ layer.
    '''
    def __init__(self, num_channels, num_latents, **kwargs):
        super().__init__()
        self.vq = VQ(num_channels, num_latents, **kwargs)

    def encode(self, x):
        '''
        Encode a Tensor before the VQ layer.

        Args:
            x: the input Tensor.

        Returns:
            A Tensor with the correct number of output
              channels (according to self.vq).
        '''
        raise NotImplementedError

    def forward(self, x):
        '''
        Apply the encoder.

        See VQ.forward() for return values.
        '''
        return self.vq(self.encode(x))


class EighthEncoder(Encoder):
    '''
    The encoder from the original VQ-VAE paper that cuts
    the dimensions down by a factor of 4 in both
    directions.
    '''
    def __init__(self, in_channels, out_channels, num_latents, **kwargs):
        super().__init__(out_channels, num_latents, **kwargs)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 4, stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 4, stride=2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 4, stride=2)
        self.residual1 = _make_residual(out_channels)
        self.residual2 = _make_residual(out_channels)

    def encode(self, x):
        ## Padding is uneven, so we make the right and
        ## bottom more padded arbitrarily.
        x = F.pad(x, (1, 2, 1, 2))
        x = self.conv1(x)
        x = F.relu(x)
        x = F.pad(x, (1, 2, 1, 2))
        x = self.conv2(x)
        x = F.pad(x, (1, 2, 1, 2))
        x = self.conv3(x)
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        return x


class QuarterEncoder(Encoder):
    '''
    The encoder from the original VQ-VAE paper that cuts
    the dimensions down by a factor of 4 in both
    directions.
    '''
    def __init__(self, in_channels, out_channels, num_latents, **kwargs):
        super().__init__(out_channels, num_latents, **kwargs)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 4, stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 4, stride=2)
        self.residual1 = _make_residual(out_channels)
        self.residual2 = _make_residual(out_channels)

    def encode(self, x):
        ## Padding is uneven, so we make the right and
        ## bottom more padded arbitrarily.
        x = F.pad(x, (1, 2, 1, 2))
        x = self.conv1(x)
        x = F.relu(x)
        x = F.pad(x, (1, 2, 1, 2))
        x = self.conv2(x)
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        return x


class HalfEncoder(Encoder):
    '''
    An encoder that cuts the input size in half in both
    dimensions.
    '''
    def __init__(self, in_channels, out_channels, num_latents, **kwargs):
        super().__init__(out_channels, num_latents, **kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.residual1 = _make_residual(out_channels)
        self.residual2 = _make_residual(out_channels)

    def encode(self, x):
        x = self.conv(x)
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        return x


class Decoder(nn.Module):
    '''
    An abstract VQ-VAE decoder, which takes a stack of
    (differently-sized) input Tensors and produces a
    predicted output Tensor.

    Sub-classes should overload the forward() method.
    '''
    def forward(self, inputs, **kwargs):
        '''
        Apply the decoder to a list of inputs.

        Args:
            inputs: a sequence of input Tensors. There may
              be more than one in the case of a hierarchy,
              in which case the top levels come first.

        Returns:
            A decoded Tensor.
        '''
        raise NotImplementedError


class EighthDecoder(Decoder):
    '''
    The decoder from the original VQ-VAE paper that
    upsamples the dimensions by a factor of 8 in both
    directions.
    '''

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.residual1 = _make_residual(in_channels)
        self.residual2 = _make_residual(in_channels)
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, inputs):
        assert len(inputs) == 1
        x = inputs[0]
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x


class QuarterDecoder(Decoder):
    '''
    The decoder from the original VQ-VAE paper that
    upsamples the dimensions by a factor of 4 in both
    directions.
    '''

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.residual1 = _make_residual(in_channels)
        self.residual2 = _make_residual(in_channels)
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, inputs):
        assert len(inputs) == 1
        x = inputs[0]
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x


class HalfDecoder(Decoder):
    '''
    A decoder that upsamples by a factor of 2 in both
    dimensions.
    '''

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.residual1 = _make_residual(in_channels)
        self.residual2 = _make_residual(in_channels)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, inputs):
        assert len(inputs) == 1
        x = inputs[0]
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        x = F.relu(x)
        x = self.conv(x)
        return x


class HalfQuarterDecoder(Decoder):
    '''
    A decoder that takes two inputs. The first one is
    upsampled by a factor of two, and then combined with
    the second input which is further upsampled by a
    factor of four.
    '''

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.residual1 = _make_residual(in_channels)
        self.residual2 = _make_residual(in_channels)
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, 3, padding=1)
        self.residual3 = _make_residual(in_channels)
        self.residual4 = _make_residual(in_channels)
        self.conv3 = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, inputs):
        assert len(inputs) == 2

        ## Upsample the top input to match the shape of the
        ## bottom input.
        x = inputs[0]
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)

        ## Mix together the bottom and top inputs.
        x = torch.cat([x, inputs[1]], dim=1)
        x = self.conv2(x)

        x = x + self.residual3(x)
        x = x + self.residual4(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        return x


class VQVAEBase(nn.Module):
    '''
    A complete VQ-VAE hierarchy.

    There are N encoders, stored from the bottom level to
    the top level, and N decoders stored from top to
    bottom.
    '''

    def __init__(self, encoders, decoders, **kwargs):
        super().__init__()
        assert len(encoders) == len(decoders)
        self.encoders = encoders
        self.decoders = decoders
        for i, enc in enumerate(encoders):
            self.add_module('encoder_%d' % i, enc)
        for i, dec in enumerate(decoders):
            self.add_module('decoder_%d' % i, dec)


class VQVAE(VQVAEBase):


    def forward(self, inputs, commitment=0.25, **kwargs):
        '''
        Compute training losses for a batch of inputs.

        Args:
            inputs: the input Tensor. If this is a Tensor
              of integers, then cross-entropy loss will be
              used for the final decoder. Otherwise, MSE
              will be used.
            commitment: the commitment loss coefficient.

        Returns:
            A dict of Tensors, containing at least:
              loss: the total training loss.
              losses: the MSE/log-loss from each decoder.
              reconstructions: a reconstruction Tensor
                from each decoder.
              embedded: outputs from every encoder, passed
                through the vector-quantization table.
                Ordered from bottom to top level.
        '''

        all_encoded = [inputs]
        all_vq_outs = []
        all_vq_indicies = []
        total_vq_loss = 0.0
        total_recon_loss = 0.0
        for encoder in self.encoders:
            encoded = encoder.encode(all_encoded[-1])
            embedded, embedded_pt, idxs = encoder.vq(encoded)
            all_encoded.append(encoded)
            all_vq_outs.append(embedded_pt)
            all_vq_indicies.append(idxs)
            total_vq_loss = total_vq_loss + vq_loss(encoded, embedded, commitment=commitment)
        losses = []
        reconstructions = []
        for i, decoder in enumerate(self.decoders):
            dec_inputs = all_vq_outs[::-1][:i + 1]
            target = all_encoded[::-1][i + 1]
            recon = decoder(dec_inputs)
            reconstructions.append(recon)
            if target.dtype.is_floating_point:
                recon_loss = torch.mean(torch.abs(recon - target.detach()))
            else:
                recon_loss = F.cross_entropy(recon.view(-1, recon.shape[-1]), target.view(-1))
            total_recon_loss = total_recon_loss + recon_loss
            losses.append(recon_loss)
        return {
            'Overall_Loss': total_vq_loss + total_recon_loss,
            'VQ_Loss': total_vq_loss,
            'Recon_Loss': total_recon_loss,
            'reconstructions': reconstructions,
            'embedded': all_vq_outs,
            "indices": all_vq_indicies,
        }

    def decode(self, all_vq_outs):
        reconstructions = []
        for i, decoder in enumerate(self.decoders):
            dec_inputs = all_vq_outs[::-1][:i + 1]
            recon = decoder(dec_inputs)
            reconstructions.append(recon)
        return reconstructions

    def revive_dead_entries(self):
        '''
        Revive dead entries from all of the VQ layers.

        Only call this once the encoders have all been
        through a forward pass in training mode.
        '''
        for enc in self.encoders:
            enc.vq.revive_dead_entries()

    def full_reconstructions(self, inputs):
        '''
        Compute reconstructions of the inputs using all
        the different layers of the hierarchy.

        The first reconstruction uses only information
        from the top-level codes, the second uses only
        information from the top-level and second-to-top
        level codes, etc.

        This is not forward(inputs)['reconstructions'],
        since said reconstructions are simply each level's
        reconstruction of the next level's features.
        Instead, full_reconstructions reconstructs the
        original inputs.
        '''

        terms = self(inputs)
        layer_recons = []
        for encoder, recon in zip(self.encoders[:-1][::-1], terms['reconstructions'][:-1]):
            _, embedded_pt, _ = encoder.vq(recon)
            layer_recons.append(embedded_pt)
        hierarchy_size = len(self.decoders)
        results = []
        for i in range(hierarchy_size - 1):
            num_actual = i + 1
            dec_in = terms['embedded'][-num_actual:][::-1] + layer_recons[num_actual - 1:]
            results.append(self.decoders[-1](dec_in))
        results.append(terms['reconstructions'][-1])
        results += terms['indices']

        return results

    def get_indices(self, inputs, latent_count, isEmbedded):
        all_encoded = [inputs]
        all_vq_outs = []

        for ei, encoder in enumerate(self.encoders):
            encoded = encoder.encode(all_encoded[-1])
            _, embedded_pt, idxs = encoder.vq(encoded)

            if isEmbedded:
                ## Use the vectors from the codebook
                if ei > 0:
                    embedded_pt = torch.nn.functional.interpolate(embedded_pt,size=(all_encoded[-1].size()[2:]))

            else:
                ## Compute histogram of indices

                idxs = idxs.view(idxs.size(0),-1)
                embedded_pt = torch.zeros( (inputs.size(0),latent_count), dtype=torch.float32 ).cuda()
                for ii in range(idxs.size(0)):
                    curr_val = torch.cat((idxs[ii,:],torch.arange(latent_count,dtype=torch.long).cuda()),0)
                    _,curr_count = torch.unique(curr_val,return_counts=True)
                    embedded_pt[ii,:] = (curr_count-1)

                ## Verify histogram computation
                #assert torch.abs(torch.sum(embedded_pt,dim=1)[0] - 1.0) < 1e-4
                assert torch.sum(embedded_pt,dim=1)[0] == idxs.size(1)

            all_encoded.append(encoded)
            all_vq_outs.append(embedded_pt)

        return torch.cat(all_vq_outs,1)


def _make_residual(channels, dimension=2):
    '''Make a residual layer of given dimension.

    :param int channels: Number of input and output channels to the
        convolutional layers.
    :param dimension: Dimension of convolution requested.  Either
        int (``1, 2, or 3``), string (``'1d', '2d', or '3d'``) or
        can pass a convolutional layer (e.g. ``nn.Conv2d``)
    '''

    if isinstance(dimension, int) or isinstance(dimension, str):
        conv_layer = CONV_LAYERS[dimension]
    else:
        conv_layer = dimension
    return nn.Sequential(
        nn.ReLU(),
        conv_layer(channels, channels, 3, padding=1),
        nn.ReLU(),
        conv_layer(channels, channels, 1)
    )


class GeneralEncoder(Encoder):
    '''An encoder that decreases dimension by a given amount
    '''

    def __init__(self, in_channels, out_channels, num_latents, reduction=2, dimension=2, **kwargs):
        super().__init__(out_channels, num_latents, **kwargs)
        conv_layer = CONV_LAYERS[dimension]
        convs = []
        self.reduction = reduction
        self.dimension = dimension
        self.layers = int(np.log2(self.reduction))
        convs.append(conv_layer(in_channels, out_channels, 4, stride=2))
        for _ in range(self.layers - 1):
            convs.append(conv_layer(out_channels, out_channels, 4, stride=2))
        self.convs = nn.ModuleList(convs)
        self.residual1 = _make_residual(out_channels, dimension=self.dimension)
        self.residual2 = _make_residual(out_channels, dimension=self.dimension)

    def encode(self, x):
        '''Encode datapoint ``x`` using with the encoder.

        :param torch.Tensor x: A (n+1)-dimensional tensor
            where the first dimension is a batch dimension
        '''
        ## Padding is uneven, so we make the right and
        ## bottom more padded arbitrarily.

        for conv in self.convs:
            x = conv(x)
            x = F.relu(x)
        x += self.residual1(x)
        x += self.residual2(x)
        return x


class GeneralDecoder(Decoder):
    def __init__(self, in_channels, out_channels, expansion=2, dimension=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        transpose_layer = TRANSPOSE_LAYERS[dimension]
        self.residual1 = _make_residual(in_channels)
        self.residual2 = _make_residual(in_channels)
        self.dimension = dimension
        self.expansion = expansion
        self.layers = int(np.log2(self.expansion))
        transposes = []
        for _ in range(self.layers - 1):
            transposes.append(transpose_layer(in_channels, in_channels, 4, stride=2, padding=1))
        transposes.append(transpose_layer(in_channels, out_channels, 4, stride=2, padding=1))
        self.transposes = nn.ModuleList(transposes)

    def forward(self, inputs):
        assert len(inputs) == 1
        x = inputs[0]
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        for transpose in self.transposes:
            x = F.relu(x)
            x = transpose(x)
        return x


class ComposedDecoder(Decoder):
    def __init__(self, decoders):
        super().__init__()
        self.dimension = self.decoders[0].dimension
        self.decoders = nn.ModuleList(decoders)

    @classmethod
    def from_encoders(cls, encoders):
        decoders = []
        for encoder in reversed(encoders):
            decoder = GeneralDecoder(expansion=encoder.reduction, dimension=encoder.dimension)
            decoders.append(decoder)
        return cls.__init__(decoders)

    def forward(self, inputs):
        assert len(inputs) == len(self.decoders)
        x = torch.empty([0 for _ in range(self.dimension + 2)])
        for i, decoder in enumerate(self.decoders):
            x = torch.cat([x, inputs[i]], dim=1)
            x = decoder(x)
        return x
