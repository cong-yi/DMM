import torch
from networks.dmm_net import DMM


def decode_sdf(decoder, latent_vector, queries):
    if isinstance(decoder, DMM):
        sdf = decoder.inference(latent_vector, queries)
    else:
        sdf = decoder(queries)

    return sdf
