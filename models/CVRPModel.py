import jax
import jax.numpy as jnp
from flax import nnx
from dataclasses import dataclass
from .TSPModel import TSPModel, TSPModelConfig
from typing import Any


@dataclass
class CVRPModelConfig(TSPModelConfig):
    encoder_input_dim: int = 3

    @staticmethod
    def get_config(which: str = ''):
        match which:
            case '' | 'default':
                return CVRPModelConfig()
            case 'qk_norm_base':
                return CVRPModelConfig(
                    qk_norm=True, softcap=None,
                    final_norm=True, final_norm_use_scale=True,
                    final_softcap=None, final_logit_scale=16 / 256,
                    num_layers=(32, 12),
                )
            case 'qk_norm_base_dim512':
                return CVRPModelConfig(
                    embed_dim=512, num_heads=8,
                    qk_norm=True, softcap=None,
                    final_norm=True, final_norm_use_scale=True,
                    final_softcap=None, final_logit_scale=16 / 512,
                    num_layers=(16, 6),
                )
            case 'qk_norm_small_dim512':
                return CVRPModelConfig(
                    embed_dim=512, num_heads=8,
                    qk_norm=True, softcap=None,
                    final_norm=True, final_norm_use_scale=True,
                    final_softcap=None, final_logit_scale=16 / 512,
                    num_layers=(8, 3),
                )
            case _:
                raise ValueError()

    def construct_model(self):
        return CVRPModel(**vars(self))
    

class CVRPModel(TSPModel):
    def __init__(
        self,
        num_layers: tuple[int, int] = (16, 6) ,
        embed_dim: tuple[int, int] = 256,
        hidden_dim: int | None = None,
        num_heads: tuple[int, int] = 8,
        activation: str = 'silu',
        use_swiglu: bool = True,
        *,
        dtype: jax.typing.DTypeLike = jnp.bfloat16,
        encoder_input_dim: int = 3,
        softcap: float | None = 30.,
        final_norm: bool = True,
        qk_norm: bool = False,
        sm_scale: float | None = None,
        final_norm_use_scale: bool = False,
        final_softcap: float | None = 50.,
        final_logit_scale: float = 1.,
        rngs: nnx.Rngs | int,
    ):
        if isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
            
        super().__init__(
            num_layers,
            embed_dim,
            hidden_dim,
            num_heads,
            activation,
            use_swiglu,
            dtype=dtype,
            encoder_input_dim=encoder_input_dim,
            softcap=softcap,
            qk_norm=qk_norm,
            sm_scale=sm_scale,
            final_norm=final_norm,
            final_norm_use_scale=final_norm_use_scale,
            final_softcap=final_softcap,
            final_logit_scale=final_logit_scale,
            rngs=rngs,
        )
        
        encoder_embed_dim, _ = embed_dim
        self.depot_bias = nnx.Param(
            nnx.initializers.normal(1e-6)(rngs.params(), [encoder_embed_dim], jnp.float32)
        )

    def encode(self, raw_features: jax.Array, attn_options: dict[str, Any] = {}):
        attn_options.update(softcap=self.softcap, sm_scale=self.sm_scale)
        depot_bias = self.depot_bias.value.astype(self.dtype)
        features = self.init_proj(raw_features)
        features = features.at[:, 0].add(depot_bias[None])
        features = self.encoder(features, attn_options=attn_options)
        return features
