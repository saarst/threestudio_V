import json
import os
from dataclasses import dataclass
from threestudio.utils.misc import barrier, cleanup, get_rank

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from transformers import Gemma2PreTrainedModel, GemmaTokenizer, GemmaTokenizerFast
from diffusers import SanaPipeline

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessor, hash_prompt, PromptProcessorOutput, DirectionConfig, shifted_expotional_decay, shift_azimuth_deg
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *

@dataclass
class SANAPromptProcessorOutput():
    text_embeddings: Float[Tensor, "N Nf"]
    text_attention_mask: Float[Tensor, "N Nf"]
    uncond_text_embeddings: Float[Tensor, "N Nf"]
    uncond_text_attention_mask: Float[Tensor, "N Nf"]
    text_embeddings_vd: Float[Tensor, "Nv N Nf"]
    text_attention_mask_vd: Float[Tensor, "Nv N Nf"]
    uncond_text_embeddings_vd: Float[Tensor, "Nv N Nf"]
    uncond_text_attention_mask_vd: Float[Tensor, "Nv N Nf"]
    directions: List[DirectionConfig]
    direction2idx: Dict[str, int]
    use_perp_neg: bool
    perp_neg_f_sb: Tuple[float, float, float]
    perp_neg_f_fsb: Tuple[float, float, float]
    perp_neg_f_fs: Tuple[float, float, float]
    perp_neg_f_sf: Tuple[float, float, float]
    prompt: str
    prompts_vd: List[str]

    def get_text_embeddings(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        view_dependent_prompting: bool = True,
    ) -> Float[Tensor, "BB N Nf"]:
        batch_size = elevation.shape[0]

        if view_dependent_prompting:
            # Get direction
            direction_idx = torch.zeros_like(elevation, dtype=torch.long)
            for d in self.directions:
                direction_idx[
                    d.condition(elevation, azimuth, camera_distances)
                ] = self.direction2idx[d.name]

            # Get text embeddings
            text_embeddings = self.text_embeddings_vd[direction_idx]  # type: ignore
            text_attention_masks = self.text_attention_mask_vd[direction_idx]  # type: ignore
            uncond_text_embeddings = self.uncond_text_embeddings_vd[direction_idx]  # type: ignore
            uncond_attention_masks = self.uncond_text_attention_mask_vd[direction_idx]  # type: ignore
        else:
            text_embeddings = self.text_embeddings.expand(batch_size, -1, -1)  # type: ignore
            text_attention_masks = self.text_attention_mask.expand(batch_size, -1, -1)
            uncond_text_embeddings = self.uncond_text_embeddings.expand(  # type: ignore
                batch_size, -1, -1
            )
            uncond_attention_masks = self.uncond_text_attention_mask.expand(  # type: ignore
                batch_size, -1, -1
            )

        # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
        return torch.cat([text_embeddings, uncond_text_embeddings], dim=0), torch.cat([text_attention_masks, uncond_attention_masks], dim=0)

    def get_text_embeddings_perp_neg(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        view_dependent_prompting: bool = True,
    ) -> Tuple[Float[Tensor, "BBBB N Nf"], Float[Tensor, "B 2"]]:
        assert (
            view_dependent_prompting
        ), "Perp-Neg only works with view-dependent prompting"

        batch_size = elevation.shape[0]

        direction_idx = torch.zeros_like(elevation, dtype=torch.long)
        for d in self.directions:
            direction_idx[
                d.condition(elevation, azimuth, camera_distances)
            ] = self.direction2idx[d.name]
        # 0 - side view
        # 1 - front view
        # 2 - back view
        # 3 - overhead view

        pos_text_embeddings = []
        pos_text_attention_masks = []
        neg_text_embeddings = []
        neg_text_attention_masks = []
        neg_guidance_weights = []
        uncond_text_embeddings = []
        uncond_text_attention_masks = []

        side_emb = self.text_embeddings_vd[0]
        side_attention_mask = self.text_attention_mask_vd[0]
        front_emb = self.text_embeddings_vd[1]
        front_attention_mask = self.text_attention_mask_vd[1]
        back_emb = self.text_embeddings_vd[2]
        back_attention_mask = self.text_attention_mask_vd[2]
        overhead_emb = self.text_embeddings_vd[3]
        overhead_attention_mask = self.text_attention_mask_vd[3]

        for idx, ele, azi, dis in zip(
            direction_idx, elevation, azimuth, camera_distances
        ):
            azi = shift_azimuth_deg(azi)  # to (-180, 180)
            uncond_text_embeddings.append(
                self.uncond_text_embeddings_vd[idx]
            )  # should be ""
            uncond_text_attention_masks.append(
                self.uncond_text_attention_mask_vd[idx]
            )
            if idx.item() == 3:  # overhead view
                pos_text_embeddings.append(overhead_emb)  # side view
                pos_text_attention_masks.append(overhead_attention_mask)
                # dummy
                neg_text_embeddings += [
                    self.uncond_text_embeddings_vd[idx],
                    self.uncond_text_embeddings_vd[idx],
                ]
                neg_text_attention_masks += [
                    self.uncond_text_attention_mask_vd[idx],
                    self.uncond_text_attention_mask_vd[idx],
                ]
                neg_guidance_weights += [0.0, 0.0]
            else:  # interpolating views
                if torch.abs(azi) < 90:
                    # front-side interpolation
                    # 0 - complete side, 1 - complete front
                    r_inter = 1 - torch.abs(azi) / 90
                    pos_text_embeddings.append(
                        r_inter * front_emb + (1 - r_inter) * side_emb
                    )
                    pos_text_attention_masks.append(
                        r_inter * front_attention_mask + (1 - r_inter) * side_attention_mask
                    )
                    neg_text_embeddings += [front_emb, side_emb]
                    neg_text_attention_masks += [front_attention_mask, side_attention_mask]
                    neg_guidance_weights += [
                        -shifted_expotional_decay(*self.perp_neg_f_fs, r_inter),
                        -shifted_expotional_decay(*self.perp_neg_f_sf, 1 - r_inter),
                    ]
                else:
                    # side-back interpolation
                    # 0 - complete back, 1 - complete side
                    r_inter = 2.0 - torch.abs(azi) / 90
                    pos_text_embeddings.append(
                        r_inter * side_emb + (1 - r_inter) * back_emb
                    )
                    pos_text_attention_masks.append(
                        r_inter * side_attention_mask + (1 - r_inter) * back_attention_mask
                    )
                    neg_text_embeddings += [side_emb, front_emb]
                    neg_text_attention_masks += [side_attention_mask, front_attention_mask]
                    neg_guidance_weights += [
                        -shifted_expotional_decay(*self.perp_neg_f_sb, r_inter),
                        -shifted_expotional_decay(*self.perp_neg_f_fsb, r_inter),
                    ]

        text_embeddings = torch.cat(
            [
                torch.stack(pos_text_embeddings, dim=0),
                torch.stack(uncond_text_embeddings, dim=0),
                torch.stack(neg_text_embeddings, dim=0),
            ],
            dim=0,
        )
        text_attention_masks = torch.cat(
            [
                torch.stack(pos_text_attention_masks, dim=0),
                torch.stack(uncond_text_attention_masks, dim=0),
                torch.stack(neg_text_attention_masks, dim=0),
            ],
            dim=0,
        )

        return text_embeddings, text_attention_masks, torch.as_tensor(
            neg_guidance_weights, device=elevation.device
        ).reshape(batch_size, 2)



@threestudio.register("sana-prompt-processor")
class SANAPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pass

    cfg: Config

    ### these functions are unused, kept for debugging ###
    def configure_text_encoder(self) -> None:
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
        # )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # self.text_encoder = Gemma2PreTrainedModel.from_pretrained(
        #     self.cfg.pretrained_model_name_or_path, subfolder="text_encoder"
        # ).to(self.device)

        # for p in self.text_encoder.parameters():
        #     p.requires_grad_(False)
        
        self.pipe = SanaPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            variant="fp16",
            torch_dtype=torch.float16,
        ).to(self.device)
        

        self.pipe.vae.to(torch.bfloat16)
        self.pipe.text_encoder.to(torch.bfloat16)

        del self.pipe.vae
        del self.pipe.transformer
        del self.pipe.scheduler
        cleanup()

        # Create model
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.text_encoder
        cleanup()

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B 77 768"], Float[Tensor, "B 77 768"]]:

        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = ...
        self.pipe.encode_prompt(prompt, negative_prompt, do_classifier_free_guidance=True)

        # if isinstance(prompt, str):
        #     prompt = [prompt]
        # if isinstance(negative_prompt, str):
        #     negative_prompt = [negative_prompt]

        # # Tokenize text and get embeddings
        # tokens = self.tokenizer(
        #     prompt,
        #     padding="max_length",
        #     max_length=self.tokenizer.model_max_length,
        #     return_tensors="pt",
        # )
        # uncond_tokens = self.tokenizer(
        #     negative_prompt,
        #     padding="max_length",
        #     max_length=self.tokenizer.model_max_length,
        #     return_tensors="pt",
        # )

        # with torch.no_grad():
        #     text_embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0]
        #     uncond_text_embeddings = self.text_encoder(
        #         uncond_tokens.input_ids.to(self.device)
        #     )[0]

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    def load_text_embeddings(self):
        # synchronize, to ensure the text embeddings have been computed and saved to cache
        barrier()
        text_embeddings = self.load_from_cache(self.prompt)
        self.text_embeddings = text_embeddings['embedding'][None, ...]
        self.text_attention_mask = text_embeddings['attention_mask'][None, ...]

        uncond_text_embeddings = self.load_from_cache(self.negative_prompt)
        self.uncond_text_embeddings = uncond_text_embeddings['embedding'][None, ...]
        self.uncond_text_attention_mask = uncond_text_embeddings['attention_mask'][None, ...]


        self.text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt)['embedding'] for prompt in self.prompts_vd], dim=0
        )
        self.text_attention_mask_vd = torch.stack(
            [self.load_from_cache(prompt)['attention_mask'] for prompt in self.prompts_vd], dim=0
        )

        self.uncond_text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt)['embedding'] for prompt in self.negative_prompts_vd], dim=0
        )
        self.uncond_text_attention_mask_vd = torch.stack(
            [self.load_from_cache(prompt)['attention_mask'] for prompt in self.negative_prompts_vd], dim=0
        )
        threestudio.debug(f"Loaded text embeddings.")

    def __call__(self) -> SANAPromptProcessorOutput:
        return SANAPromptProcessorOutput(
            text_embeddings=self.text_embeddings,
            uncond_text_embeddings=self.uncond_text_embeddings,
            text_attention_mask=self.text_attention_mask,
            uncond_text_attention_mask=self.uncond_text_attention_mask,
            prompt=self.prompt,
            text_embeddings_vd=self.text_embeddings_vd,
            text_attention_mask_vd=self.text_attention_mask_vd,
            uncond_text_embeddings_vd=self.uncond_text_embeddings_vd,
            uncond_text_attention_mask_vd=self.uncond_text_attention_mask_vd,
            prompts_vd=self.prompts_vd,
            directions=self.directions,
            direction2idx=self.direction2idx,
            use_perp_neg=self.cfg.use_perp_neg,
            perp_neg_f_sb=self.cfg.perp_neg_f_sb,
            perp_neg_f_fsb=self.cfg.perp_neg_f_fsb,
            perp_neg_f_fs=self.cfg.perp_neg_f_fs,
            perp_neg_f_sf=self.cfg.perp_neg_f_sf,
        )

    ###

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # tokenizer = AutoTokenizer.from_pretrained(
        #     pretrained_model_name_or_path, subfolder="tokenizer"
        # )
        # text_encoder = Gemma2PreTrainedModel.from_pretrained(
        #     pretrained_model_name_or_path,
        #     subfolder="text_encoder",
        #     device_map="auto",
        # )
        pipe = SanaPipeline.from_pretrained(
            pretrained_model_name_or_path,
            variant="fp16",
            torch_dtype=torch.float16,
        ).to("cuda")
        
        pipe.vae.to(torch.bfloat16)
        pipe.text_encoder.to(torch.bfloat16)


        with torch.no_grad():
            output = pipe.encode_prompt(prompts, do_classifier_free_guidance=False)
            
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = output
        for prompt, embedding, attention_mask in zip(prompts, prompt_embeds, prompt_attention_mask):
            torch.save(
                {
                    "embedding": embedding,
                    "attention_mask": attention_mask,
                },
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )

        del pipe
