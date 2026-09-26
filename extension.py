from __future__ import annotations

import sys
import traceback

from comfy_api.latest import ComfyExtension, io

from .conditioning_node import YimoH3Conditioning
from .sequence_sampler_node import YimoH3SequenceSampler
from .single_sampler_node import YimoH3SingleSampler
from .concatenate_node import YimoH3ConcatenateSegments
from .face_restore_node import YimoH3FaceRestore
from .load_saved_latent_node import YimoH3LoadSavedLatent
from .prompt_composer_node import YimoH3PromptComposer
from .post_process_node import YimoH3PostProcessSplit, YimoH3PostProcessMerge
from .highres_resampler_node import YimoH3HighResResampler
from .pdd_two_pass_node import YimoH3PDDTwoPass


class YimoMiniMaxH3Extension(ComfyExtension):
    async def get_node_list(self):
        return [
            YimoH3Conditioning,
            YimoH3SequenceSampler,
            YimoH3SingleSampler,
            YimoH3ConcatenateSegments,
            YimoH3FaceRestore,
            YimoH3LoadSavedLatent,
            YimoH3PromptComposer,
            YimoH3PostProcessSplit,
            YimoH3PostProcessMerge,
            YimoH3HighResResampler,
            YimoH3PDDTwoPass,
        ]


def comfy_entrypoint():
    try:
        return YimoMiniMaxH3Extension()
    except Exception as e:
        print("=" * 70, file=sys.stderr)
        print("YimoH3 comfy_entrypoint FAILED:", e, file=sys.stderr)
        traceback.print_exc()
        print("=" * 70, file=sys.stderr)
        raise