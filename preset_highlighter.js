import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("[YimoH3 v3.0.0] Extension loaded");

const WORKFLOW_MODE_VISIBILITY = {
    "text": {
        show: ["prompt", "negative_prompt", "width", "height", "length", "strict_prompt_tags", "show_all_params"],
        hide: ["first_frame", "last_frame", "keyframes", "keyframe_positions", "source_audio", "override_audio", "ref_images", "ref_videos", "style_audios", "audio_denoise_strength", "identity_image_indices", "ref_image_size", "reference_video_policy", "ref_video_start_frame", "audio_offset_frames", "clip_video_grayout", "ref_video_preprocessing", "auto_resolution", "reference_strength"],
    },
    "first_frame": {
        show: ["prompt", "negative_prompt", "first_frame", "keyframes", "keyframe_positions", "width", "height", "length", "auto_resolution", "identity_image_indices", "reference_strength", "strict_prompt_tags", "show_all_params", "clip_video_grayout", "ref_video_preprocessing"],
        hide: ["last_frame", "source_audio", "override_audio", "ref_images", "ref_videos", "style_audios", "audio_denoise_strength", "audio_offset_frames"],
    },
    "last_frame": {
        show: ["prompt", "negative_prompt", "last_frame", "keyframes", "keyframe_positions", "width", "height", "length", "auto_resolution", "identity_image_indices", "reference_strength", "strict_prompt_tags", "show_all_params", "clip_video_grayout", "ref_video_preprocessing"],
        hide: ["first_frame", "source_audio", "override_audio", "ref_images", "ref_videos", "style_audios", "audio_denoise_strength", "audio_offset_frames"],
    },
    "first_last_frame": {
        show: ["prompt", "negative_prompt", "first_frame", "last_frame", "keyframes", "keyframe_positions", "width", "height", "length", "auto_resolution", "identity_image_indices", "reference_strength", "strict_prompt_tags", "show_all_params", "clip_video_grayout", "ref_video_preprocessing"],
        hide: ["source_audio", "override_audio", "ref_images", "ref_videos", "style_audios", "audio_denoise_strength", "audio_offset_frames"],
    },
    "references": {
        show: ["prompt", "negative_prompt", "ref_images", "ref_videos", "style_audios", "width", "height", "length", "ref_image_size", "reference_video_policy", "reference_strength", "identity_image_indices", "ref_video_start_frame", "strict_prompt_tags", "show_all_params", "clip_video_grayout", "ref_video_preprocessing"],
        hide: ["first_frame", "last_frame", "keyframes", "keyframe_positions", "auto_resolution"],
    },
    "hybrid": {
        show: ["prompt", "negative_prompt", "first_frame", "last_frame", "keyframes", "keyframe_positions", "ref_images", "ref_videos", "style_audios", "width", "height", "length", "auto_resolution", "identity_image_indices", "reference_strength", "ref_image_size", "reference_video_policy", "ref_video_start_frame", "strict_prompt_tags", "show_all_params", "clip_video_grayout", "ref_video_preprocessing"],
        hide: [],
    },
};

const AUDIO_POLICY_VISIBILITY = {
    "keep_source": { show: ["audio_offset_frames"], hide: ["audio_denoise_strength"] },
    "remix_source": { show: ["audio_denoise_strength", "audio_offset_frames"], hide: [] },
    "reference_only": { show: ["audio_offset_frames"], hide: ["audio_denoise_strength"] },
    "generate_new": { show: [], hide: ["audio_denoise_strength", "audio_offset_frames"] },
};

const ALWAYS_VISIBLE = ["clip", "video_vae", "audio_vae", "workflow_mode", "audio_policy", "show_all_params"];

// v2.5.8: 提示词格式化节点 —— 三段式 / 六段式的可见性
// 每个字段对应一组 widget：[标题 Combo, 内容 String]
const COMPOSER_STRUCTURE_VISIBILITY = {
    "three_section": {
        show: [
            "_header_integrated_multimodal_description",
            "integrated_multimodal_description",
            "_header_overall_soundscape",
            "overall_soundscape",
            "_header_non_diegetic_music",
            "non_diegetic_music",
            "_header_global_suffix",
            "global_suffix",
        ],
        hide: [
            "_header_subject_definitions",
            "subject_definitions",
            "_header_summary",
            "summary",
            "_header_retention_analysis",
            "retention_analysis",
            "_header_detailed_description",
            "detailed_description",
        ],
    },
    "six_section": {
        show: [
            "_header_subject_definitions",
            "subject_definitions",
            "_header_summary",
            "summary",
            "_header_retention_analysis",
            "retention_analysis",
            "_header_detailed_description",
            "detailed_description",
            "_header_overall_soundscape",
            "overall_soundscape",
            "_header_non_diegetic_music",
            "non_diegetic_music",
            "_header_global_suffix",
            "global_suffix",
        ],
        hide: [
            "_header_integrated_multimodal_description",
            "integrated_multimodal_description",
        ],
    },
};

const COMPOSER_ALWAYS_VISIBLE = ["structure"];

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    if (visible) {
        widget.type = widget.origType || widget.type || "number";
        widget.hidden = false;
        if (widget.options) widget.options.hidden = false;
    } else {
        if (!widget.origType && widget.type && widget.type !== "hidden") {
            widget.origType = widget.type;
        }
        widget.type = "hidden";
        widget.hidden = true;
        if (widget.options) widget.options.hidden = true;
    }
}

function updateVisibility(node) {
    const modeWidget = node.widgets.find(w => w.name === "workflow_mode");
    const policyWidget = node.widgets.find(w => w.name === "audio_policy");
    const showAllWidget = node.widgets.find(w => w.name === "show_all_params");
    if (!modeWidget || !policyWidget) return;

    const showAll = showAllWidget ? showAllWidget.value : false;
    const currentMode = modeWidget.value;

    const modesWithoutAudio = ["text", "first_frame", "last_frame", "first_last_frame"];
    const policiesNeedingSource = ["keep_source", "remix_source", "reference_only"];
    const currentPolicy = policyWidget.value;

    if (modesWithoutAudio.includes(currentMode) && policiesNeedingSource.includes(currentPolicy)) {
        policyWidget.inputEl?.classList?.add("yimo-warning");
    } else {
        policyWidget.inputEl?.classList?.remove("yimo-warning");
    }

    if (showAll) {
        for (const w of node.widgets) {
            if (!w.name) continue;
            if (ALWAYS_VISIBLE.includes(w.name)) continue;
            if (w.comfyInputDef && w.comfyInputDef.hidden) continue;
            setWidgetVisible(w, true);
        }
    } else {
        const modeCfg = WORKFLOW_MODE_VISIBILITY[currentMode] || WORKFLOW_MODE_VISIBILITY["text"];
        const policyCfg = AUDIO_POLICY_VISIBILITY[policyWidget.value] || AUDIO_POLICY_VISIBILITY["generate_new"];

        const showSet = new Set([...modeCfg.show, ...policyCfg.show]);
        const hideSet = new Set([...modeCfg.hide, ...policyCfg.hide]);

        for (const w of node.widgets) {
            if (!w.name) continue;
            if (ALWAYS_VISIBLE.includes(w.name)) continue;
            if (w.comfyInputDef && w.comfyInputDef.hidden) continue;

            if (hideSet.has(w.name)) {
                setWidgetVisible(w, false);
            } else if (showSet.has(w.name)) {
                setWidgetVisible(w, true);
            } else {
                setWidgetVisible(w, false);
            }
        }
    }

    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}

function updateComposerVisibility(node) {
    const structureWidget = node.widgets.find(w => w.name === "structure");
    if (!structureWidget) return;

    const currentStructure = structureWidget.value;
    const cfg = COMPOSER_STRUCTURE_VISIBILITY[currentStructure] || COMPOSER_STRUCTURE_VISIBILITY["three_section"];

    const showSet = new Set(cfg.show);
    const hideSet = new Set(cfg.hide);

    for (const w of node.widgets) {
        if (!w.name) continue;
        if (COMPOSER_ALWAYS_VISIBLE.includes(w.name)) continue;
        if (w.comfyInputDef && w.comfyInputDef.hidden) continue;

        if (hideSet.has(w.name)) {
            setWidgetVisible(w, false);
        } else if (showSet.has(w.name)) {
            setWidgetVisible(w, true);
        }
    }

    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}

function isTargetNode(node) {
    return (node.type || node.comfyClass || "") === "YimoH3Conditioning";
}

function isSequenceSamplerNode(node) {
    return (node.type || node.comfyClass || "") === "YimoH3SequenceSampler";
}

function isSingleSamplerNode(node) {
    return (node.type || node.comfyClass || "") === "YimoH3SingleSampler";
}

function isPromptComposerNode(node) {
    return (node.type || node.comfyClass || "") === "YimoH3PromptComposer";
}

function isSamplerNode(node) {
    return isSequenceSamplerNode(node) || isSingleSamplerNode(node);
}

function updateSequenceSamplerOutputs(node) {
    if (!node.inputs || !node.outputs) return;
    let visibleSegmentInputs = 0;
    for (let i = 0; i < node.inputs.length; i++) {
        const inp = node.inputs[i];
        if (inp && inp.name && inp.name.startsWith("segment_")) {
            visibleSegmentInputs++;
        }
    }
    const showUpTo = Math.max(1, visibleSegmentInputs - 1);
    let changed = false;
    for (let i = 2; i < node.outputs.length; i++) {
        const out = node.outputs[i];
        if (!out) continue;
        const segIndex = i - 2;
        const shouldHide = segIndex > showUpTo;
        if (!!out.hidden !== shouldHide) {
            out.hidden = shouldHide;
            changed = true;
        }
    }
    if (changed) {
        node.setSize(node.computeSize());
        node.setDirtyCanvas(true, true);
    }
}

function setupExecutionCleanup() {
    if (typeof api === "undefined" || !api.addEventListener) return;

    api.addEventListener("execution_start", () => {
        for (const node of app.graph._nodes) {
            if (isSamplerNode(node)) {
                node.progress = undefined;
                if (node.imgs) {
                    node.imgs = [];
                }
                node.setDirtyCanvas(true, true);
            }
        }
    });
}

app.registerExtension({
    name: "Yimo.MiniMaxH3.v2",

    async setup() {
        setupExecutionCleanup();
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "YimoH3Conditioning") {
            const onWidgetChanged = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function(name, value, old_value) {
                if (name === "workflow_mode" || name === "audio_policy" || name === "show_all_params") {
                    updateVisibility(this);
                }
                return onWidgetChanged?.apply(this, arguments);
            };
        }

        if (nodeData.name === "YimoH3SequenceSampler") {
            const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function(type, slot, isConnected, link, ioSlot) {
                const r = origOnConnectionsChange ? origOnConnectionsChange.apply(this, arguments) : undefined;
                if (type === LiteGraph.INPUT && this.inputs && this.inputs[slot]) {
                    const input = this.inputs[slot];
                    if (input.name && input.name.startsWith("segment_")) {
                        updateSequenceSamplerOutputs(this);
                    }
                }
                return r;
            };
        }

        if (nodeData.name === "YimoH3PromptComposer") {
            const onWidgetChanged = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function(name, value, old_value) {
                if (name === "structure") {
                    updateComposerVisibility(this);
                }
                return onWidgetChanged?.apply(this, arguments);
            };
        }
    },

    async nodeCreated(node) {
        if (isTargetNode(node)) {
            for (const w of node.widgets) {
                if (w.type && w.type !== "hidden") {
                    w.origType = w.type;
                }
            }
            setTimeout(() => updateVisibility(node), 50);
            const origOnConfigure = node.onConfigure;
            node.onConfigure = function(o) {
                const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
                setTimeout(() => updateVisibility(this), 50);
                return r;
            };
        }

        if (isSequenceSamplerNode(node)) {
            for (let i = 2; i < node.outputs.length; i++) {
                if (node.outputs[i]) node.outputs[i].hidden = true;
            }
            updateSequenceSamplerOutputs(node);
            setTimeout(() => updateSequenceSamplerOutputs(node), 100);

            const origOnConfigure = node.onConfigure;
            node.onConfigure = function(o) {
                const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
                setTimeout(() => updateSequenceSamplerOutputs(this), 100);
                return r;
            };
        }

        if (isPromptComposerNode(node)) {
            for (const w of node.widgets) {
                if (w.type && w.type !== "hidden") {
                    w.origType = w.type;
                }
            }
            setTimeout(() => updateComposerVisibility(node), 50);
            const origOnConfigure = node.onConfigure;
            node.onConfigure = function(o) {
                const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
                setTimeout(() => updateComposerVisibility(this), 50);
                return r;
            };
        }
    }
});