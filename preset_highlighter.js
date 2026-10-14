import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("[YimoH3 v5.0.0] Extension loaded");

const WORKFLOW_MODE_VISIBILITY = {
    "text": {
        show: ["prompt", "negative_prompt", "width", "height", "length",
               "strict_prompt_tags", "show_all_params",
               "reference_retention", "reference_retention_scale",
               "ref_curve", "ref_curve_direction_custom", "ref_curve_shape_custom"],
        hide: ["first_frame", "last_frame", "keyframes", "keyframe_positions",
               "source_audio", "override_audio", "ref_images", "ref_videos",
               "style_audios", "audio_denoise_strength", "identity_image_indices",
               "ref_image_size", "reference_video_policy", "ref_video_start_frame",
               "audio_offset_frames"],
    },
    "first_frame": {
        show: ["prompt", "negative_prompt", "first_frame", "keyframes", "keyframe_positions",
               "width", "height", "length", "identity_image_indices",
               "reference_retention", "reference_retention_scale",
               "ref_curve", "ref_curve_direction_custom", "ref_curve_shape_custom",
               "strict_prompt_tags", "show_all_params"],
        hide: ["last_frame", "source_audio", "override_audio", "ref_images", "ref_videos",
               "style_audios", "audio_denoise_strength", "audio_offset_frames"],
    },
    "last_frame": {
        show: ["prompt", "negative_prompt", "last_frame", "keyframes", "keyframe_positions",
               "width", "height", "length", "identity_image_indices",
               "reference_retention", "reference_retention_scale",
               "ref_curve", "ref_curve_direction_custom", "ref_curve_shape_custom",
               "strict_prompt_tags", "show_all_params"],
        hide: ["first_frame", "source_audio", "override_audio", "ref_images", "ref_videos",
               "style_audios", "audio_denoise_strength", "audio_offset_frames"],
    },
    "first_last_frame": {
        show: ["prompt", "negative_prompt", "first_frame", "last_frame",
               "keyframes", "keyframe_positions",
               "width", "height", "length", "identity_image_indices",
               "reference_retention", "reference_retention_scale",
               "ref_curve", "ref_curve_direction_custom", "ref_curve_shape_custom",
               "strict_prompt_tags", "show_all_params"],
        hide: ["source_audio", "override_audio", "ref_images", "ref_videos",
               "style_audios", "audio_denoise_strength", "audio_offset_frames"],
    },
    "references": {
        show: ["prompt", "negative_prompt", "ref_images", "ref_videos", "style_audios",
               "width", "height", "length", "ref_image_size", "reference_video_policy",
               "identity_image_indices", "ref_video_start_frame",
               "reference_retention", "reference_retention_scale",
               "ref_curve", "ref_curve_direction_custom", "ref_curve_shape_custom",
               "strict_prompt_tags", "show_all_params"],
        hide: ["first_frame", "last_frame", "keyframes", "keyframe_positions"],
    },
    "hybrid": {
        show: ["prompt", "negative_prompt", "first_frame", "last_frame",
               "keyframes", "keyframe_positions",
               "ref_images", "ref_videos", "style_audios",
               "width", "height", "length", "identity_image_indices",
               "ref_image_size", "reference_video_policy", "ref_video_start_frame",
               "reference_retention", "reference_retention_scale",
               "ref_curve", "ref_curve_direction_custom", "ref_curve_shape_custom",
               "strict_prompt_tags", "show_all_params"],
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


function updateConditioningCurveVisibility(node) {
    const presetWidget = node.widgets.find(w => w.name === "ref_curve");
    if (!presetWidget) return;
    const isCustom = presetWidget.value === "custom";
    for (const name of ["ref_curve_direction_custom", "ref_curve_shape_custom"]) {
        const w = node.widgets.find(x => x.name === name);
        if (w) setWidgetVisible(w, isCustom);
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

    updateConditioningCurveVisibility(node);

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

function isSamplerNode(node) {
    return isSequenceSamplerNode(node) || isSingleSamplerNode(node);
}


// =============================================================================
// 序列采样器：输出端「分段 AV 潜变量」按已连接的 segment_# 数量显示
// =============================================================================

const _SEGMENT_OUTPUT_RE = /^分段(\d+)AV潜变量$/;

function _countConnectedSegmentInputs(node) {
    if (!node.inputs) return 0;
    let connected = 0;
    for (let i = 0; i < node.inputs.length; i++) {
        const inp = node.inputs[i];
        if (!inp || !inp.name) continue;
        if (!inp.name.startsWith("segment_")) continue;
        if (inp.link != null) {
            connected++;
        }
    }
    return connected;
}

function _collectSegmentOutputs(node) {
    const result = [];
    if (!node.outputs) return result;
    for (let i = 0; i < node.outputs.length; i++) {
        const out = node.outputs[i];
        if (!out || !out.name) continue;
        const m = _SEGMENT_OUTPUT_RE.exec(out.name);
        if (!m) continue;
        const idx = parseInt(m[1], 10);
        if (!Number.isFinite(idx)) continue;
        result.push({ out, idx });
    }
    return result;
}

function updateSequenceSamplerOutputs(node) {
    if (!isSequenceSamplerNode(node)) return;
    if (!node.inputs || !node.outputs) return;

    const segmentOutputs = _collectSegmentOutputs(node);
    if (segmentOutputs.length === 0) return;

    const connectedInputs = _countConnectedSegmentInputs(node);

    const showUpTo = Math.max(0, connectedInputs - 1);

    let changed = false;
    for (const { out, idx } of segmentOutputs) {
        const shouldHide = idx > showUpTo;
        if (!!out.hidden !== shouldHide) {
            out.hidden = shouldHide;
            changed = true;
        }
    }

    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}

function scheduleSequenceSamplerOutputs(node) {
    if (!isSequenceSamplerNode(node)) return;
    setTimeout(() => {
        try {
            updateSequenceSamplerOutputs(node);
        } catch (e) {
            console.warn("[YimoH3] updateSequenceSamplerOutputs failed:", e);
        }
    }, 30);
    setTimeout(() => {
        try {
            updateSequenceSamplerOutputs(node);
        } catch (e) {
            console.warn("[YimoH3] updateSequenceSamplerOutputs failed (retry):", e);
        }
    }, 200);
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
    name: "Yimo.MiniMaxH3.v5.0.0",

    async setup() {
        setupExecutionCleanup();
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "YimoH3Conditioning") {
            const onWidgetChanged = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function(name, value, old_value) {
                if (name === "workflow_mode" || name === "audio_policy"
                    || name === "show_all_params" || name === "ref_curve") {
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
                        scheduleSequenceSamplerOutputs(this);
                    }
                }
                return r;
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
            for (const { out } of _collectSegmentOutputs(node)) {
                out.hidden = true;
            }
            scheduleSequenceSamplerOutputs(node);

            const origOnConfigure = node.onConfigure;
            node.onConfigure = function(o) {
                const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
                scheduleSequenceSamplerOutputs(this);
                return r;
            };
        }
    }
});