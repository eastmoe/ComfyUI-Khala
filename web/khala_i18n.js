const { app } = window.comfyAPI.app;

const LOCALE_URL = "/comfy-khala/locale/zh-sn/nodes.json";
let cachedText = null;

function currentLocale() {
  const configured =
    app.ui?.settings?.getSettingValue?.("Comfy.Locale") ??
    app.ui?.settings?.getSettingValue?.("ComfyUI.Locale") ??
    navigator.language;
  return String(configured || "en").toLowerCase();
}

async function loadText() {
  if (cachedText) return cachedText;
  if (!currentLocale().startsWith("zh")) return null;
  try {
    const response = await fetch(LOCALE_URL);
    if (response.ok) {
      cachedText = await response.json();
      return cachedText;
    }
  } catch (error) {
    console.warn("[Comfy-Khala] Failed to load zh-sn locale:", error);
  }
  return null;
}

function chainCallback(target, name, callback) {
  const original = target[name];
  target[name] = function (...args) {
    const result = original?.apply(this, args);
    callback.apply(this, args);
    return result;
  };
}

function applyLabels(node, text) {
  if (!text) return;
  const nodeText = text.nodes?.[node.constructor?.comfyClass] ?? text.nodes?.[node.type];
  if (!nodeText) return;

  node.title = nodeText.title ?? node.title;

  const inputText = nodeText.inputs ?? {};
  for (const input of node.inputs ?? []) {
    const item = inputText[input.name] ?? inputText[input.label];
    if (!item) continue;
    input.label = item.label;
    input.localized_name = item.label;
  }

  const outputText = nodeText.outputs ?? {};
  for (const output of node.outputs ?? []) {
    const item = outputText[output.name] ?? outputText[output.label];
    if (!item) continue;
    output.label = item.label;
    output.localized_name = item.label;
  }

  for (const widget of node.widgets ?? []) {
    const item = inputText[widget.name] ?? inputText[widget.label];
    if (!item) continue;
    widget.label = item.label;
    widget.localized_name = item.label;
    widget.options = widget.options ?? {};
    widget.options.tooltip = item.tooltip ?? item.comment ?? widget.options.tooltip;
  }

  app.graph?.setDirtyCanvas(true, true);
}

function patchNodeData(nodeData, nodeText) {
  nodeData.display_name = nodeText.title ?? nodeData.display_name;
  nodeData.description = nodeText.description ?? nodeData.description;
  nodeData.output_name = Object.values(nodeText.outputs ?? {}).map((item) => item.label);
  nodeData.output_tooltips = Object.values(nodeText.outputs ?? {}).map((item) => item.tooltip ?? item.comment ?? "");

  for (const section of ["required", "optional"]) {
    const inputs = nodeData.input?.[section];
    if (!inputs) continue;
    for (const [name, spec] of Object.entries(inputs)) {
      const item = nodeText.inputs?.[name];
      if (!item || !Array.isArray(spec)) continue;
      const options = spec[1] ?? {};
      options.display_name = item.label;
      options.label = item.label;
      options.localized_name = item.label;
      options.tooltip = item.tooltip ?? item.comment ?? options.tooltip;
      spec[1] = options;
    }
  }
}

app.registerExtension({
  name: "eastmoe.ComfyKhala.i18n",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const text = await loadText();
    const nodeText = text?.nodes?.[nodeData?.name];
    if (!nodeText) return;

    patchNodeData(nodeData, nodeText);

    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      applyLabels(this, text);
    });

    chainCallback(nodeType.prototype, "onConfigure", function () {
      applyLabels(this, text);
    });
  },

  async nodeCreated(node) {
    const text = await loadText();
    applyLabels(node, text);
  },
});
