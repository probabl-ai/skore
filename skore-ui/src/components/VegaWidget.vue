<script setup lang="ts">
import { View as VegaView } from "vega";
import embed, { type Config, type VisualizationSpec } from "vega-embed";
import { onBeforeUnmount, onMounted, ref, watch } from "vue";

import { isDeepEqual } from "@/services/utils";

const props = defineProps<{
  spec: VisualizationSpec;
  theme: string;
}>();

const container = ref<HTMLDivElement>();
const font = "GeistMono, monospace";
const vegaConfig: Config = {
  axis: { labelFont: font, titleFont: font },
  legend: { labelFont: font, titleFont: font },
  header: { labelFont: font, titleFont: font },
  mark: { font: font },
  title: { font: font, subtitleFont: font },
  background: "transparent",
};
let vegaView: VegaView | null = null;

async function makePlot(spec: VisualizationSpec) {
  const mySpec = { ...spec, width: "container" } as VisualizationSpec;
  const r = await embed(container.value!, mySpec, {
    theme: props.theme as any,
    config: vegaConfig,
    actions: false,
  });
  vegaView = r.view;
}

function replot() {
  makePlot(props.spec);
}

const preferredColorScheme = window.matchMedia("(prefers-color-scheme: dark)");

onMounted(async () => {
  if (container.value) {
    makePlot(props.spec);
    preferredColorScheme.addEventListener("change", replot);
  }
});

onBeforeUnmount(() => {
  if (vegaView) {
    vegaView.finalize();
  }
  preferredColorScheme.removeEventListener("change", replot);
});

watch(
  () => props.spec,
  async (newSpec, oldSpec) => {
    if (!isDeepEqual(newSpec, oldSpec)) {
      makePlot(newSpec);
    }
  }
);

watch(
  () => props.theme,
  () => {
    makePlot(props.spec);
  }
);
</script>

<template>
  <div ref="container"></div>
</template>

<style scoped>
div {
  width: 100%;
}
</style>
