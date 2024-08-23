<script setup lang="ts">
import { View as VegaView } from "vega";
import embed, { type Config, type VisualizationSpec } from "vega-embed";
import { onBeforeUnmount, onMounted, ref } from "vue";

const props = defineProps<{ spec: VisualizationSpec }>();

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
const resizeObserver = new ResizeObserver(async () => {
  const w = container.value?.clientWidth || 0;
  await vegaView?.width(w).runAsync();
});

onMounted(async () => {
  if (container.value) {
    const r = await embed(
      container.value,
      {
        width: container.value?.clientWidth || 0,
        ...props.spec,
      },
      {
        theme: "dark",
        config: vegaConfig,
        actions: false,
      }
    );
    vegaView = r.view;
    resizeObserver.observe(container.value);
  }
});

onBeforeUnmount(() => {
  if (container.value) {
    resizeObserver.unobserve(container.value);
  }
  if (vegaView) {
    vegaView.finalize();
  }
});
</script>

<template>
  <div ref="container"></div>
</template>

<style scoped>
div {
  width: 100%;
}
</style>
