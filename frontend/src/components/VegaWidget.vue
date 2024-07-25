<script setup lang="ts">
import embed, { type Config, type VisualizationSpec } from "vega-embed";
import { onMounted, ref } from "vue";

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

onMounted(async () => {
  if (container.value) {
    await embed(container.value, props.spec, {
      theme: "dark",
      config: vegaConfig,
    });
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
