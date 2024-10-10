<script setup lang="ts">
import { isDeepEqual, isUserInDarkMode } from "@/services/utils";
import { View as VegaView } from "vega";
import embed, { type Config, type VisualizationSpec } from "vega-embed";
import { onBeforeUnmount, onMounted, ref, watch } from "vue";

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
        theme: isUserInDarkMode() ? "dark" : undefined,
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

watch(
  () => props.spec,
  async (newSpec, oldSpec) => {
    if (isDeepEqual(newSpec, oldSpec)) {
      return;
    }
    // Refresh view
    // TODO: This perhaps could be done in a more fine-grained way
    const r = await embed(
      container.value!,
      {
        width: container.value?.clientWidth || 0,
        ...newSpec,
      },
      {
        theme: isUserInDarkMode() ? "dark" : undefined,
        config: vegaConfig,
        actions: false,
      }
    );
    vegaView = r.view;
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
