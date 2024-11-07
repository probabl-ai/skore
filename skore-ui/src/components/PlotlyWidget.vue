<script setup lang="ts">
import { addFrames, react, purge, relayout, type Layout } from "plotly.js-dist-min";
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { isDeepEqual } from "@/services/utils";

const props = defineProps<{
  spec: { data: any; layout: any; frames: any };
}>();

const container = ref<HTMLDivElement>();

function makeLayout(): Partial<Layout> {
  return {
    ...props.spec.layout,
    width: container.value?.clientWidth,
    height: container.value?.clientHeight,
    font: {
      family: "GeistMono, monospace",
    },
  };
}

async function buildPlot(containerValue: HTMLDivElement, plotData: any) {
  const plot = await react(containerValue, plotData, makeLayout());
  if (props.spec.frames) {
    addFrames(plot, props.spec.frames);
  }
}

const resizeObserver = new ResizeObserver(() => {
  if (container.value) {
    relayout(container.value, makeLayout());
  }
});

onMounted(async () => {
  if (container.value) {
    resizeObserver.observe(container.value);
    buildPlot(container.value, props.spec.data);
  }
});

onBeforeUnmount(() => {
  if (container.value) {
    resizeObserver.unobserve(container.value);
    purge(container.value);
  }
});

watch(
  () => props.spec.data,
  async (newData, oldData) => {
    if (!isDeepEqual(newData, oldData)) {
      if (container.value) {
        buildPlot(container.value, newData);
      }
    }
  }
);
</script>

<template>
  <div class="plotly-widget" ref="container"></div>
</template>

<style scoped>
.plotly-widget {
  width: 100%;
}

/*
plotly "dark mode" fix
https://github.com/plotly/plotly.js/issues/2006
*/
@media (prefers-color-scheme: dark) {
  .plotly-widget {
    filter: invert(75%) hue-rotate(180deg);
  }
}
</style>
