<script setup lang="ts">
import { addFrames, newPlot, purge, relayout, type Layout } from "plotly.js-dist-min";
import { onBeforeUnmount, onMounted, ref } from "vue";

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

const resizeObserver = new ResizeObserver(() => {
  if (container.value) {
    relayout(container.value, makeLayout());
  }
});

onMounted(async () => {
  if (container.value) {
    resizeObserver.observe(container.value);
    const plot = await newPlot(container.value, props.spec.data, makeLayout());
    if (props.spec.frames) {
      addFrames(plot, props.spec.frames);
    }
  }
});

onBeforeUnmount(() => {
  if (container.value) {
    resizeObserver.unobserve(container.value);
    purge(container.value);
  }
});
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
