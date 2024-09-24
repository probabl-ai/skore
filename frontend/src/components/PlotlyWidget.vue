<script setup lang="ts">
import { newPlot, purge, relayout, type Layout } from "plotly.js-dist-min";
import { onBeforeUnmount, onMounted, ref } from "vue";

const props = defineProps<{
  spec: { data: any; layout: any };
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

onMounted(() => {
  if (container.value) {
    resizeObserver.observe(container.value);
    newPlot(container.value, props.spec.data, makeLayout());
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
