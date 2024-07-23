<script setup lang="ts">
import embed, { type VisualizationSpec } from "vega-embed";
import { onMounted, ref } from "vue";

const props = defineProps<{ spec: VisualizationSpec }>();

const container = ref<HTMLDivElement>();

onMounted(async () => {
  if (container.value) {
    await embed(container.value, props.spec);
  }
});
</script>

<template>
  <div ref="container"></div>
</template>
