<script setup lang="ts">
import { ref, onMounted } from "vue";
import embed from "vega-embed";
import { type TopLevelSpec } from "vega-lite";

const props = defineProps<{ spec: TopLevelSpec }>();

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
