<script setup lang="ts">
import DataStoreCard from "@/components/DataStoreCard.vue";
import { useCanvasStore, type KeyLayoutSize } from "@/stores/canvas";

const canvasStore = useCanvasStore();

function onLayoutChange(key: string, size: KeyLayoutSize) {
  canvasStore.setKeyLayoutSize(key, size);
}

function onCardRemoved(key: string) {
  canvasStore.hideKey(key);
}
</script>

<template>
  <DataStoreCard
    v-for="key in canvasStore.displayedKeys"
    :key="key"
    :title="key"
    :class="[canvasStore.layoutSizes[key] || 'large']"
    class="canvas-element"
    @layout-changed="onLayoutChange(key, $event)"
    @card-removed="onCardRemoved(key)"
  >
    <div v-html="canvasStore.get(key).toString().substring(0, 100)"></div>
  </DataStoreCard>
</template>

<style scoped>
.canvas {
  display: grid;
  padding: var(--spacing-gap-normal);
  background-color: var(--background-color-normal);
  gap: var(--spacing-gap-normal);
  grid-template-columns: 1fr 1fr 1fr;
  transition: grid-column var(--transition-duration) var(--transition-easing);

  & .canvas-element {
    max-width: 100%;

    &.small {
      grid-column: span 1;
    }

    &.medium {
      grid-column: span 2;
    }

    &.large {
      grid-column: span 3;
    }
  }
}
</style>
