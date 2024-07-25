<script setup lang="ts">
import { computed } from "vue";

import DataFrameWidget from "@/components/DataFrameWidget.vue";
import DataStoreCard from "@/components/DataStoreCard.vue";
import MarkdownWidget from "@/components/MarkdownWidget.vue";
import VegaWidget from "@/components/VegaWidget.vue";
import { useCanvasStore, type KeyLayoutSize } from "@/stores/canvas";

const canvasStore = useCanvasStore();
const items = computed(() => {
  const dataStore = canvasStore.dataStore;
  const r: { [key: string]: any } = {};
  if (dataStore) {
    for (const k of canvasStore.displayedKeys) {
      const item = dataStore.get(k);
      r[k] = { type: item.type, data: item.data };
    }
  }
  return r;
});

function onLayoutChange(key: string, size: KeyLayoutSize) {
  canvasStore.setKeyLayoutSize(key, size);
}

function onCardRemoved(key: string) {
  canvasStore.hideKey(key);
}
</script>

<template>
  <div class="canvas">
    <DataStoreCard
      v-for="(value, key) in items"
      :key="key"
      :title="key as string"
      :class="[canvasStore.layoutSizes[key] || 'large']"
      class="canvas-element"
      @layout-changed="onLayoutChange(key as string, $event)"
      @card-removed="onCardRemoved(key as string)"
    >
      <VegaWidget v-if="value.type === 'vega'" :spec="value.data" />
      <DataFrameWidget
        v-if="value.type === 'datatable'"
        :columns="value.data.columns"
        :data="value.data.data"
      />
      <MarkdownWidget
        v-if="
          [
            'boolean',
            'integer',
            'number',
            'string',
            'any',
            'array',
            'date',
            'datetime',
            'markdown',
          ].includes(value.type)
        "
        :source="value.data"
      />
      <div v-if="value.type === 'html'" v-html="value.data"></div>
    </DataStoreCard>
  </div>
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
