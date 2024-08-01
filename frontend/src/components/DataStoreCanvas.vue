<script setup lang="ts">
import { computed } from "vue";

import DataFrameWidget from "@/components/DataFrameWidget.vue";
import DataStoreCard from "@/components/DataStoreCard.vue";
import HtmlSnippetWidget from "@/components/HtmlSnippetWidget.vue";
import ImageWidget from "@/components/ImageWidget.vue";
import MarkdownWidget from "@/components/MarkdownWidget.vue";
import VegaWidget from "@/components/VegaWidget.vue";
import { useCanvasStore, type KeyLayoutSize } from "@/stores/canvas";
import CrossValidationResultsWidget from "./CrossValidationResultsWidget.vue";

const canvasStore = useCanvasStore();
const items = computed(() => {
  const dataStore = canvasStore.dataStore;
  const r: { [key: string]: any } = {};
  if (dataStore) {
    for (const k of canvasStore.displayedKeys) {
      r[k] = dataStore.get(k);
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
        v-if="value.type === 'dataframe'"
        :columns="value.data.columns"
        :data="value.data.data"
      />
      <ImageWidget
        v-if="value.type === 'image'"
        :mime-type="value.data['mime-type']"
        :base64-src="value.data.data"
        :alt="key as string"
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
      <HtmlSnippetWidget v-if="value.type === 'html'" :src="value.data" />
      <CrossValidationResultsWidget
        v-if="value.type === 'cv_results'"
        :test_score_plot="value.data.test_score_plot.data"
        :cv_results_table="value.data.cv_results_table.data"
      />
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
