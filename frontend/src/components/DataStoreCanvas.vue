<script setup lang="ts">
import { computed } from "vue";

import CrossValidationResultsWidget from "@/components/CrossValidationResultsWidget.vue";
import DataFrameWidget from "@/components/DataFrameWidget.vue";
import DataStoreCard from "@/components/DataStoreCard.vue";
import HtmlSnippetWidget from "@/components/HtmlSnippetWidget.vue";
import ImageWidget from "@/components/ImageWidget.vue";
import MarkdownWidget from "@/components/MarkdownWidget.vue";
import VegaWidget from "@/components/VegaWidget.vue";
import type { KeyLayoutSize } from "@/models";
import { useReportsStore } from "@/stores/reports";

const reportsStore = useReportsStore();
const items = computed(() => {
  const report = reportsStore.report;
  const layout = reportsStore.layout;

  return layout.map(({ key, size }) => {
    const item = report?.get(key);
    return { key, size, data: item?.data, type: item?.type };
  });
});

function onLayoutChange(key: string, size: KeyLayoutSize) {
  reportsStore.setKeyLayoutSize(key, size);
}

function onCardRemoved(key: string) {
  reportsStore.hideKey(key);
}
</script>

<template>
  <div class="canvas">
    <DataStoreCard
      v-for="{ key, size, data, type } in items"
      :key="key"
      :title="key"
      :class="size"
      class="canvas-element"
      @layout-changed="onLayoutChange(key, $event)"
      @card-removed="onCardRemoved(key)"
    >
      <VegaWidget v-if="type === 'vega'" :spec="data" />
      <DataFrameWidget v-if="type === 'dataframe'" :columns="data.columns" :data="data.data" />
      <ImageWidget
        v-if="type === 'image'"
        :mime-type="data['mime-type']"
        :base64-src="data.data"
        :alt="key"
      />
      <ImageWidget
        v-if="type === 'matplotlib_figure'"
        mime-type="image/svg+xml"
        :base64-src="data"
        :alt="key"
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
            'numpy_array',
          ].includes(type!)
        "
        :source="data"
      />
      <HtmlSnippetWidget v-if="type === 'html'" :src="data" />
      <CrossValidationResultsWidget
        v-if="type === 'cv_results'"
        :roc_curve_spec="data.roc_curve_spec"
        :cv_results_table="data.cv_results_table"
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
