<script setup lang="ts">
import DataFrameWidget from "@/components/DataFrameWidget.vue";
import HtmlSnippetWidget from "@/components/HtmlSnippetWidget.vue";
import ImageWidget from "@/components/ImageWidget.vue";
import MarkdownWidget from "@/components/MarkdownWidget.vue";
import ReportCard from "@/components/ReportCard.vue";
import type { KeyLayoutSize, KeyMoveDirection, SupportedImageMimeType } from "@/models";
import { useReportStore } from "@/stores/report";
import { computed } from "vue";

const reportStore = useReportStore();

const visibleItems = computed(() => {
  if (reportStore.report !== null) {
    return reportStore.layout.map((value, index) => ({
      ...value,
      ...reportStore.report![value.key],
      index,
    }));
  }
  return [];
});

function onLayoutChange(key: string, size: KeyLayoutSize) {
  reportStore.setKeyLayoutSize(key, size);
}

function onCardRemoved(key: string) {
  reportStore.hideKey(key);
}

function onPositionChanged(key: string, direction: KeyMoveDirection) {
  reportStore.moveKey(key, direction);
}

const props = defineProps({
  showCardButtons: {
    type: Boolean,
    default: true,
  },
});
</script>

<template>
  <div class="canvas">
    <ReportCard
      v-for="{ item_type, media_type, serialized, key, index } in visibleItems"
      :key="key"
      :title="key.toString()"
      :showButtons="props.showCardButtons"
      :can-move-up="index > 0"
      :can-move-down="index < reportStore.layout.length - 1"
      class="canvas-element large"
      @layout-changed="onLayoutChange(key.toString(), $event)"
      @position-changed="onPositionChanged(key.toString(), $event)"
      @card-removed="onCardRemoved(key.toString())"
    >
      <DataFrameWidget
        v-if="item_type === 'pandas_dataframe'"
        :columns="serialized.columns"
        :data="serialized.data"
      />
      <ImageWidget
        v-if="
          item_type === 'media' &&
          media_type &&
          ['image/svg+xml', 'image/png', 'image/jpeg', 'image/webp'].includes(media_type)
        "
        :mime-type="media_type as SupportedImageMimeType"
        :base64-src="serialized"
        :alt="key.toString()"
      />
      <MarkdownWidget v-if="['json', 'numpy_array'].includes(item_type)" :source="serialized" />
      <HtmlSnippetWidget
        v-if="'media' === item_type && media_type === 'text/html'"
        :src="serialized"
      />
      <HtmlSnippetWidget
        v-if="'sklearn_base_estimator' === item_type && media_type === 'text/html'"
        :src="serialized.html"
      />
    </ReportCard>
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
