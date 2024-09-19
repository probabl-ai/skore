<script setup lang="ts">
import { formatDistance } from "date-fns";
import { computed } from "vue";

import DataFrameWidget from "@/components/DataFrameWidget.vue";
import HtmlSnippetWidget from "@/components/HtmlSnippetWidget.vue";
import ImageWidget from "@/components/ImageWidget.vue";
import MarkdownWidget from "@/components/MarkdownWidget.vue";
import ReportCard from "@/components/ReportCard.vue";
import VegaWidget from "@/components/VegaWidget.vue";
import type { KeyLayoutSize, KeyMoveDirection } from "@/models";
import { useReportStore } from "@/stores/report";

const reportStore = useReportStore();

const visibleItems = computed(() => {
  const items = [];
  let index = 0;
  if (reportStore.items !== null) {
    for (const { key, size } of reportStore.layout) {
      const item = reportStore.items[key];
      if (item) {
        const mediaType = item.media_type || "";
        let data;
        if (
          [
            "text/markdown",
            "application/vnd.dataframe+json",
            "application/vnd.sklearn.estimator+html",
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/svg+xml",
          ].includes(mediaType)
        ) {
          data = item.value;
        } else {
          data = atob(item.value);
          if (mediaType === "application/vnd.vega.v5+json") {
            data = JSON.parse(data);
          }
        }
        const createdAt = new Date(item.created_at);
        const updatedAt = new Date(item.updated_at);
        items.push({
          key,
          size,
          mediaType,
          data,
          createdAt,
          updatedAt,
          index,
        });
        index++;
      }
    }
  }
  return items;
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

function getItemSubtitle(created_at: Date, updated_at: Date) {
  const now = new Date();
  return `Created ${formatDistance(created_at, now)} ago, updated ${formatDistance(updated_at, now)} ago`;
}
</script>

<template>
  <div class="canvas">
    <ReportCard
      v-for="{ key, size, mediaType, data, createdAt, updatedAt, index } in visibleItems"
      :key="key"
      :title="key.toString()"
      :subtitle="getItemSubtitle(createdAt, updatedAt)"
      :showButtons="props.showCardButtons"
      :can-move-up="index > 0"
      :can-move-down="index < reportStore.layout.length - 1"
      :class="size"
      class="canvas-element"
      @layout-changed="onLayoutChange(key.toString(), $event)"
      @position-changed="onPositionChanged(key.toString(), $event)"
      @card-removed="onCardRemoved(key.toString())"
    >
      <DataFrameWidget
        v-if="mediaType === 'application/vnd.dataframe+json'"
        :columns="data.columns"
        :data="data.data"
      />
      <!-- <div v-if="mediaType === 'application/vnd.dataframe+json'">
        <div style=" width: 1500px; height:10px; background-color: red"></div>
      </div> -->
      <ImageWidget
        v-if="['image/svg+xml', 'image/png', 'image/jpeg', 'image/webp'].includes(mediaType)"
        :mediaType="mediaType"
        :base64-src="data"
        :alt="key.toString()"
      />
      <MarkdownWidget v-if="mediaType === 'text/markdown'" :source="data" />
      <VegaWidget v-if="mediaType === 'application/vnd.vega.v5+json'" :spec="data" />
      <HtmlSnippetWidget
        v-if="mediaType === 'application/vnd.sklearn.estimator+html'"
        :src="data"
      />
      <HtmlSnippetWidget v-if="mediaType === 'text/html'" :src="data" />
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
