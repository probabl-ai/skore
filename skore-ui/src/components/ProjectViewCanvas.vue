<script setup lang="ts">
import { formatDistance } from "date-fns";
import { computed } from "vue";

import DataFrameWidget from "@/components/DataFrameWidget.vue";
import HtmlSnippetWidget from "@/components/HtmlSnippetWidget.vue";
import ImageWidget from "@/components/ImageWidget.vue";
import MarkdownWidget from "@/components/MarkdownWidget.vue";
import PlotlyWidget from "@/components/PlotlyWidget.vue";
import ProjectViewCard from "@/components/ProjectViewCard.vue";
import VegaWidget from "@/components/VegaWidget.vue";
import { useProjectStore } from "@/stores/project";

const props = defineProps({
  showCardActions: {
    type: Boolean,
    default: true,
  },
});

const projectStore = useProjectStore();

const visibleItems = computed(() => {
  const items = [];
  let index = 0;
  if (projectStore.items !== null && projectStore.currentView !== null) {
    const v = projectStore.views[projectStore.currentView];
    for (const key of v) {
      const item = projectStore.items[key];
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
          if (mediaType.includes("json")) {
            data = JSON.parse(data);
          }
        }
        const createdAt = new Date(item.created_at);
        const updatedAt = new Date(item.updated_at);
        items.push({
          key,
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

function onCardRemoved(key: string) {
  if (projectStore.currentView) {
    projectStore.hideKey(projectStore.currentView, key);
  }
}

function getItemSubtitle(created_at: Date, updated_at: Date) {
  const now = new Date();
  return `Created ${formatDistance(created_at, now)} ago, updated ${formatDistance(updated_at, now)} ago`;
}
</script>

<template>
  <div class="canvas">
    <ProjectViewCard
      v-for="{ key, mediaType, data, createdAt, updatedAt } in visibleItems"
      :key="key"
      :title="key.toString()"
      :subtitle="getItemSubtitle(createdAt, updatedAt)"
      :showActions="props.showCardActions"
      class="canvas-element"
      @card-removed="onCardRemoved(key)"
    >
      <DataFrameWidget
        v-if="mediaType === 'application/vnd.dataframe+json'"
        :index="data.index"
        :columns="data.columns"
        :data="data.data"
        :index-names="data.index_names"
      />
      <ImageWidget
        v-if="['image/svg+xml', 'image/png', 'image/jpeg', 'image/webp'].includes(mediaType)"
        :mediaType="mediaType"
        :base64-src="data"
        :alt="key.toString()"
      />
      <MarkdownWidget v-if="mediaType === 'text/markdown'" :source="data" />
      <VegaWidget v-if="mediaType === 'application/vnd.vega.v5+json'" :spec="data" />
      <PlotlyWidget v-if="mediaType === 'application/vnd.plotly.v1+json'" :spec="data" />
      <HtmlSnippetWidget
        v-if="mediaType === 'application/vnd.sklearn.estimator+html'"
        :src="data"
      />
      <HtmlSnippetWidget v-if="mediaType === 'text/html'" :src="data" />
    </ProjectViewCard>
  </div>
</template>

<style scoped>
.canvas {
  display: flex;
  flex-direction: column;
  padding: var(--spacing-gap-normal);
  background-color: var(--background-color-normal);
  gap: var(--spacing-gap-normal);
  transition: grid-column var(--transition-duration) var(--transition-easing);

  & .canvas-element {
    max-width: 100%;
  }
}
</style>
