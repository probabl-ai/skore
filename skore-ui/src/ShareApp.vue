<script setup lang="ts">
import { formatDistance } from "date-fns";
import Simplebar from "simplebar-vue";

import DataFrameWidget from "@/components/DataFrameWidget.vue";
import HtmlSnippetWidget from "@/components/HtmlSnippetWidget.vue";
import ImageWidget from "@/components/ImageWidget.vue";
import MarkdownWidget from "@/components/MarkdownWidget.vue";
import PlotlyWidget from "@/components/PlotlyWidget.vue";
import ProjectViewCard from "@/components/ProjectViewCard.vue";
import VegaWidget from "@/components/VegaWidget.vue";
import { useProjectStore } from "@/stores/project";

const projectStore = useProjectStore();

function getItemSubtitle(created_at: Date, updated_at: Date) {
  const now = new Date();
  return `Created ${formatDistance(created_at, now)} ago, updated ${formatDistance(updated_at, now)} ago`;
}
</script>

<template>
  <div class="share">
    <div class="share-header">
      <h1>{{ projectStore.currentView }}</h1>
    </div>
    <Simplebar class="cards">
      <div class="inner">
        <ProjectViewCard
          v-for="{ key, mediaType, data, createdAt, updatedAt } in projectStore.currentViewItems"
          :key="key"
          :title="key.toString()"
          :subtitle="getItemSubtitle(createdAt, updatedAt)"
          :showActions="false"
        >
          <DataFrameWidget
            v-if="mediaType === 'application/vnd.dataframe+json'"
            :columns="data.columns"
            :data="data.data"
            :index="data.index"
            :indexNames="data.indexNames"
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
    </Simplebar>
  </div>
</template>

<style scoped>
@media (prefers-color-scheme: dark) {
  main {
    --editor-placeholder-image: url("./assets/images/editor-placeholder-dark.svg");
  }
}

@media (prefers-color-scheme: light) {
  main {
    --editor-placeholder-image: url("./assets/images/editor-placeholder-light.svg");
  }
}

.share {
  display: flex;
  height: 100dvh;
  max-height: 100vh;
  flex-direction: column;
  flex-grow: 1;

  & .share-header {
    display: flex;
    height: 44px;
    align-items: center;
    padding: var(--spacing-padding-large);
    border-right: solid var(--border-width-normal) var(--border-color-normal);
    border-bottom: solid var(--border-width-normal) var(--border-color-normal);
    background-color: var(--background-color-elevated);

    & h1 {
      flex-grow: 1;
      color: var(--text-color-normal);
      font-size: var(--text-size-title);
      font-weight: var(--text-weight-title);
      text-align: center;
    }
  }

  & .cards {
    height: 0;
    flex-grow: 1;
    padding: var(--spacing-padding-large);

    & .inner {
      display: flex;
      flex-direction: column;
      gap: var(--spacing-gap-large);
    }
  }
}
</style>
