<script setup lang="ts">
import Simplebar from "simplebar-vue";

import DataFrameWidget from "@/components/DataFrameWidget.vue";
import HtmlSnippetWidget from "@/components/HtmlSnippetWidget.vue";
import ImageWidget from "@/components/ImageWidget.vue";
import MarkdownWidget from "@/components/MarkdownWidget.vue";
import PlotlyWidget from "@/components/PlotlyWidget.vue";
import VegaWidget from "@/components/VegaWidget.vue";
import type { PresentableItem } from "@/stores/project";
import ActivityFeedCardHeader from "@/views/activity/ActivityFeedCardHeader.vue";

type Fake = Partial<PresentableItem & { name: string; datetime: string; icon: string }>;
const fakes: Fake[] = [
  {
    name: "mlkj",
    datetime: "2019-02-11T03:27:21+01:00",
    icon: "icon-pill",
    mediaType: "text/markdown",
    data: "# hello\n- lorem\n- ipsum",
  },
  {
    name: "azert",
    datetime: "2019-04-27T10:25:22+01:00",
    icon: "icon-playground",
    mediaType: "text/markdown",
    data: "# hello\n- lorem\n- ipsum",
  },
  {
    name: "cvbn",
    datetime: "2019-08-01T21:13:28+01:00",
    icon: "icon-pill",
    mediaType: "text/markdown",
    data: "# hello\n- lorem\n- ipsum",
  },
  {
    name: "poiu",
    datetime: "2019-06-16T20:58:12+01:00",
    icon: "icon-pill",
    mediaType: "text/markdown",
    data: "# hello\n- lorem\n- ipsum",
  },
  {
    name: "qsdf",
    datetime: "2019-02-11T03:27:21+01:00",
    icon: "icon-pill",
    mediaType: "text/markdown",
    data: "# hello\n- lorem\n- ipsum",
  },
  {
    name: "vghui",
    datetime: "2019-06-21T22:23:42+01:00",
    icon: "icon-pill",
    mediaType: "text/markdown",
    data: "# hello\n- lorem\n- ipsum",
  },
  {
    name: "pokjnb",
    datetime: "2019-06-21T22:23:49+01:00",
    icon: "icon-pill",
    mediaType: "text/html",
    data: "<h1>yooo</h1>",
  },
  {
    name: "pokjnb",
    datetime: "2019-06-21T22:23:49+01:00",
    icon: "icon-pill",
    mediaType: "application/vnd.dataframe+json",
    data: {
      index: [],
      columns: ["A", "B", "C", "D"],
      data: [
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
      ],
      indexNames: [],
    },
  },
];
</script>

<template>
  <main class="activity-feed">
    <Simplebar class="activity-feed-container">
      <h1>Activity feed</h1>
      <h2>Find all your activity, right below.</h2>
      <div class="items">
        <div class="item" v-for="({ icon, datetime, name, mediaType, data }, i) in fakes" :key="i">
          <ActivityFeedCardHeader :icon="icon!" :datetime="datetime!" :name="name!" />
          <DataFrameWidget
            v-if="mediaType!.startsWith('application/vnd.dataframe+json')"
            :columns="data.columns"
            :data="data.data"
            :index="data.index"
            :index-names="data.indexNames"
          />
          <ImageWidget
            v-if="mediaType!.startsWith('image/')"
            :mediaType="mediaType!"
            :base64-src="data"
            :alt="name"
          />
          <MarkdownWidget v-if="mediaType!.startsWith('text/markdown')" :source="data" />
          <VegaWidget v-if="mediaType!.startsWith('application/vnd.vega.v5+json')" :spec="data" />
          <PlotlyWidget
            v-if="mediaType!.startsWith('application/vnd.plotly.v1+json')"
            :spec="data"
          />
          <HtmlSnippetWidget
            v-if="mediaType!.startsWith('application/vnd.sklearn.estimator+html')"
            :src="data"
          />
          <HtmlSnippetWidget v-if="mediaType!.startsWith('text/html')" :src="data" />
        </div>
      </div>
    </Simplebar>
  </main>
</template>

<style scoped>
.activity-feed {
  & .activity-feed-container {
    height: 100dvh;
    padding: var(--spacing-24) 11%;

    h1 {
      color: var(--color-text-primary);
      font-size: var(--font-size-lg);
      font-weight: var(--font-weight-medium);
      letter-spacing: var(--letter-spacing);
      word-wrap: break-word;
    }

    h2 {
      color: var(--color-text-secondary);
      font-size: var(--font-size-xs);
      font-weight: var(--font-weight-regular);
      letter-spacing: var(--letter-spacing);
      word-wrap: break-word;
    }

    .items {
      display: flex;
      flex-direction: column;
      margin: var(--spacing-24) 0;
      gap: var(--spacing-20);
    }
  }
}
</style>
