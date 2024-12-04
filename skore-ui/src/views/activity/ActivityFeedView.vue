<script setup lang="ts">
import Simplebar from "simplebar-vue";
import { onBeforeUnmount, ref } from "vue";

import DataFrameWidget from "@/components/DataFrameWidget.vue";
import HtmlSnippetWidget from "@/components/HtmlSnippetWidget.vue";
import ImageWidget from "@/components/ImageWidget.vue";
import MarkdownWidget from "@/components/MarkdownWidget.vue";
import PlotlyWidget from "@/components/PlotlyWidget.vue";
import VegaWidget from "@/components/VegaWidget.vue";
import type { ProjectItem } from "@/models";
import { fetchActivityFeed } from "@/services/api";
import { poll } from "@/services/utils";
import ActivityFeedCardHeader from "@/views/activity/ActivityFeedCardHeader.vue";
import ActivityFeedCurvedArrow from "@/views/activity/ActivityFeedCurvedArrow.vue";

type PresentableItem = ProjectItem & { icon: string };

const items = ref<PresentableItem[]>([]);
let lastFetchTime = new Date(1, 1, 1, 0, 0, 0, 0);

async function fetch() {
  const now = new Date();
  const feed = await fetchActivityFeed(lastFetchTime.toISOString());
  lastFetchTime = now;
  if (feed !== null) {
    const newItems = feed.map((i) => ({
      ...i,
      icon: i.media_type.startsWith("text") ? "icon-pill" : "icon-playground",
    }));
    items.value.unshift(...newItems);
  }
}

const stopPolling = await poll(fetch, 1000);

onBeforeUnmount(() => {
  stopPolling();
});
</script>

<template>
  <main>
    <Simplebar class="scroll">
      <div class="activity-feed">
        <h1>Activity feed</h1>
        <h2>Find all your activity, right below.</h2>
        <Transition name="fade" mode="out-in">
          <div class="items" v-if="items.length > 0">
            <div
              class="item"
              v-for="({ icon, name, updated_at, media_type, value }, i) in items"
              :key="updated_at"
            >
              <ActivityFeedCurvedArrow :has-arrow="i === 0" />
              <ActivityFeedCardHeader :icon="icon" :datetime="updated_at" :name="name" />
              <DataFrameWidget
                v-if="media_type.startsWith('application/vnd.dataframe+json')"
                :columns="value.columns"
                :data="value.data"
                :index="value.index"
                :index-names="value.indexNames"
              />
              <ImageWidget
                v-if="media_type.startsWith('image/')"
                :mediaType="media_type"
                :base64-src="value"
                :alt="name"
              />
              <MarkdownWidget v-if="media_type.startsWith('text/markdown')" :source="value" />
              <VegaWidget
                v-if="media_type.startsWith('application/vnd.vega.v5+json')"
                :spec="value"
              />
              <PlotlyWidget
                v-if="media_type.startsWith('application/vnd.plotly.v1+json')"
                :spec="value"
              />
              <HtmlSnippetWidget
                v-if="media_type.startsWith('application/vnd.sklearn.estimator+html')"
                :src="value"
              />
              <HtmlSnippetWidget v-if="media_type.startsWith('text/html')" :src="value" />
            </div>
          </div>
          <div class="placeholder" v-else>
            <div class="inner">No crumbs detected yet, we will add crumbs automatically.</div>
          </div>
        </Transition>
      </div>
    </Simplebar>
  </main>
</template>

<style scoped>
.scroll {
  max-height: 100dvh;
}

.activity-feed {
  display: flex;
  min-height: 100dvh;
  flex-direction: column;
  padding: var(--spacing-24) 11%;

  & h1 {
    color: var(--color-text-primary);
    font-size: var(--font-size-lg);
    font-weight: var(--font-weight-medium);
    letter-spacing: var(--letter-spacing);
    word-wrap: break-word;
  }

  & h2 {
    color: var(--color-text-secondary);
    font-size: var(--font-size-xs);
    font-weight: var(--font-weight-regular);
    letter-spacing: var(--letter-spacing);
    word-wrap: break-word;
  }

  & .simplebar-content {
    height: 100%;
  }

  & .items {
    display: flex;
    flex-direction: column;
    margin: var(--spacing-24) 0;
    gap: var(--spacing-20);

    & .item {
      position: relative;

      & .arrow {
        position: absolute;
        bottom: 0;
        left: -21px;
        overflow: visible;
        height: calc(100% + var(--spacing-20) + 20px);
      }

      &:first-child {
        & .arrow {
          height: 100%;
        }
      }
    }
  }

  & .placeholder {
    display: flex;
    flex: 1;
    align-items: center;
    justify-content: center;

    & .inner {
      padding-top: 189px;
      background-position: center;
      background-repeat: no-repeat;
      background-size: 189px 180px;
      color: var(--color-text-secondary);
      font-size: var(--font-size-xs);
      font-weight: var(--font-weight-regular);
      letter-spacing: var(--letter-spacing);
      text-align: center;
    }
  }
}

@media (prefers-color-scheme: light) {
  .activity-feed .placeholder .inner {
    background-image: url("../../assets/images/activity-feed-placeholder-light.svg");
  }
}

@media (prefers-color-scheme: dark) {
  .activity-feed .placeholder .inner {
    background-image: url("../../assets/images/activity-feed-placeholder-dark.svg");
  }
}
</style>
