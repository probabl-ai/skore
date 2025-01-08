<script setup lang="ts">
import Simplebar from "simplebar-vue";
import { onBeforeUnmount, shallowRef } from "vue";

import MediaWidgetSelector from "@/components/MediaWidgetSelector.vue";
import { deserializeProjectItemDto, type PresentableItem } from "@/models";
import { fetchActivityFeed } from "@/services/api";
import { poll } from "@/services/utils";
import ActivityFeedCardHeader from "@/views/activity/ActivityFeedCardHeader.vue";
import ActivityFeedCurvedArrow from "@/views/activity/ActivityFeedCurvedArrow.vue";

type ActivityPresentableItem = PresentableItem & { icon: string };

const items = shallowRef<ActivityPresentableItem[]>([]);
let lastFetchTime = new Date(1, 1, 1, 0, 0, 0, 0);

async function fetch() {
  const now = new Date();
  const feed = await fetchActivityFeed(lastFetchTime.toISOString());
  lastFetchTime = now;
  if (feed !== null) {
    const newItems = feed.map((i) => ({
      ...deserializeProjectItemDto(i),
      icon: i.media_type.startsWith("text") ? "icon-pill" : "icon-playground",
    }));
    items.value = [...newItems, ...items.value];
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
              v-for="(item, i) in items"
              :key="`${item.name}-${item.updatedAt.getTime()}`"
            >
              <ActivityFeedCurvedArrow :has-arrow="i === 0" />
              <ActivityFeedCardHeader
                :icon="item.icon"
                :datetime="item.updatedAt"
                :name="item.name"
              />
              <MediaWidgetSelector :item="item" />
            </div>
          </div>
          <div class="placeholder" v-else>
            <div class="inner">No items detected yet. Items will be added automatically.</div>
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
      background-image: var(--image-activity-feed-placeholder);
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
</style>
