<script setup lang="ts">
import DataStoreKey from "@/components/DataStoreKey.vue";

import { useReportsStore } from "@/stores/reports";

const reportsStore = useReportsStore();
const props = defineProps<{ icon: string; title: string; keys: string[] }>();
</script>

<template>
  <div class="data-store-list-item">
    <h2><span :class="props.icon"></span>{{ props.title }}</h2>
    <div class="keys">
      <DataStoreKey
        v-for="key in props.keys"
        :key="key"
        :item-key="key"
        :metadata="reportsStore.report?.get(key).metadata"
      />
    </div>
  </div>
</template>

<style scoped>
.data-store-list-item {
  border-bottom: 1px solid var(--border-color-normal);
  background-color: var(--background-color-normal);

  & h2 {
    padding: var(--spacing-padding-small) var(--spacing-padding-large);
    border-bottom: var(--border-color-normal) 1px solid;
    color: var(--text-color-title);
    font-size: var(--text-size-title);
    font-weight: var(--text-weight-title);

    & > span {
      margin-right: 4px;
    }
  }

  & .keys {
    display: flex;
    box-sizing: border-box;
    flex: none;
    flex-direction: column;
    flex-grow: 1;
    align-items: flex-start;
    align-self: stretch;
    padding:
      var(--spacing-padding-normal) var(--spacing-padding-large),
      var(--spacing-padding-large),
      var(--spacing-padding-large);
    padding: 16px 24px 24px;
    border-right: var(--border-color-normal) 1px solid;
    background-color: var(--background-color-elevated);
    gap: 10px;
  }
}
</style>
