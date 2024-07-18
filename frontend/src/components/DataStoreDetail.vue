<script setup lang="ts">
import { type DataStore } from "../models";
import DataStoreItemList from "./DataStoreItemList.vue";

const props = defineProps<{ dataStore: DataStore }>();
</script>

<template>
  <div class="item-list">
    <DataStoreItemList title="Views" icon="icon-plot" :elements="Object.keys(dataStore.views)" />
    <DataStoreItemList title="Info" icon="icon-text" :elements="Object.keys(dataStore.info)" />
    <DataStoreItemList title="Logs" icon="icon-gift" :elements="Object.keys(dataStore.logs)" />
    <DataStoreItemList
      title="Logs"
      icon="icon-folder"
      :elements="Object.keys(dataStore.artifacts)"
    />
  </div>
</template>

<style scoped>
.item-list {
  display: flex;
  flex-direction: column;
}
</style>
