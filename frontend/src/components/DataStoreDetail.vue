<script setup lang="ts">
import { type DataStore } from "../models";
import DataStoreItemList from "./DataStoreItemList.vue";

const props = defineProps<{ dataStore: DataStore }>();
</script>

<template>
  <div class="item-list">
    <DataStoreItemList
      title="Views"
      icon="icon-plot"
      :elements="Object.keys(props.dataStore.views)"
    />
    <DataStoreItemList
      title="Info"
      icon="icon-text"
      :elements="Object.keys(props.dataStore.info)"
    />
    <DataStoreItemList
      title="Logs"
      icon="icon-gift"
      :elements="Object.keys(props.dataStore.logs)"
    />
    <DataStoreItemList
      title="Artifacts"
      icon="icon-folder"
      :elements="Object.keys(props.dataStore.artifacts)"
    />
  </div>
</template>

<style scoped>
.item-list {
  display: flex;
  flex-direction: column;
}
</style>
