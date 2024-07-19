<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useRoute } from "vue-router";

import DataStoreCanvas from "@/components/DataStoreCanvas.vue";
import DataStoreItemList from "@/components/DataStoreItemList.vue";
import FileTree, { transformUrisToTree } from "@/components/FileTree.vue";
import { type DataStore } from "@/models";
import { fetchAllManderUris, fetchMander } from "@/services/api";
import { useCanvasStore } from "@/stores/canvas";

const route = useRoute();
const dataStoreUris = ref<string[]>([]);
const dataStore = ref<DataStore | null>();
const canvasStore = useCanvasStore();
const fileTree = computed(() => transformUrisToTree(dataStoreUris.value));

async function fetchDataStoreDetail(path: string | string[]) {
  const p = Array.isArray(path) ? path.join("/") : path;
  const m = await fetchMander(p);
  dataStore.value = m;
  canvasStore.setDataStore(m);
}

watch(
  () => route.params.segments,
  async (newSegments) => {
    await fetchDataStoreDetail(newSegments);
  }
);

dataStoreUris.value = await fetchAllManderUris();
await fetchDataStoreDetail(route.params.segments);
</script>

<template>
  <main>
    <nav>
      <FileTree :nodes="fileTree" />
    </nav>
    <article :class="{ 'not-found': dataStore == null }">
      <div class="item-list" v-if="dataStore">
        <DataStoreItemList title="Views" icon="icon-plot" :keys="Object.keys(dataStore.views)" />
        <DataStoreItemList title="Info" icon="icon-text" :keys="Object.keys(dataStore.info)" />
        <DataStoreItemList title="Logs" icon="icon-gift" :keys="Object.keys(dataStore.logs)" />
        <DataStoreItemList
          title="Logs"
          icon="icon-folder"
          :keys="Object.keys(dataStore.artifacts)"
        />
      </div>
      <div v-else>mandr not found...</div>
      <DataStoreCanvas />
    </article>
  </main>
</template>

<style scoped>
main {
  display: flex;
  flex-direction: row;

  nav,
  article {
    overflow: scroll;
    height: 100dvh;
  }

  nav {
    width: 285px;
    flex-shrink: 0;
    padding: 10px;
    border-left: 1px solid #e9e9e9;
    background-color: #f2f1f1;
  }

  article {
    display: flex;
    flex-direction: row;
    flex-grow: 1;

    &.not-found {
      display: flex;
      align-items: center;
      justify-content: center;
      color: #c8c7c7;
      font-size: x-large;
      font-weight: 200;
    }

    & .item-list {
      display: flex;
      overflow: scroll;
      width: 285px;
      height: 100dvh;
      flex-direction: column;
    }
  }
}
</style>
