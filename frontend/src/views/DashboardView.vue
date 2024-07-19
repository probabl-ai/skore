<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useRoute } from "vue-router";

import DataStoreCanvas from "@/components/DataStoreCanvas.vue";
import DataStoreItemList from "@/components/DataStoreItemList.vue";
import FileTree, { type FileTreeNode } from "@/components/FileTree.vue";
import { type DataStore } from "@/models";
import { fetchAllManderUris, fetchMander } from "@/services/api";
import { useCanvasStore } from "@/stores/canvas";

const route = useRoute();
const dataStoreUris = ref<string[]>([]);
const dataStore = ref<DataStore | null>();
const canvasStore = useCanvasStore();

const pathsAsFileTreeNodes = computed(() => {
  const tree: FileTreeNode[] = [];
  for (let p of dataStoreUris.value) {
    const segments = p.split("/");
    const rootSegment = segments[0];
    let currentNode = tree.find((n) => n.uri == rootSegment);
    if (!currentNode) {
      currentNode = { uri: rootSegment };
      tree.push(currentNode);
    }
    let n = currentNode!;
    for (let s of segments.slice(1)) {
      n.children = n.children || [];
      const uri = `${n.uri}/${s}`;
      let childNode = n.children.find((n) => n.uri == uri);
      if (!childNode) {
        childNode = { uri };
        n.children.push(childNode);
      }
      n = childNode;
    }
  }
  return tree;
});

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
      <FileTree :nodes="pathsAsFileTreeNodes" />
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
