<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useRoute } from "vue-router";

import FileTree, { type FileTreeNode } from "../components/FileTree.vue";
import ManderDetail from "../components/ManderDetail.vue";
import { type Mander } from "../models";
import { fetchAllManderPaths, fetchMander } from "../services/api";

const route = useRoute();
const manderPaths = ref<string[]>([]);
const mander = ref<Mander | null>();

const pathsAsFileTreeNodes = computed(() => {
  const tree: FileTreeNode[] = [];
  for (let p of manderPaths.value) {
    const slugs = p.split("/");
    const rootSlug = slugs[0];
    let currentNode = tree.find((n) => n.path == rootSlug);
    if (!currentNode) {
      currentNode = { path: rootSlug };
      tree.push(currentNode);
    }
    let n = currentNode!;
    for (let s of slugs.slice(1)) {
      n.children = n.children || [];
      const path = `${n.path}/${s}`;
      let childNode = n.children.find((n) => n.path == path);
      if (!childNode) {
        childNode = { path };
        n.children.push(childNode);
      }
      n = childNode;
    }
  }
  return tree;
});

async function fetchManderDetail(path: string | string[]) {
  console.log(path);
  const p = Array.isArray(path) ? path.join("/") : path;
  const m = await fetchMander(p);
  mander.value = m;
  console.log(m);
}
watch(
  () => route.params.slug,
  async (newSlug) => {
    await fetchManderDetail(newSlug);
  }
);

manderPaths.value = await fetchAllManderPaths();
await fetchManderDetail(route.params.slug);
</script>

<template>
  <main>
    <nav>
      <FileTree :nodes="pathsAsFileTreeNodes" />
    </nav>
    <article :class="{ 'not-found': mander == null }">
      <ManderDetail v-if="mander" :mander="mander" />
      <div v-else>mandr not found...</div>
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
    flex-grow: 1;

    &.not-found {
      display: flex;
      align-items: center;
      justify-content: center;
      color: #c8c7c7;
      font-size: x-large;
      font-weight: 200;
    }
  }
}
</style>
