<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useRoute } from "vue-router";

import { type Mander } from "../models";
import { fetchAllManderPaths, fetchMander } from "../services/api";
import FileTree, { type FileTreeNode } from "../components/FileTree.vue";
import ManderView from "./ManderView.vue";

const route = useRoute();
const manderPaths = ref<string[]>([]);
const mander = ref<Mander>();

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

watch(
  () => route.params.slug,
  async (newSlug) => {
    const path = Array.isArray(newSlug) ? newSlug.join("/") : newSlug;
    const m = await fetchMander(path);
    if (m) {
      mander.value = m;
    }
    // TODO handle error
  }
);

manderPaths.value = await fetchAllManderPaths();
</script>

<template>
  <main>
    <nav>
      <FileTree :nodes="pathsAsFileTreeNodes" />
    </nav>
    <article>
      <ManderView v-if="mander" :mander="mander" />
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
  }
}
</style>
