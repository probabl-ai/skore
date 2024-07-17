<script setup lang="ts">
import { computed } from "vue";
import FileTree, { type FileTreeNode } from "@/components/FileTree.vue";
import { getAllManderPaths } from "@/services/api";

const manderPaths = await getAllManderPaths();
const pathsAsFileTreeNodes = computed(() => {
  const tree: FileTreeNode[] = [];
  for (let p of manderPaths) {
    const slugs = p.split("/");
    const rootSlug = slugs[0];
    let currentNode = tree.find((n) => n.label == rootSlug);
    if (!currentNode) {
      currentNode = { label: rootSlug };
      tree.push(currentNode);
    }
    // Force the compiler to understand that currentNode can't be undefined at this point
    let n = currentNode!;
    for (let s of slugs.slice(1)) {
      n.children = n.children || [];
      let childNode = n.children.find((n) => n.label == s);
      if (!childNode) {
        childNode = { label: s };
        n.children.push(childNode);
      }
      n = childNode;
    }
  }

  return tree;
});
</script>

<template>
  <div class="dashboard">
    <FileTree :nodes="pathsAsFileTreeNodes"></FileTree>
  </div>
</template>
