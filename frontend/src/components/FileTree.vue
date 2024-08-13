<script lang="ts">
export interface FileTreeNode {
  uri: string;
  children?: FileTreeNode[];
  indentationLevel?: number;
}

export function transformUrisToTree(uris: string[]) {
  const tree: FileTreeNode[] = [];
  for (let p of uris) {
    const segments = p.split("/").filter((s) => s.length > 0);
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
}
</script>

<script setup lang="ts">
import FileTreeItem from "./FileTreeItem.vue";

const props = defineProps<{ nodes: FileTreeNode[] }>();
</script>

<template>
  <div class="file-tree">
    <FileTreeItem
      v-for="(node, index) in props.nodes"
      :key="index"
      :uri="node.uri"
      :children="node.children"
      :indentation-level="0"
    />
  </div>
</template>

<style scoped>
.file-tree {
  padding: var(--spacing-padding-large);
  border-left: var(--border-size-small) solid var(--border-color-normal);
  background-color: var(--background-color-normal);
}
</style>
