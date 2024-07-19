<script lang="ts">
export interface FileTreeNode {
  uri: string;
  children?: FileTreeNode[];
  indentationLevel?: number;
}
</script>

<script setup lang="ts">
import FileTreeItem from "./FileTreeItem.vue";

const props = defineProps<{ nodes: FileTreeNode[] }>();
</script>

<template>
  <FileTreeItem
    v-for="(node, index) in props.nodes"
    :key="index"
    :uri="node.uri"
    :children="node.children"
    :indentation-level="0"
  />
</template>

<style scoped>
.file-tree {
  padding: 1em;
  background-color: green;

  .tree-item {
    padding: 1em;
    background-color: yellow;
  }
}
</style>
