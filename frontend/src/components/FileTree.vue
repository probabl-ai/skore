<script lang="ts">
export interface FileTreeNode {
  label: string;
  children?: FileTreeNode[];
  indentationLevel?: number;
}
</script>

<script setup lang="ts">
import FileTreeItem from "./FileTreeItem.vue";

defineProps<{ nodes: FileTreeNode[] }>();
</script>

<template>
  <FileTreeItem
    v-for="(node, index) in nodes"
    :key="index"
    :label="node.label"
    :children="node.children"
    :indentation-level="0"
  />
</template>

<style>
.file-tree {
  padding: 1em;
  background-color: green;

  .tree-item {
    padding: 1em;
    background-color: yellow;
  }
}
</style>
