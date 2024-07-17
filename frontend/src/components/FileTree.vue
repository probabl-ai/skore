<script lang="ts">
export interface FileTreeNode {
  path: string;
  children?: FileTreeNode[];
  indentationLevel?: number;
}
</script>

<script setup lang="ts">
import FileTreeItem from "./FileTreeItem.vue";

const props = defineProps<{ nodes: FileTreeNode[] }>();
defineEmits<{
  onItemClick: [nodePath: string];
}>();
</script>

<template>
  <FileTreeItem
    v-for="(node, index) in props.nodes"
    :key="index"
    :path="node.path"
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
