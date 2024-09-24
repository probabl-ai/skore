<script lang="ts">
export interface FileTreeNode {
  stem: string;
  children?: FileTreeNode[];
  indentationLevel?: number;
}

/**
 * Transform a list of strings into a tree of FileTreeNodes.
 *
 * i.e. ["a", "a/b", "a/c", "a/b/d", "a/b/e", "a/b/f", "a/b/f/g"] ->
 * [
 *   { stem: "a", children: [
 *     { stem: "b", children: [
 *       { stem: "d" },
 *       { stem: "e" },
 *       { stem: "f", children: [{ stem: "g" }] }
 *     ] }
 *   ] }
 * ]
 *
 * @param list - A list of strings to transform into a tree.
 * @returns A tree of FileTreeNodes.
 */
export function transformListToTree(list: string[]) {
  const tree: FileTreeNode[] = [];
  for (let p of list) {
    const segments = p.split("/").filter((s) => s.length > 0);
    const rootSegment = segments[0];
    let currentNode = tree.find((n) => n.stem == rootSegment);
    if (!currentNode) {
      currentNode = { stem: rootSegment };
      tree.push(currentNode);
    }
    let n = currentNode!;
    for (let s of segments.slice(1)) {
      n.children = n.children || [];
      const stem = `${n.stem}/${s}`;
      let childNode = n.children.find((n) => n.stem == stem);
      if (!childNode) {
        childNode = { stem };
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
      :stem="node.stem"
      :children="node.children"
      :indentation-level="0"
    />
  </div>
</template>

<style scoped>
.file-tree {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-gap-large);
}
</style>
