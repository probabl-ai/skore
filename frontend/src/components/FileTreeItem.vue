<script setup lang="ts">
import { computed } from "vue";
import { type FileTreeNode } from "./FileTree.vue";
import FolderIcon from "./icons/FolderIcon.vue";
import ManderIcon from "./icons/ManderIcon.vue";

const props = defineProps<FileTreeNode>();

const hasChildren = computed(() => props.children?.length);
</script>

<template>
  <div class="file-tree-item" :style="`--indentation-level: ${indentationLevel};`">
    <div class="label">
      <div class="icon">
        <FolderIcon v-if="hasChildren" />
        <ManderIcon v-else />
      </div>
      <div class="text">{{ props.label }}</div>
    </div>
    <div class="children" v-if="hasChildren">
      <FileTreeItem
        v-for="(child, index) in props.children"
        :key="index"
        :label="child.label"
        :children="child.children"
        :indentation-level="(indentationLevel ?? 0) + 1"
      />
    </div>
  </div>
</template>

<style>
.file-tree-item {
  margin-left: calc(8px * var(--indentation-level));

  .label {
    display: flex;
    flex-direction: row;
    align-items: center;
    padding: 3px;
    border-radius: 4px;
    cursor: pointer;
    transition: var(--transition);

    &:hover {
      background-color: #ddd;
    }

    & .icon {
      display: flex;
      width: 24px;
      height: 24px;
      align-items: center;
    }
  }
}
</style>
