<script setup lang="ts">
import { computed } from "vue";
import { useRouter } from "vue-router";
import { type FileTreeNode } from "./FileTree.vue";
import FolderIcon from "./icons/FolderIcon.vue";
import ManderIcon from "./icons/ManderIcon.vue";

const router = useRouter();
const props = defineProps<FileTreeNode>();

const hasChildren = computed(() => props.children?.length);
const label = computed(() => {
  const segment = props.uri.split("/");
  return segment[segment.length - 1];
});

function onClick() {
  router.push({
    name: "dashboard",
    params: {
      segments: props.uri.split("/"),
    },
  });
}
</script>

<template>
  <div class="file-tree-item" :style="`--indentation-level: ${indentationLevel};`">
    <div class="label" @click="onClick">
      <div class="icon">
        <FolderIcon v-if="hasChildren" />
        <ManderIcon v-else />
      </div>
      <div class="text">{{ label }}</div>
    </div>
    <div class="children" v-if="hasChildren">
      <FileTreeItem
        v-for="(child, index) in props.children"
        :key="index"
        :uri="child.uri"
        :children="child.children"
        :indentation-level="(indentationLevel ?? 0) + 1"
      />
    </div>
  </div>
</template>

<style scoped>
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
      font-weight: 500;
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
