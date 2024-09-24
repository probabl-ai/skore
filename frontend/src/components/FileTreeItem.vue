<script setup lang="ts">
import { computed, ref } from "vue";

import { type FileTreeNode } from "@/components/FileTree.vue";

const props = defineProps<FileTreeNode>();

const isCollapsed = ref(false);

const hasChildren = computed(() => props.children?.length);
const label = computed(() => {
  const segment = props.stem.split("/");
  return segment[segment.length - 1];
});

function toggleChildren() {
  if (hasChildren.value) {
    isCollapsed.value = !isCollapsed.value;
  }
}

function countChildrenRecursively(node: FileTreeNode): number {
  if (!node.children || node.children.length === 0) {
    return 0;
  }

  return node.children.reduce((total, child) => {
    return total + 1 + countChildrenRecursively(child);
  }, 0);
}

const totalChildrenCount = computed(() => {
  return countChildrenRecursively(props);
});
</script>

<template>
  <div
    class="file-tree-item"
    :class="{ first: props.indentationLevel === 0 }"
    :style="{ '--children-count': totalChildrenCount }"
  >
    <div class="label-container" @click="toggleChildren">
      <div class="label" :class="{ 'has-children': hasChildren }">
        <span class="children-indicator icon-branch" v-if="props.indentationLevel ?? 0 > 0" />
        <span class="icon icon-pill" />
        <span class="text">{{ label }}</span>
      </div>
      <button
        class="collapse"
        :class="{ collapsed: isCollapsed }"
        aria-label="Collapse"
        v-if="hasChildren"
      >
        <span class="icon-chevron-down" />
      </button>
    </div>
    <Transition name="toggle-children">
      <div class="children" v-if="hasChildren && !isCollapsed">
        <FileTreeItem
          v-for="(child, index) in props.children"
          :key="index"
          :stem="child.stem"
          :children="child.children"
          :indentation-level="(props.indentationLevel ?? 0) + 1"
        />
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.file-tree-item {
  --label-height: 17px;

  position: relative;
  overflow: hidden;
  margin-left: 19px;

  &.first {
    margin-left: 0;
  }

  .label-container {
    position: relative;
    z-index: 2;
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    background-color: var(--background-color-elevated-high);

    .label {
      display: flex;
      height: var(--label-height);
      flex-direction: row;
      align-items: center;
      cursor: pointer;
      transition: background-color var(--transition-duration) var(--transition-easing);

      &:not(.has-children):hover {
        background-color: var(--background-color-selected);
        color: var(--text-color-highlight);
      }

      & .children-indicator {
        color: var(--text-color-normal);
      }

      & .icon {
        margin-right: var(--spacing-gap-small);
        color: var(--color-primary);
      }

      & .text {
        border-radius: var(--border-radius);
        color: var(--text-color-normal);
        font-size: var(--text-size-title);
        font-weight: var(--text-weight-normal);
      }

      &.has-children {
        & .text,
        & .icon {
          opacity: 0.4;
        }
      }
    }

    .collapse {
      padding: 0;
      border: none;
      margin: 0;
      background-color: transparent;
      color: var(--text-color-normal);
      cursor: pointer;
      transition: transform var(--transition-duration) var(--transition-easing);

      &.collapsed {
        transform: rotate(90deg);
      }
    }
  }

  .children {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    padding: var(--spacing-padding-small) 0 0 0;
    gap: var(--spacing-padding-small);
  }

  & .toggle-children-move,
  & .toggle-children-enter-active,
  & .toggle-children-leave-active {
    transform-origin: top;
    transition: all var(--transition-duration) var(--transition-easing);
  }

  & .toggle-children-enter-from,
  & .toggle-children-leave-to {
    margin-top: calc(
      (var(--label-height) + var(--spacing-padding-small)) * var(--children-count) * -1
    );
    opacity: 0;
  }
}
</style>
