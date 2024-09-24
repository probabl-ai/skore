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
</script>

<template>
  <div class="file-tree-item" :style="`--indentation-level: ${indentationLevel};`">
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
  margin-left: 19px;

  .label-container {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;

    .label {
      display: flex;
      height: 17px;
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
    display: flex;
    flex-direction: column;
    padding: var(--spacing-padding-small) 0 0 0;
    gap: var(--spacing-padding-small);
  }
}

.toggle-children-move,
.toggle-children-enter-active,
.toggle-children-leave-active {
  overflow: hidden;
  transition: all 3s var(--transition-easing);
}

.toggle-children-enter-from,
.toggle-children-leave-to {
  height: 0;
  opacity: 0;
}
</style>
