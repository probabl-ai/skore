<script setup lang="ts">
import { computed, inject, ref } from "vue";

import { type TreeAccordionNode } from "@/components/TreeAccordion.vue";

const props = defineProps<TreeAccordionNode>();

const isCollapsed = ref(false);
const isDraggable = ref(false);
const emitItemAction = inject<(action: string, itemName: string) => void>("emitItemAction");

const hasChildren = computed(() => props.children?.length);
const label = computed(() => {
  const segment = props.name.split("/");
  return segment[segment.length - 1];
});

function toggleChildren() {
  if (hasChildren.value) {
    isCollapsed.value = !isCollapsed.value;
  }
}

function countChildrenRecursively(node: TreeAccordionNode): number {
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

function onDragStart(event: DragEvent) {
  if (event.dataTransfer) {
    event.dataTransfer.setData("application/x-skore-item-name", props.name);
  }
}

function onAction(action: string) {
  emitItemAction && emitItemAction(action, props.name);
}
</script>

<template>
  <div
    class="tree-accordion-item"
    :class="{ first: isRoot }"
    :style="{ '--children-count': totalChildrenCount }"
    :data-name="props.name"
  >
    <div class="label-container" @click="toggleChildren">
      <div
        class="label"
        :class="{ 'has-children': hasChildren }"
        :draggable="isDraggable && !hasChildren"
        @dragstart="onDragStart($event)"
        @mousedown="isDraggable = true"
        @mouseup="isDraggable = false"
        @mouseleave="isDraggable = false"
      >
        <span class="children-indicator icon-branch" v-if="!isRoot" />
        <span class="icon icon-pill" />
        <span class="text">{{ label }}</span>
      </div>
      <div class="actions">
        <button
          v-for="(action, index) in props.actions"
          :key="index"
          @click="onAction(action.actionName)"
        >
          <span :class="action.icon" />
        </button>
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
        <TreeAccordionItem
          v-for="(child, index) in props.children"
          :key="index"
          :name="child.name"
          :children="child.children"
          :is-root="false"
          :actions="child.actions"
        />
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.tree-accordion-item {
  --label-height: 17px;

  position: relative;
  overflow: hidden;
  margin-left: 19px;
  cursor: pointer;

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
    background-color: var(--background-color-normal);

    .label {
      display: flex;
      height: var(--label-height);
      flex-direction: row;
      align-items: center;
      transition: background-color var(--transition-duration) var(--transition-easing);

      &:not(.has-children):hover {
        background-color: var(--background-color-elevated);
        color: var(--text-color-highlight);
      }

      &[draggable="true"] {
        cursor: grabbing;
      }

      & .children-indicator {
        color: var(--text-color-normal);
      }

      & .icon {
        margin-right: var(--spacing-gap-small);
        color: var(--color-blue);
      }

      & .text {
        border-radius: var(--border-radius);
        color: var(--text-color-highlight);
        font-size: var(--text-size-title);
        font-weight: var(--text-weight-normal);
      }

      &.has-children {
        & .icon {
          opacity: 0.4;
        }
      }
    }

    .actions {
      display: flex;
      flex-direction: row;
      align-items: center;
      gap: var(--spacing-gap-small);
      opacity: 0;
      transition: opacity var(--transition-duration) var(--transition-easing);

      & button {
        padding: 0;
        border: none;
        margin: 0;
        background-color: transparent;
        color: var(--text-color-normal);
        cursor: pointer;
        font-size: var(--text-size-normal);
        transition: color var(--transition-duration) var(--transition-easing);

        &:hover {
          color: var(--color-primary);
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

    &:hover {
      .actions {
        opacity: 1;
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
