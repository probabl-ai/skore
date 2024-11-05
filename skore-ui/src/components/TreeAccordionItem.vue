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
        :draggable="props.enabled && isDraggable && !hasChildren"
        @dragstart="onDragStart($event)"
        @mousedown="isDraggable = true"
        @mouseup="isDraggable = false"
        @mouseleave="isDraggable = false"
        @click="onAction('clicked')"
      >
        <span class="children-indicator icon-branch" v-if="!isRoot" />
        <span class="icon icon-pill" />
        <span class="text">{{ label }}</span>
      </div>
      <div class="actions" v-if="props.enabled">
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
          :enabled="child.enabled"
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
    background-color: var(--color-background-primary);

    .label {
      display: flex;
      height: var(--label-height);
      flex-direction: row;
      align-items: center;
      transition: background-color var(--animation-duration) var(--animation-easing);

      &:not(.has-children):hover {
        background-color: var(--color-background-secondary);
      }

      &[draggable="true"] {
        cursor: grabbing;
      }

      & .children-indicator {
        color: var(--color-text-secondary);
      }

      & .icon {
        margin-right: var(--spacing-4);
        color: var(--color-icon-tertiary);
      }

      & .text {
        border-radius: var(--radius-xs);
      }

      &.has-children {
        & .icon {
          color: var(--color-icon-quartenary);
        }
      }
    }

    .actions {
      display: flex;
      flex-direction: row;
      align-items: center;
      gap: var(--spacing-8);
      opacity: 0;
      transition: opacity var(--animation-duration) var(--animation-easing);

      & button {
        padding: 0;
        border: none;
        margin: 0;
        background-color: transparent;
        cursor: pointer;
        transition: color var(--animation-duration) var(--animation-easing);

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
      color: var(--color-text-secondary);
      cursor: pointer;
      transition: transform var(--animation-duration) var(--animation-easing);

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
    padding-top: var(--spacing-6);
    gap: var(--spacing-6);
  }

  & .toggle-children-move,
  & .toggle-children-enter-active,
  & .toggle-children-leave-active {
    transform-origin: top;
    transition: all var(--animation-duration) var(--animation-easing);
  }

  & .toggle-children-enter-from,
  & .toggle-children-leave-to {
    margin-top: calc((var(--label-height) + var(--spacing-6)) * var(--children-count) * -1);
    opacity: 0;
  }
}
</style>
