<script lang="ts">
export interface TreeAccordionNode {
  name: string;
  children?: TreeAccordionNode[];
  isRoot?: boolean;
}
</script>

<script setup lang="ts">
import TreeAccordionItem from "./TreeAccordionItem.vue";

const props = defineProps<{ nodes: TreeAccordionNode[] }>();

const emit = defineEmits<{ itemSelected: [key: string] }>();

function onDoubleClick(event: MouseEvent) {
  const target = event.target as HTMLElement;
  const closestNamedElement = target.closest("[data-name]") as HTMLElement;

  if (closestNamedElement) {
    emit("itemSelected", closestNamedElement.dataset.name ?? "");
  }
}
</script>

<template>
  <div class="accordion" @dblclick="onDoubleClick">
    <TreeAccordionItem
      v-for="(node, index) in props.nodes"
      :key="index"
      :name="node.name"
      :children="node.children"
      :is-root="true"
    />
  </div>
</template>

<style scoped>
.accordion {
  display: flex;
  flex-direction: column;
  background-color: var(--background-color-elevated-high);
  gap: var(--spacing-gap-large);
}
</style>
