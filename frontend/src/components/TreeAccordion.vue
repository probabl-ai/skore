<script lang="ts">
export interface TreeAccordionNode {
  name: string;
  children?: TreeAccordionNode[];
  indentationLevel?: number;
}
</script>

<script setup lang="ts">
import TreeAccordionItem from "./TreeAccordionItem.vue";

const props = defineProps<{ nodes: TreeAccordionNode[] }>();
</script>

<template>
  <div class="accordion">
    <TreeAccordionItem
      v-for="(node, index) in props.nodes"
      :key="index"
      :name="node.name"
      :children="node.children"
      :indentation-level="0"
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
