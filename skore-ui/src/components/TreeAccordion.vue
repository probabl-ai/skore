<script lang="ts">
export interface TreeAccordionItemAction {
  icon: string;
  actionName: string;
  enabled: boolean;
}

export interface TreeAccordionNode {
  name: string;
  enabled: boolean;
  children?: TreeAccordionNode[];
  isRoot?: boolean;
  actions?: TreeAccordionItemAction[];
}
</script>

<script setup lang="ts">
import { provide } from "vue";
import TreeAccordionItem from "./TreeAccordionItem.vue";

const props = defineProps<{
  nodes: TreeAccordionNode[];
}>();
const emit = defineEmits<{
  itemAction: [action: string, itemName: string];
}>();

provide("emitItemAction", (action: string, itemName: string) => {
  emit("itemAction", action, itemName);
});
</script>

<template>
  <div class="accordion">
    <TreeAccordionItem
      v-for="(node, index) in props.nodes"
      :key="index"
      :name="node.name"
      :children="node.children"
      :is-root="true"
      :actions="node.actions"
      :enabled="node.enabled"
    />
  </div>
</template>

<style scoped>
.accordion {
  display: flex;
  flex-direction: column;
  background-color: var(--color-background-primary);
  gap: var(--spacing-20);
}
</style>
