<script lang="ts">
import { ref, watch } from "vue";

export interface TreeAccordionItemAction {
  icon: string;
  actionName: string;
}

export interface TreeAccordionNode {
  name: string;
  children?: TreeAccordionNode[];
  isRoot?: boolean;
  actions?: TreeAccordionItemAction[];
}
</script>

<script setup lang="ts">
import TreeAccordionItem from "./TreeAccordionItem.vue";

const props = defineProps<{
  nodes: TreeAccordionNode[];
}>();
const emit = defineEmits<{
  itemAction: [action: string, itemName: string];
}>();
const lastItemAction = ref<string | null>(null);
const lastItemName = ref<string | null>(null);

watch([lastItemAction, lastItemName], () => {
  if (lastItemAction.value && lastItemName.value) {
    emit("itemAction", lastItemAction.value, lastItemName.value);
  }
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
      v-model:last-item-action="lastItemAction"
      v-model:last-item-name="lastItemName"
    />
  </div>
</template>

<style scoped>
.accordion {
  display: flex;
  flex-direction: column;
  background-color: var(--background-color-elevated);
  gap: var(--spacing-gap-large);
}
</style>
