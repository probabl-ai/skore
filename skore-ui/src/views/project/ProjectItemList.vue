<script setup lang="ts">
import Simplebar from "simplebar-vue";

import SectionHeader from "@/components/SectionHeader.vue";
import TreeAccordion, { type TreeAccordionNode } from "@/components/TreeAccordion.vue";
import { useProjectStore } from "@/stores/project";
import { useToastsStore } from "@/stores/toasts";
import { computed } from "vue";

const projectStore = useProjectStore();
const toastsStore = useToastsStore();

const itemsAsTree = computed(() => {
  const source = projectStore.keysAsTree();
  const tree = structuredClone(source) as unknown as TreeAccordionNode[];
  // add actions to the leaf nodes
  function addActions(node: TreeAccordionNode) {
    if (node.children?.length === 0) {
      node.actions = [{ icon: "icon-plus", actionName: "add" }];
    }
    for (const child of node.children ?? []) {
      addActions(child);
    }
  }
  for (const node of tree) {
    addActions(node);
  }
  return tree;
});

async function onItemAction(action: string, key: string) {
  if (projectStore.currentView) {
    const success = await projectStore.displayKey(projectStore.currentView, key);
    console.log(success);

    if (success) {
      // Scroll to last element
      const lastItemElement = document.querySelector(".editor-container .item:last-child");
      if (lastItemElement) {
        lastItemElement.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }
  } else {
    toastsStore.addToast("No view selected", "error");
  }
}
</script>

<template>
  <div class="keys-list">
    <SectionHeader title="Items" icon="icon-pie-chart" />
    <Simplebar class="scrollable">
      <TreeAccordion :nodes="itemsAsTree" @item-action="onItemAction" />
    </Simplebar>
  </div>
</template>

<style scoped>
.keys-list {
  display: flex;
  max-height: 100%;
  flex-direction: column;

  & .scrollable {
    height: 0;
    flex: 1;
    padding: var(--spacing-padding-large);
    background-color: var(--background-color-normal);
  }
}
</style>
