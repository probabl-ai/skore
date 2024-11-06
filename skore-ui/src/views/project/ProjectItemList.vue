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
      node.actions = [{ icon: "icon-plus", actionName: "add", enabled: true }];
    }
    for (const child of node.children ?? []) {
      addActions(child);
    }
  }
  for (const node of tree) {
    addActions(node);
  }
  // add interactive status to nodes
  function addInteractiveStatus(node: TreeAccordionNode) {
    node.enabled =
      projectStore.currentView !== null
        ? !projectStore.isKeyDisplayed(projectStore.currentView, node.name)
        : false;
    for (const child of node.children ?? []) {
      addInteractiveStatus(child);
    }
  }
  for (const node of tree) {
    addInteractiveStatus(node);
  }
  return tree;
});

async function onItemAction(action: string, key: string) {
  if (projectStore.currentView) {
    switch (action) {
      case "add": {
        const success = await projectStore.displayKey(projectStore.currentView, key);

        if (success) {
          // Scroll to last element
          const lastItemElement = document.querySelector(".editor-container .item:last-child");
          if (lastItemElement) {
            lastItemElement.scrollIntoView({ behavior: "smooth", block: "start" });
          }
        }
        break;
      }
      case "clicked": {
        // if clicke item is added to the view then scroll to it
        const relatedCard = document.querySelector(`.item [data-name="${key}"]`);
        if (relatedCard) {
          relatedCard.scrollIntoView({ behavior: "smooth", block: "start" });
          relatedCard.classList.add("blink");
        }
        break;
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
    padding: var(--spacing-16);
    background-color: var(--color-background-primary);
  }
}
</style>
